# -*- coding: utf-8 -*-
#
# Author: GavinGong aka VisualDust
# Github: github.com/visualDust

import atexit
import os
import time
import warnings
from functools import wraps
from typing import Any, Callable, Dict, List, Union

import torch
import torch.distributed.rpc as rpc
from loguru import logger
from torch.distributed.rpc import RRef
from vdtoys.cache import CachedDict
from vdtoys.mvc import patch

from tiberate import CkksEngine
from tiberate.typing import *

# Global RPC init flag
_RPC_INITIALIZED = False
_REGISTERED_WORKERS = []


def init_rpc_once():
    global _RPC_INITIALIZED
    if _RPC_INITIALIZED:
        return

    assert "RANK" in os.environ and "WORLD_SIZE" in os.environ, "Please launch with torchrun"
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    name = f"worker{rank}" if rank > 0 else "scheduler"

    if rank > 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(rank - 1)

    rpc.init_rpc(
        name=name,
        rank=rank,
        world_size=world_size,
        backend=rpc.BackendType.TENSORPIPE,
    )

    _RPC_INITIALIZED = True
    logger.info(f"[rank {rank}] RPC initialized as {name}")


IM_SCHEDULER = lambda: int(os.environ["RANK"]) == 0


def warn_not_on_local_rank(rank: int):
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            local_rank = int(os.environ["RANK"])
            if local_rank != rank:
                logger.warning(
                    f"{func.__name__} is called on local rank {local_rank}, while it is supposed to be called on local rank {rank}."
                )
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


class WorkerContext:
    def __init__(
        self, *, local_rank: int, ckks_params: dict, allow_sk_gen: bool = True, cache: dict = {}
    ):
        self.local_rank = local_rank
        self.engine = CkksEngine(ckks_params=ckks_params, allow_sk_gen=allow_sk_gen)
        self.cache = cache

    def get_sk(self) -> Union[SecretKey, torch.jit.Future]:
        return self.engine.sk.to("cpu")

    def set_sk(self, sk: SecretKey):
        self.engine.sk = sk.to(self.engine.device0)
        logger.debug(f"Worker {self.local_rank} got secret key")

    def get_pk(self) -> Union[PublicKey, torch.jit.Future]:
        return self.engine.pk.to("cpu")

    def set_pk(self, pk: PublicKey):
        self.engine.pk = pk.to(self.engine.device0)
        logger.debug(f"Worker {self.local_rank} got public key")

    def get_evk(self) -> Union[EvaluationKey, torch.jit.Future]:
        return self.engine.evk.to("cpu")

    def set_evk(self, evk: EvaluationKey):
        self.engine.evk = evk.to(self.engine.device0)
        logger.debug(f"Worker {self.local_rank} got evaluation key")

    def get_rotk(self) -> Union[CachedDict, torch.jit.Future]:
        return DataStruct.copy_tensor_to_device_recursive(self.engine.rotk, "cpu")

    def set_rotk(self, rotk: Union[CachedDict, Dict]):
        self.engine.rotk = DataStruct.copy_tensor_to_device_recursive(rotk, self.engine.device0)
        logger.debug(f"Worker {self.local_rank} got rotation key {rotk.keys()}")

    def run(self, func: Callable, *args, **kwargs) -> Union[torch.jit.Future, Any]:
        local_rank = int(os.environ["RANK"])
        logger.debug(
            f"Call {func.__name__} on local rank {local_rank}, will run on worker {self.local_rank}"
        )
        args = DataStruct.copy_tensor_to_device_recursive(args, self.engine.device0)
        kwargs = DataStruct.copy_tensor_to_device_recursive(kwargs, self.engine.device0)
        result = func(self.engine, self.cache, *args, **kwargs)
        return result


class MultiGPUEngineContext:
    def __init__(self, ckks_params: dict, allow_sk_gen: bool = True):
        init_rpc_once()

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        self.name = f"worker{rank}" if rank > 0 else "scheduler"

        if rank == 0:
            self._workers: List[RRef] = [
                rpc.remote(
                    f"worker{i}",
                    WorkerContext,
                    kwargs={
                        "local_rank": i,
                        "ckks_params": ckks_params,
                        "allow_sk_gen": allow_sk_gen,
                    },
                )
                for i in range(1, world_size)
            ]

            if allow_sk_gen and len(self._workers) > 1:
                self.sk = self._workers[0].rpc_sync().get_sk()
                for worker in self._workers[1:]:
                    worker.rpc_sync().set_sk(self.sk)
        else:
            self._context = WorkerContext(
                local_rank=rank,
                ckks_params=ckks_params,
                allow_sk_gen=allow_sk_gen,
            )

        _REGISTERED_WORKERS.append(self)

    def set_sk(self, sk: SecretKey):
        for worker in self._workers:
            worker.rpc_sync().set_sk(sk)

    def set_pk(self, pk: PublicKey):
        for worker in self._workers:
            worker.rpc_sync().set_pk(pk)

    def set_evk(self, evk: EvaluationKey):
        for worker in self._workers:
            worker.rpc_sync().set_evk(evk)

    def set_rotk(self, rotk: Union[CachedDict, Dict]):
        for worker in self._workers:
            worker.rpc_sync().set_rotk(rotk)

    @property
    def workers(self) -> List[WorkerContext]:
        local_rank = int(os.environ["RANK"])
        if local_rank != 0:
            raise RuntimeError(
                f"Workers are only visible to the scheduler(local rank 0). Cannot access workers from local rank {local_rank}."
            )
        return self._workers

    def __len__(self) -> int:
        return len(self.workers)

    def __getitem__(self, rank: int) -> WorkerContext:
        local_rank = int(os.environ["RANK"])
        if local_rank != 0:
            raise RuntimeError(
                f"Workers are only visible to the scheduler(local rank 0). Cannot index workers from local rank {local_rank}."
            )
        if not (0 <= rank < len(self._workers)):
            raise RuntimeError(
                f"Invalid rank: {rank}, available ranks: {list(range(0, len(self._workers)))}"
            )
        return self._workers[rank].rpc_async()


def try_shutdown():
    try:
        if _RPC_INITIALIZED:
            rpc.shutdown()
    except Exception as e:
        logger.warning(f"RPC shutdown failed: {e}")


atexit.register(try_shutdown)
