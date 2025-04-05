# -*- coding: utf-8 -*-
#
# Author: GavinGong aka VisualDust
# Github: github.com/visualDust


import atexit
import os
import time
import warnings
from functools import wraps
from typing import Callable, Dict, Union

import torch
import torch.distributed.rpc as rpc
from loguru import logger
from torch.distributed.rpc import RRef
from vdtoys.cache import CachedDict
from vdtoys.mvc import patch

from tiberate import CkksEngine
from tiberate.typing import *


def warn_not_on_local_rank(rank: int):
    """Decorator to warn if the function is not run on the specified local rank.

    Args:
        rank (int): The local rank to check.
    """

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
        self,
        *,
        local_rank: int,
        ckks_params: dict,
        allow_sk_gen: bool = True,
        cache: dict = {},
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
        """Run a function on this worker

        Args:
            rank (int): The rank of the worker to run the function on.
            func (Callable): The function to run.
            - The first argument must be CkksEngine instance.
            - The second argument must be a dict, which is the cache of the specific worker. The changes made to the cache will be stored.
            - If the result returns any data structure that includes tensor, it should be moved to CPU before returning.

            *args: The arguments to pass to the function.
            **kwargs: The keyword arguments to pass to the function.

        Returns:
            any: The result of the function, in the form of a Future.
        """
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
        # assume the environment variables are set by torch.distributed.launch
        assert "RANK" in os.environ, "Please launch with torchrun script"
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        if world_size < 2:
            raise RuntimeError(
                f"world_size must be at least 2 (1 scheduler + 1 worker), got {world_size}"
            )
        # master_addr = os.environ["MASTER_ADDR"]
        # master_port = os.environ["MASTER_PORT"]
        self.name = f"worker{rank}" if rank > 0 else "scheduler"
        if rank > 0:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(rank - 1)  # only one GPU per worker
        rpc.init_rpc(
            name=self.name,
            rank=rank,
            world_size=world_size,
            backend=rpc.BackendType.TENSORPIPE,
        )

        if rank == 0:
            logger.info("[scheduler] Initializing engines on all workers")
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
                # read sk from worker 0
                self.sk = self._workers[0].rpc_sync().get_sk()
                for worker in self._workers[1:]:
                    worker.rpc_sync().set_sk(self.sk)

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
    def im_scheduler(self) -> bool:
        return int(os.environ["RANK"]) == 0

    @property
    def workers(self) -> List[WorkerContext]:
        """Get the list of workers.

        Returns:
            List[WorkerContext]: The list of workers.
        """
        local_rank = int(os.environ["RANK"])
        if local_rank != 0:
            raise RuntimeError(
                f"Workers are only visible to the scheduler(local rank 0). Cannot access workers from local rank {local_rank}."
            )
        return self._workers

    def __len__(self) -> int:
        """Get the number of workers.

        Returns:
            int: The number of workers.
        """
        return len(self.workers)

    def __getitem__(self, rank: int) -> WorkerContext:
        """Get the engine for a specific rank.

        Args:
            rank (int): The rank of the worker to get the engine for.

        Returns:
            CkksEngine: The engine for the specified rank.
        """
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
        rpc.shutdown()
    except Exception as e:
        pass


atexit.register(try_shutdown)  # register shutdown function on exit
