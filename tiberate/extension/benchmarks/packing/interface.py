#
# Author: GavinGong aka VisualDust
# Github: github.com/visualDust

from typing import Any

import torch

from tiberate import CkksEngine
from tiberate.typing import *
from tiberate.utils.massive import copy_some_datastruct, next_power_of_2


class PackingMetadata:
    def __init__(
        self,
        original_shape: (
            list[int] | None
        ) = None,  # B,C,H,W or B,C,N or H,W or N, or any other shape
        logical_num_slots: (
            int | None
        ) = None,  # todo remove this from must-have properties
        packed_by: Any = None,
        encoded_by: Any = None,
        misc: dict = {},
    ):
        self.original_shape = original_shape
        self.packed_by = packed_by
        self.encoded_by = encoded_by
        self.logical_num_slots = logical_num_slots
        self.misc = misc

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(original_shape={self.original_shape}, logical_num_slots={self.logical_num_slots}, packed_by={self.packed_by}, encoded_by={self.encoded_by}, misc={self.misc})"


class PTPacking:
    debug = False

    @classmethod
    def pack(cls, **kwargs) -> "PackedCT":
        raise NotImplementedError

    @classmethod
    def unpack(cls, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @classmethod
    def pad_to_next_power_of_2_on_dims(
        cls, x: torch.Tensor, dims: int | tuple[int] | list[int]
    ) -> torch.Tensor:
        if isinstance(dims, int):
            dims = (dims,)
        current_dims = x.shape
        target_dims = list(current_dims)
        for dim in dims:
            target_dims[dim] = next_power_of_2(current_dims[dim])
        # pad zeros to the right of the tensor, so that the tensor shape becomes target_dims
        pad_tensor = torch.zeros(target_dims, dtype=x.dtype, device=x.device)
        pad_tensor[tuple([slice(0, s) for s in current_dims])] = x
        return pad_tensor


class CTEncoding:
    debug = False

    def encodecrypt(
        cls, *, src: torch.Tensor, engine: CkksEngine
    ) -> "PackedCT":
        raise NotImplementedError

    @classmethod
    def decryptcode(
        cls,
        *,
        packed_ct: "PackedCT",
        engine: CkksEngine,
        sk: SecretKey,
    ) -> torch.Tensor:
        raise NotImplementedError


class PackedCT:
    debug = False  # class variable

    metadata: PackingMetadata
    cts: list[Ciphertext] | Dict[Any, list[Ciphertext]]

    def __init__(
        self,
        cts: list | Dict,  # ciphertext or list of ciphertexts
        metadata: PackingMetadata,
    ):
        self.metadata = metadata
        self.cts = cts

    def clone(self):
        # create a copy of cts
        cts_copy = copy_some_datastruct(self.cts)
        return self.__class__(cts=cts_copy, metadata=self.metadata)

    def to(self, device: str):
        return self.__class__(
            cts=DataStruct.copy_tensor_to_device_recursive(
                data=self.cts, device=device
            ),
            metadata=self.metadata,
        )

    def __add__(self, other):
        raise NotImplementedError

    def __iadd__(self, other):
        raise NotImplementedError

    def __sub__(self, other):
        raise NotImplementedError

    def __isub__(self, other):
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    def __imul__(self, other):
        raise NotImplementedError

    def __matmul__(self, other):
        raise NotImplementedError

    def __imatmul__(self, other):
        raise NotImplementedError

    def __truediv__(self, other):
        raise NotImplementedError

    def __itruediv__(self, other):
        raise NotImplementedError

    def __neg__(self):
        raise NotImplementedError

    # def __floordiv__(self, other):
    #     raise NotImplementedError()

    # def __mod__(self, other):
    #     raise NotImplementedError()

    def __pow__(self, other):
        raise NotImplementedError

    def __ipow__(self, other):
        raise NotImplementedError

    def __lshift__(self, other):
        raise NotImplementedError

    def __ilshift__(self, other):
        raise NotImplementedError

    def __rshift__(self, other):
        raise NotImplementedError

    def __irshift__(self, other):
        raise NotImplementedError

    @classmethod
    def _get_ct_shape_of_any_depth(
        cls, x, current_shape_tuple: tuple[int, ...] = ()
    ):
        if isinstance(x, list):
            return cls._get_ct_shape_of_any_depth(
                x[0], (*current_shape_tuple, len(x))
            )
        elif isinstance(x, dict):
            return cls._get_ct_shape_of_any_depth(
                next(iter(x.values())), (*current_shape_tuple, len(x))
            )
        else:
            return current_shape_tuple

    @property
    def cts_shape(self):
        return self._get_ct_shape_of_any_depth(self.cts)

    def __repr__(self) -> str:
        return f"PackedCT(cts_shape={self.__class__._get_ct_shape_of_any_depth(self.cts)}, metadata={self.metadata})"

    def __str__(self) -> str:
        return self.__repr__()

    def __len__(self):
        return len(self.cts)  # assume cts depth is 1 by default

    @property
    def original_shape(self):
        return self.metadata.original_shape

    @property
    def logical_num_slots(self):
        return self.metadata.logical_num_slots

    @property
    def packed_by(self):
        return self.metadata.packed_by

    @property
    def encoded_by(self):
        return self.metadata.encoded_by

    @property
    def misc(self):
        return self.metadata.misc

    def __getitem__(self, idx):
        if isinstance(idx, (slice, int)):
            return self.cts[idx]
        elif isinstance(idx, str):
            return self.metadata.get(idx, None) or self.metadata.misc.get(
                idx, None
            )
        else:
            raise ValueError(f"Invalid index type {type(idx)} to read value")

    def __setitem__(self, idx, value):
        if isinstance(idx, (slice, int)):
            self.cts[idx] = value
        elif isinstance(idx, str):
            raise ValueError(f"Cannot set metadata value {idx}")
        else:
            raise ValueError(f"Invalid index type {type(idx)} to set value")

    def __contains__(self, item):
        return item in self.metadata or item in self.metadata.misc

    def __iter__(self):
        return iter(self.cts)

    def __reversed__(self):
        return reversed(self.cts)
