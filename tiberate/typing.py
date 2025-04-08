import pickle
from collections import defaultdict
from enum import Flag, auto
from logging import warning
from typing import Any, Dict, List, Union

import torch
from loguru import logger
from torch import Tensor
from vdtoys.cache import CachedDict


def _default_none():
    # use this instead of lambda: None in defaultdict to get rid of pytorch RPC Can't get local object 'DataStruct.__init__.<locals>.<lambda>' ERROR
    return None


class FLAGS(Flag):
    NTT_STATE = auto()
    MONTGOMERY_STATE = auto()
    INCLUDE_SPECIAL = auto()
    NEED_RESCALE = auto()
    NEED_RELINERIZE = auto()


class DataStruct:
    def __init__(
        self, data, *, flags: Union[FLAGS, List[FLAGS], None] = None, level: int, **kwargs
    ):
        self.data = data
        self._flags = FLAGS(0)  # Initialize flags to 0
        # if flags is a list of FLAGS, convert it to a single FLAGS
        flags = flags or []
        if isinstance(flags, list):
            for flag in flags:
                self._flags |= flag
        elif isinstance(flags, FLAGS):
            self._flags = flags

        self.level = level
        self.misc = defaultdict(_default_none)
        self.misc.update(kwargs)

    def has(self, flag: FLAGS) -> bool:
        """Check if a specific flag is set.
        Args:
            flag (FLAGS): The flag to check.
        Returns:
            bool: True if the flag is set, False otherwise.
        """
        return bool(self._flags & flag)

    def set(self, flag: FLAGS):
        """Set a specific flag.
        Args:
            flag (FLAGS): The flag to set.
        """
        self._flags |= flag

    def remove(self, flag: FLAGS):
        """Clear a specific flag.
        Args:
            flag (FLAGS): The flag to clear.
        """
        self._flags &= ~flag

    def toggle(self, flag: FLAGS):
        """Toggle a specific flag.
        Args:
            flag (FLAGS): The flag to toggle.
        """
        self._flags ^= flag

    @property
    def flags(self, only_set: bool = True) -> list[FLAGS]:
        """Returns a list of all flags, optionally only those that are set."""
        return [flag for flag in FLAGS if not only_set or self.has(flag)]

    @flags.setter
    def flags(self, value: list[FLAGS]):
        """Set multiple flags at once."""
        self._flags = FLAGS(0)  # Reset flags to 0
        for flag in value:
            self.set(flag)

    def clone(self):
        """Clone the data structure.

        Returns:
            DataStruct or its subclasses: A new instance of the same class with cloned data.
        """
        cls = self.__class__  # Get the class of the current instance
        return cls(
            data=cls.copy_tensor_to_device_recursive(self.data, self.device),
            flags=self._flags,
            level=self.level,
            **self.misc,
        )

    @classmethod
    def wrap(cls, another: "DataStruct", **kwargs):
        """Wrap another data structure into a new instance of the same class.
        Args:
            another (DataStruct): The data structure to wrap.
        Returns:
            DataStruct or its subclasses: A new instance of the same class with the same attributes as `another`.
        """
        return cls(
            data=another.data,
            flags=another._flags,
            level=another.level,
            **{**another.misc, **kwargs},  # another misc and kwargs
        )

    @classmethod
    def get_device_of_tensor(cls, data):
        """Get the device of the tensor in the data structure.

        Note: this method only checks the first found tensor of the data structure.
        It assumes that all elements in the data structure are on the same device.

        Args:
            data: The data structure to check.
        Returns:
            The device of the tensor.
        """
        # Recursively get the device of the tensor in the data structure
        if isinstance(data, Tensor):
            return data.device
        elif isinstance(data, list):
            return cls.get_device_of_tensor(data[0]) if data else "cpu"
        elif isinstance(data, tuple):  # legacy datastruct uses tuple
            return cls.get_device_of_tensor(data[0]) if data else "cpu"
        elif isinstance(data, dict):  # plaintext cache
            return (
                cls.get_device_of_tensor(list(data.values())[0])
                if data
                else "cpu"  # if data is empty, return cpu
            )
        elif isinstance(data, DataStruct):
            return cls.get_device_of_tensor(data.data)
        else:
            return "cpu"

    @property
    def device(self):
        """Get the device of the data structure.
        Returns:
            The device of the data structure.
        """
        return self.get_device_of_tensor(self.data)

    @classmethod
    def copy_tensor_to_device_recursive(cls, data, device: str, non_blocking=True):
        """Recursively move tensors in the data structure to a specified device.
        Args:
            data: The data structure to move.
            device: The target device.
        Returns:
            The data structure moved to the specified device.
        """
        # Recursively move tensors in the data structure to the specified device
        if isinstance(data, Tensor):
            return data.to(device, non_blocking=non_blocking)
        elif isinstance(data, list):
            return [cls.copy_tensor_to_device_recursive(item, device) for item in data]
        elif isinstance(data, tuple):  # legacy datastruct uses tuple
            return tuple(cls.copy_tensor_to_device_recursive(item, device) for item in data)
        elif isinstance(data, dict):  # plaintext cache
            return {
                cls.copy_tensor_to_device_recursive(
                    key, device
                ): cls.copy_tensor_to_device_recursive(value, device)
                for key, value in data.items()
            }
        elif isinstance(data, CachedDict):
            new_instance = data.__class__(data.generator_func)
            new_instance._cache = cls.copy_tensor_to_device_recursive(data._cache, device)
            return new_instance
        elif isinstance(data, DataStruct):
            return data.copy_to(device)
        else:
            return data

    def copy_to(self, device: str, non_blocking=True):
        """Copy the data structure to a specified device and return a new instance.
        Args:
            device: The target device.
        Returns:
            DataStruct or its subclasses: A new instance of the same class with data moved to the specified device.
        """
        cls = self.__class__
        return cls(
            data=cls.copy_tensor_to_device_recursive(
                data=self.data, device=device, non_blocking=non_blocking
            ),
            flags=self._flags,
            level=self.level,
            **self.misc,
        )

    def to(self, device: str, non_blocking=True):
        # alias for copy_to
        return self.copy_to(device, non_blocking)

    def dump(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def dumps(self):
        return pickle.dumps(self)

    @classmethod
    def load(cls, path: str) -> "DataStruct":
        with open(path, "rb") as f:
            return pickle.load(f)

    @classmethod
    def loads(cls, data) -> "DataStruct":
        return pickle.loads(data)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(flags={self.flags}, level={self.level}, misc={self.misc})"
        )

    def __str__(self):
        return self.__repr__()  # todo for better readability


# ================== #
#  Cipher Structures #
# ================== #


class Ciphertext(DataStruct):
    pass


class CiphertextTriplet(DataStruct):
    pass


# ================== #
#  Key Structures    #
# ================== #


class SecretKey(DataStruct):
    # todo does secret key even exits ntt_state and montgomery_state?
    pass


class EvaluationKey(SecretKey):
    pass


class PublicKey(DataStruct):
    pass


class KeySwitchKey(DataStruct):
    pass


class RotationKey(KeySwitchKey):
    @property
    def delta(self):
        return self.misc.get("delta")

    @delta.setter
    def delta(self, value):
        self.misc["delta"] = value


class GaloisKey(DataStruct):
    def __init__(self, data, *, flags=None, level, **kwargs):
        super().__init__(data, flags=flags, level=level, **kwargs)
        logger.warning(
            DeprecationWarning(
                "GaloisKey is deprecated, methods that uses GaloisKey will be removed in future versions."
            )
        )


class ConjugationKey(DataStruct):
    pass


# ================== #
#  Plaintext Cache   #
# ================== #


class Plaintext(DataStruct):
    def __init__(
        self,
        src: Union[list, tuple],
        *,
        cache: Dict[int, Dict[str, Any]] = None,  # level: {what_cache: cache_data}
        padding=True,  # todo remove padding flag in legacy code
        scale=None,  # by default None, which means use engine's parameter
    ):
        if not isinstance(src, torch.Tensor):
            src = torch.tensor(src)
        if src.dim() == 2 and src.size(0) == 1:
            src = src.squeeze(0)
        assert src.dim() == 1, RuntimeError(
            f"Plaintext source data must be 1D tensor, got {src.dim()}D tensor."
        )
        self.src = src
        self.data = cache or defaultdict(dict)  # cache is alias of data
        self.padding = padding
        self.scale = scale

    @property
    def cache(self):
        return self.data

    @cache.setter
    def cache(self, value):
        self.data = value

    def clone(self):
        cls = self.__class__
        cache = cls.copy_tensor_to_device_recursive(self.cache, self.device)
        return cls(self.src, cache=cache)

    @property
    def device(self):
        if not self.cache:
            return self.get_device_of_tensor(self.src)
        else:
            return self.get_device_of_tensor(self.cache)

    def copy_to(self, device, non_blocking=True):
        cls = self.__class__
        cache = cls.copy_tensor_to_device_recursive(
            data=self.cache, device=device, non_blocking=non_blocking
        )
        return cls(self.src, cache=cache)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(data={self.src}, cached levels={list(self.cache.keys())})"
        )

    @property
    def level(self):
        raise NotImplementedError("Plaintext does not have level attribute.")
