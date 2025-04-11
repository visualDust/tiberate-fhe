import functools
import pickle
from collections import defaultdict
from enum import Flag, auto
from logging import warning
from typing import TYPE_CHECKING, Any, Dict, List, Union

import numpy
import torch
from loguru import logger
from torch import Tensor
from vdtoys.cache import CachedDict
from vdtoys.registry import Registry

engClsRegistry = Registry("ENGINE_CLASS")

if TYPE_CHECKING:  # avoid circular import
    from tiberate.fhe.engine.ckks_engine import CkksEngine


def _default_none():
    # use this instead of lambda: None in defaultdict to get rid of pytorch RPC Can't get local object 'DataStruct.__init__.<locals>.<lambda>' ERROR
    return None


class FLAGS(Flag):
    NTT_STATE = auto()
    MONTGOMERY_STATE = auto()
    INCLUDE_SPECIAL = auto()
    NEED_RESCALE = auto()
    NEED_RELINERIZE = auto()

    def dumps(self) -> str:
        """Serialize the current FLAGS instance to a comma-separated string."""
        return ",".join(flag.name for flag in FLAGS if self & flag)

    @classmethod
    def loads(cls, s: str) -> "FLAGS":
        """Deserialize from a comma-separated string back to a FLAGS instance."""
        if not s:
            return cls(0)
        flags = cls(0)
        for name in s.split(","):
            flags |= cls[name]
        return flags

    def __repr__(self):
        return self.dumps()

    def __str__(self):
        return self.dumps()


class DataStruct:
    def __init__(
        self,
        data,
        *,
        flags: Union[FLAGS, List[FLAGS], None] = None,
        level: int,
        **kwargs,
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

    def has_flag(self, flag: FLAGS) -> bool:
        """Check if a specific flag is set.
        Args:
            flag (FLAGS): The flag to check.
        Returns:
            bool: True if the flag is set, False otherwise.
        """
        return bool(self._flags & flag)

    def set_flag(self, flag: FLAGS):
        """Set a specific flag.
        Args:
            flag (FLAGS): The flag to set.
        """
        self._flags |= flag

    def rm_flag(self, flag: FLAGS):
        """Clear a specific flag.
        Args:
            flag (FLAGS): The flag to clear.
        """
        self._flags &= ~flag

    def toggle_flag(self, flag: FLAGS):
        """Toggle a specific flag.
        Args:
            flag (FLAGS): The flag to toggle.
        """
        self._flags ^= flag

    @property
    def flags(self, only_set: bool = True) -> list[FLAGS]:
        """Returns a list of all flags, optionally only those that are set."""
        return [flag for flag in FLAGS if not only_set or self.has_flag(flag)]

    @flags.setter
    def flags(self, value: list[FLAGS]):
        """Set multiple flags at once."""
        self._flags = FLAGS(0)  # Reset flags to 0
        for flag in value:
            self.set_flag(flag)

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
            return (
                data.clone()
                if data.device == device
                else data.to(device=device, non_blocking=non_blocking)
            )
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

    def save(self, path: str):  # todo migrate to safetensors
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            return pickle.load(f)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(flags={self.flags}, level={self.level}, misc={self.misc})"
        )

    def __str__(self):
        return self.__repr__()

    @property
    def _default_engine(self):
        EngineCls: CkksEngine = engClsRegistry["CkksEngine"]
        return EngineCls.get_default_for_logN(self.misc["logN"])


# ================== #
# Message Type Alias #
# ================== #

VectorMessageType = Union[numpy.ndarray, torch.Tensor, list]
ScalarMessageType = Union[int, float]
MessageType = Union[VectorMessageType, ScalarMessageType]


# ================== #
#  Plaintext Cache   #
# ================== #


class Plaintext(DataStruct):
    def _m2tensor(self, m: MessageType):
        # convert any type to tensor
        if isinstance(m, torch.Tensor):
            return m
        elif isinstance(m, numpy.ndarray):
            return torch.from_numpy(m)
        elif isinstance(m, list):
            return torch.tensor(m)
        elif isinstance(m, (int, float)):
            return torch.tensor([m])
        else:
            raise TypeError(f"Unsupported type for message: {type(m)}")

    def __init__(
        self,
        m: MessageType,
        *,
        cache: Dict[int, Dict[str, Any]] = None,  # level: {what_cache: cache_data}
        padding=True,  # todo remove padding flag in legacy code
        scale=None,  # by default None, which means use engine's parameter
        **kwargs,
    ):
        if not isinstance(m, torch.Tensor):
            m = self._m2tensor(m)
        assert isinstance(m, torch.Tensor), TypeError(
            f"Plaintext source data must be torch.Tensor, got {type(m)}."
        )
        assert m.dim() == 1, RuntimeError(
            f"Plaintext source data must be 1D tensor, got {m.dim()}D tensor."
        )
        self.src = m
        self.data = cache or defaultdict(dict)  # cache is alias of data
        self.misc = defaultdict(_default_none)
        self.misc.update(kwargs)
        self.misc["padding"] = padding
        self.misc["scale"] = scale

    def __neg__(self):
        return Plaintext(-self.src)

    @property
    def padding(self):
        return self.misc.get("padding")

    @padding.setter
    def padding(self, value):
        self.misc["padding"] = value

    @property
    def scale(self):
        return self.misc.get("scale")

    @scale.setter
    def scale(self, value):
        self.misc["scale"] = value

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


# ================== #
#  Cipher Structures #
# ================== #


class Ciphertext(DataStruct):
    DEFAULT_NEG_INPLACE = False
    DEFAULT_ROTATE_INPLACE = False
    DEFAULT_PCOP_INPLACE = False
    DEFAULT_MCOP_INPLACE = False
    DEFAULT_ROTATE_MEMORY_SAVING = True

    def __add__(
        self,
        other: Union["Ciphertext", "CiphertextTriplet", Plaintext, MessageType],
    ):
        engine = self._default_engine
        if isinstance(other, Ciphertext):  # ct + ct
            return engine.cc_add_double(self, other)
        elif isinstance(other, CiphertextTriplet):  # ct + ct_triplet
            other_relin = engine.relinearize(other)
            return engine.cc_add_double(self, other_relin)
        elif isinstance(other, ScalarMessageType):  # ct + m
            return engine.add_scalar(self, other)
        elif isinstance(other, VectorMessageType):  # ct + <m>
            other_pt = Plaintext(other)
            return engine.pc_add(other_pt, self)
        elif isinstance(other, Plaintext):  # ct + pt
            return engine.pc_add(other, self)
        else:
            raise TypeError(f"Unsupported type for addition: {type(other)}")

    def __radd__(self, other: Union[Plaintext, MessageType]):
        # m + ct
        # pt + ct
        return self + other

    def __sub__(self, other: Union["Ciphertext", "CiphertextTriplet", Plaintext]):
        engine = self._default_engine
        if isinstance(other, Ciphertext):  # ct - ct
            return engine.cc_sub_double(self, other)
        elif isinstance(other, CiphertextTriplet):  # ct - ct_triplet
            other_relin = engine.relinearize(other)
            return engine.cc_sub_double(self, other_relin)
        elif isinstance(other, ScalarMessageType):  # ct - m
            return engine.add_scalar(self, -other, inplace=self.__class__.DEFAULT_MCOP_INPLACE)
        elif isinstance(other, VectorMessageType):  # ct - <m>
            other_pt = Plaintext(-other)
            return engine.pc_add(other_pt, self, inplace=self.__class__.DEFAULT_MCOP_INPLACE)
        elif isinstance(other, Plaintext):  # ct - pt
            return engine.pc_add(-other, self, inplace=self.__class__.DEFAULT_PCOP_INPLACE)
        else:
            raise TypeError(f"Unsupported type for subtraction: {type(other)}")

    def __rsub__(self, other: Union[Plaintext, MessageType]):
        # m - ct
        # pt - ct
        engine = self._default_engine
        neg_ct = engine.negate(self, inplace=self.__class__.DEFAULT_NEG_INPLACE)
        return neg_ct + other

    def __mul__(self, other: Union["Ciphertext", "CiphertextTriplet", Plaintext]):
        engine = self._default_engine
        if isinstance(other, Ciphertext):  # ct * ct
            return engine.cc_mult(self, other)
        elif isinstance(other, CiphertextTriplet):  # ct * ct_triplet
            other_relin = engine.relinearize(other)
            return engine.cc_mult(self, other_relin)
        elif isinstance(other, ScalarMessageType):  # ct * m
            return engine.mult_scalar(self, other, inplace=self.__class__.DEFAULT_MCOP_INPLACE)
        elif isinstance(other, VectorMessageType):  # ct * <m>
            return engine.mc_mult(other, self, inplace=self.__class__.DEFAULT_MCOP_INPLACE)
        elif isinstance(other, Plaintext):  # ct * pt
            return engine.pc_mult(other, self, inplace=self.__class__.DEFAULT_PCOP_INPLACE)
        else:
            raise TypeError(f"Unsupported type for multiplication: {type(other)}")

    def __rmul__(self, other: Union[Plaintext, MessageType]):
        # m * ct
        # pt * ct
        return self * other

    def __neg__(self):
        # -ct
        engine = self._default_engine
        return engine.negate(self, inplace=self.__class__.DEFAULT_NEG_INPLACE)

    def __rshift__(self, other: int):
        # ct >> offset
        engine = self._default_engine
        assert isinstance(other, int), TypeError(
            f"Ciphertext rotation offset must be int, got {type(other)}."
        )
        if self.__class__.DEFAULT_ROTATE_MEMORY_SAVING:  # using galios rotation
            return engine.rotate_offset(
                ct=self,
                offset=other,
                inplace=self.__class__.DEFAULT_ROTATE_INPLACE,
            )
        else:  # using single rotation
            return engine.rotate_single(
                ct=self,
                rotk=engine.rotk[other],
                inplace=self.__class__.DEFAULT_ROTATE_INPLACE,
            )

    def __rrshift__(self, other: int):
        # offset >> ct
        return self >> other

    def __lshift__(self, other: int):
        # ct << offset
        return self >> (-other)

    def __rlshift__(self, other: int):
        # offset << ct
        return self >> (-other)

    def __pow__(self, power: int):
        engine = self._default_engine
        return engine.pow(self, power)

    def rescale(self, exact_rounding=True, **kwargs):
        engine = self._default_engine
        return engine.rescale(self, exact_rounding=exact_rounding, **kwargs)

    def level_up(self, dst_level: int, **kwargs):
        engine = self._default_engine
        return engine.level_up(self, dst_level=dst_level, **kwargs)

    def decryptcode(self, sk: "SecretKey" = None, is_real=False, final_round=True, **kwargs):
        engine = self._default_engine
        return engine.decryptcode(self, sk=sk, is_real=is_real, final_round=final_round, **kwargs)

    @property
    def plain(self):
        return self.decryptcode(is_real=True)

    def sum(self, **kwargs):
        engine = self._default_engine
        return engine.sum(self, **kwargs)

    def mean(self, **kwargs):
        engine = self._default_engine
        return engine.mean(self, **kwargs)

    def std(self, **kwargs):
        engine = self._default_engine
        return engine.std(self, **kwargs)


class CiphertextTriplet(DataStruct):
    def __add__(self, other: Union[Ciphertext, "CiphertextTriplet", Plaintext]):
        engine = self._default_engine
        if isinstance(other, CiphertextTriplet):
            return engine.cc_add_triplet(self, other)
        elif isinstance(other, Union[Ciphertext, Plaintext, MessageType]):
            self_relin = engine.relinearize(self)
            return other + self_relin

    def __sub__(self, other: Union[Ciphertext, "CiphertextTriplet", Plaintext]):
        engine = self._default_engine
        if isinstance(other, CiphertextTriplet):
            return engine.cc_sub_triplet(self, other)
        elif isinstance(other, Union[Ciphertext, Plaintext, MessageType]):
            self_relin = engine.relinearize(self)
            return other - self_relin

    def __mul__(self, other: Union[Ciphertext, "CiphertextTriplet", Plaintext]):
        self_relin = self._default_engine.relinearize(self)
        return self_relin * other

    def __rshift__(self, other: int):
        self_relin = self._default_engine.relinearize(self)
        return self_relin >> other

    def __rrshift__(self, other: int):
        self_relin = self._default_engine.relinearize(self)
        return self_relin >> other

    def __lshift__(self, other: int):
        self_relin = self._default_engine.relinearize(self)
        return self_relin << other

    def __rlshift__(self, other: int):
        self_relin = self._default_engine.relinearize(self)
        return self_relin << other

    def __pow__(self, power: int):
        self_relin = self._default_engine.relinearize(self)
        return self_relin**power

    def rescale(self, exact_rounding=True, **kwargs):
        engine = self._default_engine
        self_relin = engine.relinearize(self)
        return engine.rescale(self_relin, exact_rounding=exact_rounding, **kwargs)

    def level_up(self, dst_level: int, **kwargs):
        engine = self._default_engine
        self_relin = engine.relinearize(self)
        return engine.level_up(self_relin, dst_level=dst_level, **kwargs)

    def decryptcode(self, sk: "SecretKey" = None, is_real=False, final_round=True, **kwargs):
        engine = self._default_engine
        self_relin = engine.relinearize(self)
        return engine.decryptcode(
            self_relin,
            sk=sk,
            is_real=is_real,
            final_round=final_round,
            **kwargs,
        )

    @property
    def plain(self):
        return self.decryptcode(is_real=True)

    def sum(self, **kwargs):
        engine = self._default_engine
        self_relin = engine.relinearize(self)
        return engine.sum(self_relin, **kwargs)

    def mean(self, **kwargs):
        engine = self._default_engine
        self_relin = engine.relinearize(self)
        return engine.mean(self_relin, **kwargs)

    def std(self, **kwargs):
        engine = self._default_engine
        self_relin = engine.relinearize(self)
        return engine.std(self_relin, **kwargs)


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
