import functools
import math
import warnings
from hashlib import sha256
from uuid import uuid4

import numpy as np
import torch
from loguru import logger
from vdtoys.cache import CachedDict

# from vdtoys.mvc import strictype # enable when debugging
import tiberate.utils.encoding as codec
from tiberate import errors
from tiberate.config import CkksConfig, Preset
from tiberate.context.ntt_context import NTTContext
from tiberate.rng import Csprng
from tiberate.typing import *  # noqa: F403
from tiberate.utils.massive import decompose_rot_offsets


class CkksEngine:
    __default: dict[str, "CkksEngine"] = {}

    def __init__(
        self,
        ckks_config: CkksConfig | dict = None,
        devices=None,
        allow_sk_gen: bool = True,  # if True, will allow sk generation
        bias_guard: bool = True,
        norm: str = "forward",
    ):
        if not ckks_config:
            logger.info("Ckks config is None. Using default config logN15.")
            ckks_config = Preset.logN15
        if not isinstance(ckks_config, CkksConfig):
            ckks_config = CkksConfig.parse(ckks_config)
        self.ckksCfg = ckks_config

        if devices is None:
            logger.info("Using default device: cuda:0")
            devices = ["cuda:0"]

        self.nttCtx = NTTContext(self.ckksCfg, devices=devices)
        self.rnsPart = self.nttCtx.rnsPart
        self.montCtx = self.nttCtx.montCtx

        self.rng = Csprng(
            num_coefs=self.ckksCfg.N,
            num_channels=[len(di) for di in self.rnsPart.d],
            num_repeating_channels=max(ckks_config.num_special_primes, 2),
            devices=devices,
        )

        self.bias_guard = bias_guard
        self.norm = norm

        self._make_adjustments_and_corrections()
        self._make_mont_PR()
        self._create_ksk_rescales()
        self._alloc_parts()
        self._leveled_devices()
        self._create_rescale_scales()

        # id
        self.id = str(uuid4())  # unique id for this engine at runtime

        # by default, do not create any keys
        self.allow_sk_gen = allow_sk_gen
        self.__sk = None
        self.__pk = None
        self.__evk = None
        self.__gk = None
        self.__rotk = {}

        # if there is no default engine, set this as default
        if self.ckksCfg.logN not in self.__class__.__default:
            logger.info(
                f"Setting engine {self.id} as default for logN {self.ckksCfg.logN}."
            )
            self.__class__.__default[self.ckksCfg.logN] = self

    @property
    def num_levels(self):  # alias
        return self.ckksCfg.num_scales

    @property
    def num_slots(self):
        return self.ckksCfg.N // 2

    # -------------------------------------------------------------------------------------------
    # Various pre-calculations.
    # -------------------------------------------------------------------------------------------

    def _create_rescale_scales(self):
        rescale_scales = []
        for level in range(self.num_levels):
            rescale_scales.append([])

            for device_id in range(self.nttCtx.num_devices):
                dest_level = self.rnsPart.destination_arrays[level]

                if device_id < len(dest_level):
                    dest = dest_level[device_id]
                    rescaler_device_id = self.rnsPart.rescaler_loc[level]
                    m0 = self.montCtx.q[level]

                    if rescaler_device_id == device_id:
                        m = [self.montCtx.q[i] for i in dest[1:]]
                    else:
                        m = [self.montCtx.q[i] for i in dest]

                    scales = [
                        (pow(m0, -1, mi) * self.montCtx.R) % mi for mi in m
                    ]

                    scales = torch.tensor(
                        scales,
                        dtype=self.ckksCfg.torch_dtype,
                        device=self.nttCtx.devices[device_id],
                    )
                    rescale_scales[level].append(scales)

        self.rescale_scales = rescale_scales

    def _leveled_devices(self):
        self.len_devices = []
        for level in range(self.num_levels):
            self.len_devices.append(
                len([[a] for a in self.rnsPart.p[level] if len(a) > 0])
            )

        self.neighbor_devices = []
        for level in range(self.num_levels):
            self.neighbor_devices.append([])
            len_devices_at = self.len_devices[level]
            available_devices_ids = range(len_devices_at)
            for src_device_id in available_devices_ids:
                neighbor_devices_at = [
                    device_id
                    for device_id in available_devices_ids
                    if device_id != src_device_id
                ]
                self.neighbor_devices[level].append(neighbor_devices_at)

    def _alloc_parts(self):
        self.parts_alloc = []
        for level in range(self.num_levels):
            num_parts = [len(parts) for parts in self.rnsPart.p[level]]
            parts_alloc = [
                alloc[-num_parts[di] - 1 : -1]
                for di, alloc in enumerate(self.rnsPart.part_allocations)
            ]
            self.parts_alloc.append(parts_alloc)

        self.stor_ids = []
        for level in range(self.num_levels):
            self.stor_ids.append([])
            alloc = self.parts_alloc[level]
            min_id = min([min(a) for a in alloc if len(a) > 0])
            for device_id in range(self.nttCtx.num_devices):
                global_ids = self.parts_alloc[level][device_id]
                new_ids = [i - min_id for i in global_ids]
                self.stor_ids[level].append(new_ids)

    def _create_ksk_rescales(self):
        # reserve the buffers.
        self.ksk_buffers = []
        for device_id in range(self.nttCtx.num_devices):
            self.ksk_buffers.append([])
            for part_id in range(len(self.rnsPart.p[0][device_id])):
                buffer = torch.empty(
                    [
                        self.ckksCfg.num_special_primes,
                        self.ckksCfg.N,
                    ],
                    dtype=self.ckksCfg.torch_dtype,
                ).pin_memory()
                self.ksk_buffers[device_id].append(buffer)

        # Create the buffers.
        R = self.montCtx.R
        P = self.montCtx.q[-self.ckksCfg.num_special_primes :][::-1]
        m = self.montCtx.q
        PiR = [
            [(pow(Pj, -1, mi) * R) % mi for mi in m[: -P_ind - 1]]
            for P_ind, Pj in enumerate(P)
        ]

        self.PiRs = []

        level = 0
        self.PiRs.append([])

        for P_ind in range(self.ckksCfg.num_special_primes):
            self.PiRs[level].append([])

            for device_id in range(self.nttCtx.num_devices):
                dest = self.rnsPart.destination_arrays_with_special[level][
                    device_id
                ]
                PiRi = [PiR[P_ind][i] for i in dest[: -P_ind - 1]]
                PiRi = torch.tensor(
                    PiRi,
                    device=self.nttCtx.devices[device_id],
                    dtype=self.ckksCfg.torch_dtype,
                )
                self.PiRs[level][P_ind].append(PiRi)

        for level in range(1, self.num_levels):
            self.PiRs.append([])

            for P_ind in range(self.ckksCfg.num_special_primes):
                self.PiRs[level].append([])

                for device_id in range(self.nttCtx.num_devices):
                    start = self.rnsPart.diff[level][device_id]
                    PiRi = self.PiRs[0][P_ind][device_id][start:]

                    self.PiRs[level][P_ind].append(PiRi)

    def _make_mont_PR(self):
        P = math.prod(self.montCtx.q[-self.ckksCfg.num_special_primes :])
        R = self.montCtx.R
        PR = P * R
        mont_PR = []
        for device_id in range(self.nttCtx.num_devices):
            dest = self.rnsPart.destination_arrays[0][device_id]
            m = [self.montCtx.q[i] for i in dest]
            PRm = [PR % mi for mi in m]
            PRm = torch.tensor(
                PRm,
                device=self.nttCtx.devices[device_id],
                dtype=self.ckksCfg.torch_dtype,
            )
            mont_PR.append(PRm)
        self.mont_PR = mont_PR

    def _make_adjustments_and_corrections(self):
        self.alpha = [
            (self.ckksCfg.scale / np.float64(q)) ** 2
            for q in self.montCtx.q[: self.ckksCfg.num_scales]
        ]
        self.deviations = [1]
        for al in self.alpha:
            self.deviations.append(self.deviations[-1] ** 2 * al)

        self.final_q_ind = [
            da[0][0] for da in self.rnsPart.destination_arrays[:-1]
        ]
        self.final_q = [self.montCtx.q[ind] for ind in self.final_q_ind]
        self.final_alpha = [
            (self.ckksCfg.scale / np.float64(q)) for q in self.final_q
        ]
        self.corrections = [
            1 / (d * fa) for d, fa in zip(self.deviations, self.final_alpha)
        ]
        self.base_prime = self.montCtx.q[self.rnsPart.base_prime_idx]

        self.final_scalar = []
        for qi, q in zip(self.final_q_ind, self.final_q):
            scalar = (
                pow(q, -1, self.base_prime) * self.montCtx.R
            ) % self.base_prime
            scalar = torch.tensor(
                [scalar],
                device=self.nttCtx.devices[0],
                dtype=self.ckksCfg.torch_dtype,
            )
            self.final_scalar.append(scalar)

    @classmethod
    def get_default_for_logN(cls, logN):
        if logN not in cls.__default:
            raise RuntimeError(
                f"No default engine for logN {logN}. Please create an engine for this logN before call this function."
            )
        return cls.__default[logN]

    def set_as_default(self):
        logN = self.ckksCfg.logN
        if logN in self.__class__.__default:
            logger.warning(
                f"Engine for logN {logN} already exists({self.__class__.__default[logN].id}). Overwriting."
            )
        self.__class__.__default[logN] = self
        logger.info(f"Setting engine {self.id} as default for logN {logN}.")

    @property
    def sk(self) -> SecretKey:
        if self.__sk is None:
            if not self.allow_sk_gen:
                raise RuntimeError("Secret key generation is disabled.")
            self.sk = self._create_secret_key()
            logger.debug("Created a new secret key.")
        return self.__sk

    @sk.setter
    def sk(self, sk: SecretKey):
        # remove all keys generated by sk
        if self.__pk:
            self.__pk = None
            logger.info(
                "Original public key is removed due to manual secret key assignment."
            )
        if self.__evk:
            self.__evk = None
            logger.info(
                "Original evaluation key is removed due to manual secret key assignment."
            )
        if self.__gk:
            self.__gk = None
            logger.info(
                "Original galois key is removed due to manual secret key assignment."
            )
        if self.__rotk:
            self.__rotk = {}
            logger.info(
                "Original rotation key is removed due to manual secret key assignment."
            )

        self.__sk = sk

    @property
    def pk(self) -> PublicKey:
        if self.__pk is None:
            self.__pk = self._create_public_key(self.sk)
            logger.debug("Created a new public key.")
        return self.__pk

    @pk.setter
    def pk(self, pk: PublicKey):
        self.__pk = pk

    @property
    def evk(self) -> EvaluationKey:
        if self.__evk is None:
            self.__evk = self._create_evk(self.sk)
            logger.debug("Created a new evaluation key.")
        return self.__evk

    @evk.setter
    def evk(self, evk: EvaluationKey):
        self.__evk = evk

    @property
    def gk(self) -> GaloisKey:
        if self.__gk is None:
            self.__gk = self._create_galois_key(self.sk)
            warnings.deprecated(
                "Galois key (gk) will deprecate in the future. Use rotation key (rotk) instead.",
            )
            logger.debug("Created a new galois key.")
        return self.__gk

    @gk.setter
    def gk(self, gk: GaloisKey):
        self.__gk = gk

    @property
    def rotk(self) -> CachedDict:
        if not self.__rotk and self.allow_sk_gen:
            self.__rotk = CachedDict(
                generator_func=functools.partial(
                    self._create_rotation_key, sk=self.sk
                )
            )
        return self.__rotk

    @rotk.setter
    def rotk(self, rotk: CachedDict | dict):
        if isinstance(rotk, CachedDict):
            new_rotk = rotk
        elif isinstance(rotk, dict):
            if self.allow_sk_gen:
                new_rotk = CachedDict(
                    generator_func=functools.partial(
                        self._create_rotation_key, sk=self.sk
                    )
                )
                new_rotk._cache = rotk
            else:
                new_rotk = rotk
        else:
            raise TypeError(
                f"Invalid type for rotation key: {type(rotk)}. Expected dict or CachedDict."
            )
        self.__rotk = new_rotk

    def __str__(self):
        result = f"{self.__class__.__name__} "
        result += f"({self.id}) "
        result += str(self.ckksCfg)
        return result

    @property
    def device0(self) -> int:
        # todo remove multi-device by default
        return self.nttCtx.devices[0]

    @property
    @functools.cache  # >= python 3.9  # noqa: B019
    def hash(self) -> str:
        """
        Hash of the engine.
        This is used to identify the engine and its parameters.
        """
        q_str = ",".join(map(str, self.montCtx.q))
        hash_input = f"{self.ckksCfg!r}_{q_str}"
        # logger.debug(f"Hash string: {hashstr}")
        return sha256(hash_input.encode("utf-8")).hexdigest()

    # -------------------------------------------------------------------------------------------
    # Encode/Decode
    # -------------------------------------------------------------------------------------------

    # @torch.compile(backend=tiberate_compiler)
    def encode(
        self, m, level: int = 0, padding=True, scale=None
    ) -> list[torch.Tensor]:
        """
        Encode a plain message m.
        Note that the encoded plain text is pre-permuted to yield cyclic rotation.
        """
        deviation = self.deviations[level]
        if padding:
            m = codec.padding(m, num_slots=self.num_slots)
        encoded = [
            codec.encode(
                m,
                scale=scale or self.ckksCfg.scale,
                rng=self.rng,
                device=self.device0,
                deviation=deviation,
                norm=self.norm,
            )
        ]

        pt_buffer = self.ksk_buffers[0][0][0]
        pt_buffer.copy_(encoded[-1])
        for dev_id in range(1, self.nttCtx.num_devices):
            encoded.append(pt_buffer.cuda(self.nttCtx.devices[dev_id]))
        return encoded

    # @torch.compile(backend=tiberate_compiler)
    def decode(self, m, level=0, is_real: bool = False) -> list:
        """
        Base prime is located at -1 of the RNS channels in GPU0.
        Assuming this is an orginary RNS deinclude_special.
        """
        correction = self.corrections[level]
        decoded = codec.decode(
            m[0].squeeze(),
            scale=self.ckksCfg.scale,
            correction=correction,
            norm=self.norm,
        )
        m = decoded[: self.ckksCfg.N // 2].cpu().numpy()
        if is_real:
            m = m.real
        return m

    # -------------------------------------------------------------------------------------------
    # secret key/public key generation.
    # -------------------------------------------------------------------------------------------

    def _create_secret_key(self, include_special: bool = True) -> SecretKey:
        uniform_ternary = self.rng.randint(amax=3, shift=-1, repeats=1)

        mult_type = -2 if include_special else -1
        unsigned_ternary = self.nttCtx.tile_unsigned(
            uniform_ternary, lvl=0, mult_type=mult_type
        )
        self.nttCtx.enter_ntt(unsigned_ternary, 0, mult_type)

        return SecretKey(
            data=unsigned_ternary,
            flags=(FLAGS.INCLUDE_SPECIAL if include_special else FLAGS(0))
            | FLAGS.MONTGOMERY_STATE
            | FLAGS.NTT_STATE,
            level=0,
            # following is metadata (not required args)
            logN=self.ckksCfg.logN,
            creator_hash=self.hash,
        )

    # @strictype # enable when debugging
    def _create_public_key(
        self,
        sk: SecretKey = None,
        *,
        include_special: bool = False,
        a: list[torch.Tensor] | None = None,
    ) -> PublicKey:
        """
        Generates a public key against the secret key sk.
        pk = -a * sk + e = e - a * sk
        """

        sk = sk or self.sk

        if include_special and not sk.has_flag(FLAGS.INCLUDE_SPECIAL):
            raise errors.SecretKeyNotIncludeSpecialPrime

        # Set the mult_type
        mult_type = -2 if include_special else -1

        # Generate errors for the ordinary case.
        level = 0
        e = self.rng.discrete_gaussian(repeats=1)
        e = self.nttCtx.tile_unsigned(e, level, mult_type)

        self.nttCtx.enter_ntt(e, level, mult_type)
        repeats = (
            self.ckksCfg.num_special_primes
            if sk.has_flag(FLAGS.INCLUDE_SPECIAL)
            else 0
        )

        # Applying mont_mult in the order of 'a', sk will
        if a is None:
            a = self.rng.randint(
                self.nttCtx.q_prepack[mult_type][level][0], repeats=repeats
            )

        sa = self.nttCtx.mont_mult(a, sk.data, 0, mult_type)
        pk0 = self.nttCtx.mont_sub(e, sa, 0, mult_type)

        return PublicKey(
            data=[pk0, a],
            flags=(FLAGS.INCLUDE_SPECIAL if include_special else FLAGS(0))
            | FLAGS.MONTGOMERY_STATE
            | FLAGS.NTT_STATE,
            level=0,
            # following is metadata (not required args)
            logN=self.ckksCfg.logN,
            creator_hash=self.hash,
        )

    # -------------------------------------------------------------------------------------------
    # Encrypt/Decrypt
    # -------------------------------------------------------------------------------------------

    # @strictype # enable when debugging
    # @torch.compile(backend=tiberate_compiler)
    def encrypt(
        self, pt: list[torch.Tensor], pk: PublicKey = None, *, level: int = 0
    ) -> Ciphertext:
        """
        We again, multiply pt by the scale.
        Since pt is already multiplied by the scale,
        the multiplied pt no longer can be stored
        in a single RNS channel.
        That means we are forced to do the multiplication
        in full RNS domain.
        Note that we allow encryption at
        levels other than 0, and that will take care of multiplying
        the deviation factors.
        """
        pk = pk or self.pk

        mult_type = -2 if pk.has_flag(FLAGS.INCLUDE_SPECIAL) else -1

        e0e1 = self.rng.discrete_gaussian(repeats=2)

        e0 = [e[0] for e in e0e1]
        e1 = [e[1] for e in e0e1]

        e0_tiled = self.nttCtx.tile_unsigned(e0, level, mult_type)
        e1_tiled = self.nttCtx.tile_unsigned(e1, level, mult_type)

        pt_tiled = self.nttCtx.tile_unsigned(pt, level, mult_type)
        self.nttCtx.mont_enter_scale(pt_tiled, level, mult_type)
        self.nttCtx.mont_reduce(pt_tiled, level, mult_type)
        pte0 = self.nttCtx.mont_add(pt_tiled, e0_tiled, level, mult_type)

        start = self.nttCtx.starts[level]
        pk0 = [
            pk.data[0][di][start[di] :] for di in range(self.nttCtx.num_devices)
        ]
        pk1 = [
            pk.data[1][di][start[di] :] for di in range(self.nttCtx.num_devices)
        ]

        v = self.rng.randint(amax=2, shift=0, repeats=1)

        v = self.nttCtx.tile_unsigned(v, level, mult_type)
        self.nttCtx.enter_ntt(v, level, mult_type)

        vpk0 = self.nttCtx.mont_mult(v, pk0, level, mult_type)
        vpk1 = self.nttCtx.mont_mult(v, pk1, level, mult_type)

        self.nttCtx.intt_exit(vpk0, level, mult_type)
        self.nttCtx.intt_exit(vpk1, level, mult_type)

        ct0 = self.nttCtx.mont_add(vpk0, pte0, level, mult_type)
        ct1 = self.nttCtx.mont_add(vpk1, e1_tiled, level, mult_type)

        self.nttCtx.reduce_2q(ct0, level, mult_type)
        self.nttCtx.reduce_2q(ct1, level, mult_type)

        ct = Ciphertext(
            data=[ct0, ct1],
            flags=(
                FLAGS.INCLUDE_SPECIAL
                if pk.has_flag(FLAGS.INCLUDE_SPECIAL)
                else FLAGS(0)
            )
            | FLAGS.NTT_STATE
            | FLAGS.MONTGOMERY_STATE,
            level=level,
            # following is metadata (not required args)
            logN=self.ckksCfg.logN,
            creator_hash=self.hash,
        )

        return ct

    # @strictype # enable when debugging
    # @torch.compile(backend=tiberate_compiler)
    def decrypt_triplet(
        self,
        ct_mult: CiphertextTriplet,
        sk: SecretKey = None,
        *,
        final_round=True,
    ) -> list[torch.Tensor]:
        sk = sk or self.sk

        if not ct_mult.has_flag(FLAGS.NTT_STATE):
            raise errors.NTTStateError(expected=True)
        if not ct_mult.has_flag(FLAGS.MONTGOMERY_STATE):
            raise errors.MontgomeryStateError(expected=True)
        if not sk.has_flag(FLAGS.NTT_STATE):
            raise errors.NTTStateError(expected=True)
        if not sk.has_flag(FLAGS.MONTGOMERY_STATE):
            raise errors.MontgomeryStateError(expected=True)

        level = ct_mult.level
        d0 = [ct_mult.data[0][0].clone()]
        d1 = [ct_mult.data[1][0]]
        d2 = [ct_mult.data[2][0]]

        self.nttCtx.intt_exit_reduce(d0, level)

        sk_data = [sk.data[0][self.nttCtx.starts[level][0] :]]

        d1_s = self.nttCtx.mont_mult(d1, sk_data, level)

        s2 = self.nttCtx.mont_mult(sk_data, sk_data, level)
        d2_s2 = self.nttCtx.mont_mult(d2, s2, level)

        self.nttCtx.intt_exit(d1_s, level)
        self.nttCtx.intt_exit(d2_s2, level)

        pt = self.nttCtx.mont_add(d0, d1_s, level)
        pt = self.nttCtx.mont_add(pt, d2_s2, level)
        self.nttCtx.reduce_2q(pt, level)

        base_at = (
            -self.ckksCfg.num_special_primes - 1
            if ct_mult.has_flag(FLAGS.INCLUDE_SPECIAL)
            else -1
        )

        base = pt[0][base_at][None, :]
        scaler = pt[0][0][None, :]

        final_scalar = self.final_scalar[level]
        scaled = self.nttCtx.mont_sub([base], [scaler], -1)
        self.nttCtx.mont_enter_scalar(scaled, [final_scalar], -1)
        self.nttCtx.reduce_2q(scaled, -1)
        self.nttCtx.make_signed(scaled, -1)

        # Round?
        if final_round:
            # The scaler and the base channels are guaranteed to be in the
            # device 0.
            rounding_prime = self.nttCtx.qlists[0][
                -self.ckksCfg.num_special_primes - 2
            ]
            rounder = (scaler[0] > (rounding_prime // 2)) * 1
            scaled[0] += rounder

        return scaled

    # @strictype # enable when debugging
    def decrypt_double(
        self, ct: Ciphertext, sk: SecretKey = None, *, final_round=True
    ) -> list[torch.Tensor]:
        sk = sk or self.sk

        if ct.has_flag(FLAGS.NTT_STATE):
            raise errors.NTTStateError(expected=False)
        if ct.has_flag(FLAGS.MONTGOMERY_STATE):
            raise errors.MontgomeryStateError(expected=False)
        if not sk.has_flag(FLAGS.NTT_STATE):
            raise errors.NTTStateError(expected=True)
        if not sk.has_flag(FLAGS.MONTGOMERY_STATE):
            raise errors.MontgomeryStateError(expected=True)

        ct0 = ct.data[0][0]
        level = ct.level
        sk_data = sk.data[0][self.nttCtx.starts[level][0] :]
        a = ct.data[1][0].clone()

        self.nttCtx.enter_ntt([a], level)
        sa = self.nttCtx.mont_mult([a], [sk_data], level)
        self.nttCtx.intt_exit(sa, level)

        pt = self.nttCtx.mont_add([ct0], sa, level)
        self.nttCtx.reduce_2q(pt, level)

        base_at = (
            -self.ckksCfg.num_special_primes - 1
            if ct.has_flag(FLAGS.INCLUDE_SPECIAL)
            else -1
        )

        base = pt[0][base_at][None, :]
        scaler = pt[0][0][None, :]

        final_scalar = self.final_scalar[level]
        scaled = self.nttCtx.mont_sub([base], [scaler], -1)
        self.nttCtx.mont_enter_scalar(scaled, [final_scalar], -1)
        self.nttCtx.reduce_2q(scaled, -1)
        self.nttCtx.make_signed(scaled, -1)

        # Round?
        if final_round:
            # The scaler and the base channels are guaranteed to be in the
            # device 0.
            rounding_prime = self.nttCtx.qlists[0][
                -self.ckksCfg.num_special_primes - 2
            ]
            rounder = (scaler[0] > (rounding_prime // 2)) * 1
            scaled[0] += rounder

        return scaled

    # @restrict_type
    # @torch.compile(backend=tiberate_compiler)
    def decrypt(
        self,
        ct: Ciphertext | CiphertextTriplet,
        sk: SecretKey = None,
        *,
        final_round=True,
    ) -> list[torch.Tensor]:
        """
        Decrypt the cipher text ct using the secret key sk.
        Note that the final rescaling must precede the actual decryption process.
        """
        sk = sk or self.sk

        if isinstance(ct, CiphertextTriplet):
            pt = self.decrypt_triplet(
                ct_mult=ct, sk=sk, final_round=final_round
            )
        else:
            # raise errors.NotMatchType(origin=ct.origin, to=f"{origin_names['ct']} or {origin_names['ctt']}")
            # try decode as ct
            if not isinstance(ct, Ciphertext):
                logger.warning(
                    f"ct type is mismatched: excepted {Ciphertext} or {CiphertextTriplet}, got {type(ct)}, will try to decode as ct anyway."
                )
            pt = self.decrypt_double(ct=ct, sk=sk, final_round=final_round)

        return pt

    # -------------------------------------------------------------------------------------------
    # Key switching.
    # -------------------------------------------------------------------------------------------

    # @strictype # enable when debugging
    # @torch.compile(backend=tiberate_compiler)
    def create_key_switching_key(
        self, sk_from: SecretKey, sk_to: SecretKey, a=None
    ) -> KeySwitchKey:
        """
        Creates a key to switch the key for sk_src to sk_dst.
        """

        if not sk_from.has_flag(FLAGS.NTT_STATE):
            raise errors.NTTStateError(expected=True)
        if not sk_from.has_flag(FLAGS.MONTGOMERY_STATE):
            raise errors.MontgomeryStateError(expected=True)
        if not sk_to.has_flag(FLAGS.NTT_STATE):
            raise errors.NTTStateError(expected=True)
        if not sk_to.has_flag(FLAGS.MONTGOMERY_STATE):
            raise errors.MontgomeryStateError(expected=True)

        level = 0

        stops = self.nttCtx.stops[-1]
        Psk_src = [
            sk_from.data[di][: stops[di]].clone()
            for di in range(self.nttCtx.num_devices)
        ]

        self.nttCtx.mont_enter_scalar(Psk_src, self.mont_PR, level)

        ksk = [[] for _ in range(self.rnsPart.num_partitions + 1)]

        for device_id in range(self.nttCtx.num_devices):
            for part_id, part in enumerate(self.rnsPart.p[level][device_id]):
                global_part_id = self.rnsPart.part_allocations[device_id][
                    part_id
                ]

                crs = a[global_part_id] if a else None
                pk = self._create_public_key(sk_to, include_special=True, a=crs)

                key = tuple(part)
                astart = part[0]
                astop = part[-1] + 1
                shard = Psk_src[device_id][astart:astop]
                pk_data = pk.data[0][device_id][astart:astop]

                _2q = self.nttCtx.parts_pack[device_id][key]["_2q"]
                update_part = torch.ops.tiberate_ntt_ops.mont_add(
                    [pk_data], [shard], _2q
                )[0]
                pk_data.copy_(update_part, non_blocking=True)

                pk.misc["description"] = (
                    f"key switch key part index {global_part_id}"  # todo allow this and rename to description
                )

                ksk[global_part_id] = pk

        return KeySwitchKey(
            data=ksk,
            flags=FLAGS.INCLUDE_SPECIAL
            | FLAGS.MONTGOMERY_STATE
            | FLAGS.NTT_STATE,
            level=level,
            # following is metadata (not required args)
            logN=self.ckksCfg.logN,
            creator_hash=self.hash,
        )

    # @torch.compile(backend=tiberate_compiler)
    def pre_extend(self, a, device_id, level, part_id, exit_ntt=False):
        # param_parts contain only the ordinary parts.
        # Hence, loop around it.
        # text_parts contain special primes.
        text_part = self.rnsPart.parts[level][device_id][part_id]
        param_part = self.rnsPart.p[level][device_id][part_id]

        # Carve out the partition.
        alpha = len(text_part)
        a_part = a[device_id][text_part[0] : text_part[-1] + 1]

        # Release ntt.
        if exit_ntt:
            self.nttCtx.intt_exit_reduce([a_part], level, device_id, part_id)

        # Prepare a state.
        # Initially, it is x[0] % m[i].
        # However, m[i] is a monotonically increasing
        # sequence, i.e., repeating the row would suffice
        # to do the job.

        # 2023-10-16, Juwhan Kim, In fact, m[i] is NOT monotonically increasing.

        state = a_part[0].repeat(alpha, 1)

        key = tuple(param_part)
        for i in range(alpha - 1):
            mont_pack = self.nttCtx.parts_pack[device_id][param_part[i + 1],][
                "mont_pack"
            ]
            _2q = self.nttCtx.parts_pack[device_id][param_part[i + 1],]["_2q"]
            Y_scalar = self.nttCtx.parts_pack[device_id][key]["Y_scalar"][i][
                None
            ]

            Y = (a_part[i + 1] - state[i + 1])[None, :]

            # mont_enter will take care of signedness.
            # ntt_cuda.make_unsigned([Y], _2q)
            torch.ops.tiberate_ntt_ops.mont_enter([Y], [Y_scalar], *mont_pack)
            # ntt_cuda.reduce_2q([Y], _2q)

            state[i + 1] = Y

            if i + 2 < alpha:
                state_key = tuple(param_part[i + 2 :])
                state_mont_pack = self.nttCtx.parts_pack[device_id][state_key][
                    "mont_pack"
                ]
                state_2q = self.nttCtx.parts_pack[device_id][state_key]["_2q"]
                L_scalar = self.nttCtx.parts_pack[device_id][key]["L_scalar"][i]
                new_state_len = alpha - (i + 2)
                new_state = Y.repeat(new_state_len, 1)
                torch.ops.tiberate_ntt_ops.mont_enter(
                    [new_state], [L_scalar], *state_mont_pack
                )
                state[i + 2 :] += new_state

        # Returned state is in plain integer format.
        return state

    # @torch.compile(backend=tiberate_compiler)
    def extend(self, state, device_id, level, part_id, target_device_id=None):
        # Note that device_id, level, and part_id is from
        # where the state has been originally calculated at.
        # The state can reside in a different GPU than
        # the original one.

        if target_device_id is None:
            target_device_id = device_id

        rns_len = len(
            self.rnsPart.destination_arrays_with_special[level][
                target_device_id
            ]
        )
        alpha = len(state)

        # Initialize the output
        extended = state[0].repeat(rns_len, 1)
        self.nttCtx.mont_enter([extended], level, target_device_id, -2)

        # Generate the search key to find the L_enter.
        part = self.rnsPart.p[level][device_id][part_id]
        key = tuple(part)

        # Extract the L_enter in the target device.
        L_enter = self.nttCtx.parts_pack[device_id][key]["L_enter"][
            target_device_id
        ]

        # L_enter covers the whole rns range.
        # Start from the leveled start.
        start = self.nttCtx.starts[level][target_device_id]

        # Loop to generate.
        for i in range(alpha - 1):
            Y = state[i + 1].repeat(rns_len, 1)

            self.nttCtx.mont_enter_scalar(
                [Y], [L_enter[i][start:]], level, target_device_id, -2
            )
            extended = self.nttCtx.mont_add(
                [extended], [Y], level, target_device_id, -2
            )[0]

        # Returned extended is in the Montgomery format.
        return extended

    # @nvtx.annotate() # todo see why memory copy with only one device
    def create_switcher(
        self,
        a: list[torch.Tensor],
        ksk: KeySwitchKey,
        level: int,
        exit_ntt: bool = False,
    ) -> tuple:
        # ksk parts allocation.
        ksk_alloc = self.parts_alloc[level]

        # Device lens and neighbor devices.
        len_devices = self.len_devices[level]
        neighbor_devices = self.neighbor_devices[level]

        # Iterate over source device ids, and then part ids.
        num_parts = sum([len(alloc) for alloc in ksk_alloc])
        part_results = [
            [
                [[] for _ in range(len_devices)],
                [[] for _ in range(len_devices)],
            ]
            for _ in range(num_parts)
        ]

        # 1. Generate states.
        states = [[] for _ in range(num_parts)]
        for src_device_id in range(len_devices):
            for part_id in range(len(self.rnsPart.p[level][src_device_id])):
                storage_id = self.stor_ids[level][src_device_id][part_id]
                state = self.pre_extend(
                    a, src_device_id, level, part_id, exit_ntt
                )
                states[storage_id] = state

        # 2. Copy to CPU.
        CPU_states = [[] for _ in range(num_parts)]
        for src_device_id in range(len_devices):
            for part_id, part in enumerate(
                self.rnsPart.p[level][src_device_id]
            ):
                storage_id = self.stor_ids[level][src_device_id][part_id]
                alpha = len(part)
                CPU_state = self.ksk_buffers[src_device_id][part_id][:alpha]
                CPU_state.copy_(states[storage_id], non_blocking=True)
                CPU_states[storage_id] = CPU_state

        # 3. Continue on with the follow ups on source devices.
        for src_device_id in range(len_devices):
            for part_id in range(len(self.rnsPart.p[level][src_device_id])):
                storage_id = self.stor_ids[level][src_device_id][part_id]
                state = states[storage_id]
                d0, d1 = self.switcher_later_part(
                    state, ksk, src_device_id, src_device_id, level, part_id
                )

                part_results[storage_id][0][src_device_id] = d0
                part_results[storage_id][1][src_device_id] = d1

        # 4. Copy onto neighbor GPUs the states.
        CUDA_states = [[] for _ in range(num_parts)]
        for src_device_id in range(len_devices):
            for j, dst_device_id in enumerate(neighbor_devices[src_device_id]):
                for part_id, part in enumerate(
                    self.rnsPart.p[level][src_device_id]
                ):
                    storage_id = self.stor_ids[level][src_device_id][part_id]
                    CPU_state = CPU_states[storage_id]
                    CUDA_states[storage_id] = CPU_state.cuda(
                        self.nttCtx.devices[dst_device_id],
                        non_blocking=True,
                    )

        # 5. Synchronize.
        # torch.cuda.synchronize()

        # 6. Do follow ups on neighbors.
        for src_device_id in range(len_devices):
            for j, dst_device_id in enumerate(neighbor_devices[src_device_id]):
                for part_id, part in enumerate(
                    self.rnsPart.p[level][src_device_id]
                ):
                    storage_id = self.stor_ids[level][src_device_id][part_id]
                    CUDA_state = CUDA_states[storage_id]
                    d0, d1 = self.switcher_later_part(
                        CUDA_state,
                        ksk,
                        src_device_id,
                        dst_device_id,
                        level,
                        part_id,
                    )
                    part_results[storage_id][0][dst_device_id] = d0
                    part_results[storage_id][1][dst_device_id] = d1

        # 7. Sum up.
        summed0 = part_results[0][0]
        summed1 = part_results[0][1]

        for i in range(1, len(part_results)):
            summed0 = self.nttCtx.mont_add(
                summed0, part_results[i][0], level, -2
            )
            summed1 = self.nttCtx.mont_add(
                summed1, part_results[i][1], level, -2
            )

        # Rename summed's.
        d0 = summed0
        d1 = summed1

        # intt to prepare for division by P.
        self.nttCtx.intt_exit_reduce(d0, level, -2)
        self.nttCtx.intt_exit_reduce(d1, level, -2)

        # 6. Divide by P.
        # This is actually done in successive order.
        # Rescale from the most outer prime channel.
        # Start from the special len and drop channels one by one.

        # Pre-montgomery enter the ordinary part.
        # Note that special prime channels remain intact.
        c0 = [d[: -self.ckksCfg.num_special_primes] for d in d0]
        c1 = [d[: -self.ckksCfg.num_special_primes] for d in d1]

        self.nttCtx.mont_enter(c0, level, -1)
        self.nttCtx.mont_enter(c1, level, -1)

        current_len = [
            len(d) for d in self.rnsPart.destination_arrays_with_special[level]
        ]

        for P_ind in range(self.ckksCfg.num_special_primes):
            PiRi = self.PiRs[level][P_ind]

            # Tile.
            P0 = [
                d[-1 - P_ind].repeat(current_len[di], 1)
                for di, d in enumerate(d0)
            ]
            P1 = [
                d[-1 - P_ind].repeat(current_len[di], 1)
                for di, d in enumerate(d1)
            ]

            # mont enter only the ordinary part.
            Q0 = [d[: -self.ckksCfg.num_special_primes] for d in P0]
            Q1 = [d[: -self.ckksCfg.num_special_primes] for d in P1]

            self.nttCtx.mont_enter(Q0, level, -1)
            self.nttCtx.mont_enter(Q1, level, -1)

            # subtract P0 and P1.
            # Note that by the consequence of the above mont_enter
            # ordinary parts will be in montgomery form,
            # while the special part remains plain.
            d0 = self.nttCtx.mont_sub(d0, P0, level, -2)
            d1 = self.nttCtx.mont_sub(d1, P1, level, -2)

            self.nttCtx.mont_enter_scalar(d0, PiRi, level, -2)
            self.nttCtx.mont_enter_scalar(d1, PiRi, level, -2)

        # Carve out again, since d0 and d1 are fresh new.
        c0 = [d[: -self.ckksCfg.num_special_primes] for d in d0]
        c1 = [d[: -self.ckksCfg.num_special_primes] for d in d1]

        # Exit the montgomery.
        self.nttCtx.mont_reduce(c0, level, -1)
        self.nttCtx.mont_reduce(c1, level, -1)

        self.nttCtx.reduce_2q(c0, level, -1)
        self.nttCtx.reduce_2q(c1, level, -1)

        # 7. Return
        return c0, c1

    def switcher_later_part(
        self,
        state,
        ksk: KeySwitchKey,
        src_device_id,
        dst_device_id,
        level,
        part_id,
    ):
        # Extend basis.
        extended = self.extend(
            state, src_device_id, level, part_id, dst_device_id
        )

        # ntt extended to prepare polynomial multiplication.
        # extended is in the Montgomery format already.
        self.nttCtx.ntt([extended], level, dst_device_id, -2)

        # Extract the ksk.
        ksk_loc = self.parts_alloc[level][src_device_id][part_id]
        ksk_part_data = ksk.data[ksk_loc].data

        start = self.nttCtx.starts[level][dst_device_id]
        ksk0_data = ksk_part_data[0][dst_device_id][start:]
        ksk1_data = ksk_part_data[1][dst_device_id][start:]

        # Multiply.
        d0 = self.nttCtx.mont_mult(
            [extended], [ksk0_data], level, dst_device_id, -2
        )
        d1 = self.nttCtx.mont_mult(
            [extended], [ksk1_data], level, dst_device_id, -2
        )

        # When returning, un-list the results by taking the 0th element.
        return d0[0], d1[0]

    # @strictype # enable when debugging
    def switch_key(self, ct: Ciphertext, ksk: KeySwitchKey) -> Ciphertext:
        level = ct.level
        a = ct.data[1]
        d0, d1 = self.create_switcher(
            a, ksk, level, exit_ntt=ct.has_flag(FLAGS.NTT_STATE)
        )

        new_ct0 = self.nttCtx.mont_add(ct.data[0], d0, level, -1)
        self.nttCtx.reduce_2q(new_ct0, level, -1)

        return Ciphertext(
            data=[new_ct0, d1],
            flags=ct._flags,
            level=level,
            # following is metadata (not required args)
            logN=self.ckksCfg.logN,
            creator_hash=self.hash,
        )

    # -------------------------------------------------------------------------------------------
    # Multiplication.
    # -------------------------------------------------------------------------------------------

    # @strictype # enable when debugging
    # @torch.compiler.disable()
    def rescale(self, ct: Ciphertext, exact_rounding=True) -> Ciphertext:
        level = ct.level
        next_level = level + 1

        if next_level >= self.num_levels:
            raise errors.MaximumLevelError(
                level=ct.level, level_max=self.num_levels
            )

        rescaler_device_id = self.rnsPart.rescaler_loc[level]
        neighbor_devices_before = self.neighbor_devices[level]
        neighbor_devices_after = self.neighbor_devices[next_level]
        len_devices_after = len(neighbor_devices_after)
        len_devices_before = len(neighbor_devices_before)

        rescaling_scales = self.rescale_scales[level]
        data0 = [[] for _ in range(len_devices_after)]
        data1 = [[] for _ in range(len_devices_after)]

        rescaler0 = [[] for _ in range(len_devices_before)]
        rescaler1 = [[] for _ in range(len_devices_before)]

        rescaler0_at = ct.data[0][rescaler_device_id][0]
        rescaler0[rescaler_device_id] = rescaler0_at

        rescaler1_at = ct.data[1][rescaler_device_id][0]
        rescaler1[rescaler_device_id] = rescaler1_at

        if rescaler_device_id < len_devices_after:
            data0[rescaler_device_id] = ct.data[0][rescaler_device_id][1:]
            data1[rescaler_device_id] = ct.data[1][rescaler_device_id][1:]

        CPU_rescaler0 = self.ksk_buffers[0][0][0]
        CPU_rescaler1 = self.ksk_buffers[0][1][0]

        CPU_rescaler0.copy_(rescaler0_at, non_blocking=True)
        CPU_rescaler1.copy_(rescaler1_at, non_blocking=True)

        for device_id in neighbor_devices_before[rescaler_device_id]:
            device = self.nttCtx.devices[device_id]
            CUDA_rescaler0 = CPU_rescaler0.cuda(device)
            CUDA_rescaler1 = CPU_rescaler1.cuda(device)

            rescaler0[device_id] = CUDA_rescaler0
            rescaler1[device_id] = CUDA_rescaler1

            if device_id < len_devices_after:
                data0[device_id] = ct.data[0][device_id]
                data1[device_id] = ct.data[1][device_id]

        if exact_rounding:
            rescale_channel_prime_id = self.rnsPart.destination_arrays[level][
                rescaler_device_id
            ][0]

            round_at = self.montCtx.q[rescale_channel_prime_id] // 2

            rounder0 = [[] for _ in range(len_devices_before)]
            rounder1 = [[] for _ in range(len_devices_before)]

            for device_id in range(len_devices_after):
                rounder0[device_id] = torch.where(
                    rescaler0[device_id] > round_at, 1, 0
                )
                rounder1[device_id] = torch.where(
                    rescaler1[device_id] > round_at, 1, 0
                )

        data0 = [(d - s) for d, s in zip(data0, rescaler0)]
        data1 = [(d - s) for d, s in zip(data1, rescaler1)]

        self.nttCtx.mont_enter_scalar(
            data0, self.rescale_scales[level], next_level
        )

        self.nttCtx.mont_enter_scalar(
            data1, self.rescale_scales[level], next_level
        )

        if exact_rounding:
            data0 = [(d + r) for d, r in zip(data0, rounder0)]
            data1 = [(d + r) for d, r in zip(data1, rounder1)]

        self.nttCtx.reduce_2q(data0, next_level)
        self.nttCtx.reduce_2q(data1, next_level)

        return Ciphertext(
            data=[data0, data1],
            level=next_level,
            # following is metadata (not required args)
            logN=self.ckksCfg.logN,
            creator_hash=self.hash,
        )

    # @strictype # enable when debugging
    def _create_evk(self, sk: SecretKey = None) -> EvaluationKey:
        sk = sk or self.sk
        sk2_data = self.nttCtx.mont_mult(sk.data, sk.data, 0, -2)
        sk2 = EvaluationKey(
            data=sk2_data,
            flags=FLAGS.MONTGOMERY_STATE
            | FLAGS.NTT_STATE
            | FLAGS.INCLUDE_SPECIAL,
            level=sk.level,
            # following is metadata (not required args)
            logN=self.ckksCfg.logN,
            creator_hash=self.hash,
        )
        return EvaluationKey.wrap(self.create_key_switching_key(sk2, sk))

    # @strictype # enable when debugging
    # @torch.compile(backend=tiberate.jit.tiberate_compiler)
    def cc_mult(
        self,
        a: Ciphertext,
        b: Ciphertext,
        evk: EvaluationKey = None,
        *,
        pre_rescale=True,
        post_relin=True,
    ) -> Ciphertext | CiphertextTriplet:
        if pre_rescale:
            x = self.rescale(a)
            y = self.rescale(b)
        else:
            x, y = a, b

        level = x.level

        # Multiply.
        x0 = x.data[0]
        x1 = x.data[1]

        y0 = y.data[0]
        y1 = y.data[1]

        self.nttCtx.enter_ntt(x0, level)
        self.nttCtx.enter_ntt(x1, level)
        self.nttCtx.enter_ntt(y0, level)
        self.nttCtx.enter_ntt(y1, level)

        d0 = self.nttCtx.mont_mult(x0, y0, level)

        x0y1 = self.nttCtx.mont_mult(x0, y1, level)
        x1y0 = self.nttCtx.mont_mult(x1, y0, level)
        d1 = self.nttCtx.mont_add(x0y1, x1y0, level)

        d2 = self.nttCtx.mont_mult(x1, y1, level)

        ct_mult = CiphertextTriplet(
            data=[d0, d1, d2],
            flags=FLAGS.NTT_STATE
            | FLAGS.MONTGOMERY_STATE
            | FLAGS.NEED_RELINERIZE,
            level=level,
            # following is metadata (not required args)
            logN=self.ckksCfg.logN,
            creator_hash=self.hash,
        )
        if post_relin:
            evk = evk or self.evk
            ct_mult = self.relinearize(ct_triplet=ct_mult, evk=evk)

        return ct_mult

    # @strictype # enable when debugging
    def relinearize(
        self, ct_triplet: CiphertextTriplet, evk: EvaluationKey = None
    ) -> Ciphertext:
        evk = evk or self.evk

        if not ct_triplet.has_flag(FLAGS.NTT_STATE):
            raise errors.NTTStateError(expected=True)
        if not ct_triplet.has_flag(FLAGS.MONTGOMERY_STATE):
            raise errors.MontgomeryStateError(expected=True)

        d0, d1, d2 = ct_triplet.data
        level = ct_triplet.level

        # intt.
        self.nttCtx.intt_exit_reduce(d0, level)
        self.nttCtx.intt_exit_reduce(d1, level)
        self.nttCtx.intt_exit_reduce(d2, level)

        # Key switch the x1y1.
        d2_0, d2_1 = self.create_switcher(d2, evk, level)

        # Add the switcher to d0, d1.
        d0 = [p + q for p, q in zip(d0, d2_0)]
        d1 = [p + q for p, q in zip(d1, d2_1)]

        # Final reduction.
        self.nttCtx.reduce_2q(d0, level)
        self.nttCtx.reduce_2q(d1, level)

        # Compose and return.
        return Ciphertext(
            data=[d0, d1],
            level=level,
            # following is metadata (not required args)
            logN=self.ckksCfg.logN,
            creator_hash=self.hash,
        )

    # -------------------------------------------------------------------------------------------
    # Rotation.
    # -------------------------------------------------------------------------------------------

    # @strictype # enable when debugging
    def _create_rotation_key(
        self,
        delta: int,
        a: list[torch.Tensor] | None = None,
        sk: SecretKey = None,
    ) -> RotationKey:
        sk = sk or self.sk
        sk_new_data = [s.clone() for s in sk.data]
        self.nttCtx.intt(sk_new_data)
        sk_new_data = [codec.rotate(s, delta) for s in sk_new_data]
        self.nttCtx.ntt(sk_new_data)
        sk_rotated = SecretKey(
            data=sk_new_data,
            flags=FLAGS.MONTGOMERY_STATE | FLAGS.NTT_STATE,
            level=0,
            # following is metadata (not required args)
            logN=self.ckksCfg.logN,
            creator_hash=self.hash,
        )

        rotk = RotationKey.wrap(
            self.create_key_switching_key(sk_rotated, sk, a=a), delta=delta
        )

        logger.debug(f"Rotation key created for delta {delta}")
        return rotk

    # @strictype # enable when debugging
    def rotate_single(
        self,
        ct: Ciphertext,
        rotk: RotationKey,
        post_key_switching=True,
        inplace: bool = True,
    ) -> Ciphertext:
        if not inplace:
            ct = ct.clone()

        level = ct.level

        rotated_ct_data = [
            [codec.rotate(d, rotk.delta) for d in ct_data]
            for ct_data in ct.data
        ]

        # Rotated ct may contain negative numbers.
        mult_type = -2 if ct.has_flag(FLAGS.INCLUDE_SPECIAL) else -1
        for ct_data in rotated_ct_data:
            self.nttCtx.make_unsigned(ct_data, level, mult_type)
            self.nttCtx.reduce_2q(ct_data, level, mult_type)

        rotated_ct = Ciphertext(
            data=rotated_ct_data,
            flags=ct._flags,
            level=level,
            # following is metadata (not required args)
            logN=self.ckksCfg.logN,
            creator_hash=self.hash,
        )
        if post_key_switching:
            rotated_ct = self.switch_key(rotated_ct, rotk)
        return rotated_ct

    # @strictype # enable when debugging
    def _create_galois_key(self, sk: SecretKey = None) -> GaloisKey:
        sk = sk or self.sk
        galois_deltas = [2**i for i in range(self.ckksCfg.logN - 1)]
        galois_key_parts = [
            self._create_rotation_key(delta=delta, sk=sk)
            for delta in galois_deltas
        ]

        galois_key = GaloisKey(
            data=galois_key_parts,
            flags=FLAGS.MONTGOMERY_STATE
            | FLAGS.NTT_STATE
            | FLAGS.INCLUDE_SPECIAL,
            level=0,
            # following is metadata (not required args)
            logN=self.ckksCfg.logN,
            creator_hash=self.hash,
        )
        return galois_key

    # @strictype # enable when debugging
    def rotate_galois(
        self,
        ct: Ciphertext,
        gk: GaloisKey = None,
        *,
        delta: int,
        return_circuit=False,
    ) -> Ciphertext:
        warnings.warn(
            DeprecationWarning(
                "rotate_galois is deprecated, please use rotate_offset instead. This function call has been redirected to rotate_offset."
            )
        )

        return self.rotate_offset(
            ct=ct, offset=delta, return_decomposed_offsets=return_circuit
        )

        # gk = gk or self.gk
        # current_delta = delta % (self.ckksCfg.N // 2)
        # galois_circuit = []
        # galois_deltas = [2**i for i in range(self.ckksCfg.logN - 1)]
        # while current_delta:
        #     galois_ind = int(math.log2(current_delta))
        #     galois_delta = galois_deltas[galois_ind]
        #     galois_circuit.append(galois_ind)
        #     current_delta -= galois_delta

        # if len(galois_circuit) > 0:
        #     rotated_ct = self.rotate_single(ct, gk.data[galois_circuit[0]])

        #     for delta_ind in galois_circuit[1:]:
        #         rotated_ct = self.rotate_single(rotated_ct, gk.data[delta_ind])
        # elif len(galois_circuit) == 0:
        #     rotated_ct = ct
        # else:
        #     pass

        # if return_circuit:
        #     return rotated_ct, galois_circuit
        # else:
        #     return rotated_ct

    # @strictype # enable when debugging
    def rotate_offset(
        self,
        ct: Ciphertext,
        offset: int,
        inplace: bool = True,
        return_decomposed_offsets=False,
    ) -> Ciphertext:
        if not inplace:
            ct = ct.clone()
        if offset == 0:
            return ct
        if offset in self.rotk:
            return self.rotate_single(ct, self.rotk[offset])
        offsets = decompose_rot_offsets(offset, self.num_slots, rotks=self.rotk)
        for delta in offsets:
            ct = self.rotate_single(ct, self.rotk[delta])
        if return_decomposed_offsets:
            ct = ct, offsets
        return ct

    # -------------------------------------------------------------------------------------------
    # Add/sub.
    # -------------------------------------------------------------------------------------------
    # @strictype # enable when debugging
    def cc_add_double(self, a: Ciphertext, b: Ciphertext) -> Ciphertext:
        if a.has_flag(FLAGS.NTT_STATE):
            raise errors.NTTStateError(expected=False)
        if a.has_flag(FLAGS.MONTGOMERY_STATE):
            raise errors.MontgomeryStateError(expected=False)
        if b.has_flag(FLAGS.NTT_STATE):
            raise errors.NTTStateError(expected=False)
        if b.has_flag(FLAGS.MONTGOMERY_STATE):
            raise errors.MontgomeryStateError(expected=False)

        level = a.level
        data = []
        c0 = self.nttCtx.mont_add(a.data[0], b.data[0], level)
        c1 = self.nttCtx.mont_add(a.data[1], b.data[1], level)
        self.nttCtx.reduce_2q(c0, level)
        self.nttCtx.reduce_2q(c1, level)
        data.extend([c0, c1])

        return Ciphertext(
            data=data,
            level=level,
            # following is metadata (not required args)
            logN=self.ckksCfg.logN,
            creator_hash=self.hash,
        )

    # @strictype # enable when debugging
    def cc_add_triplet(
        self, a: CiphertextTriplet, b: CiphertextTriplet
    ) -> CiphertextTriplet:
        if not a.has_flag(FLAGS.NTT_STATE):
            raise errors.NTTStateError(expected=True)
        if not a.has_flag(FLAGS.MONTGOMERY_STATE):
            raise errors.MontgomeryStateError(expected=True)
        if not b.has_flag(FLAGS.NTT_STATE):
            raise errors.NTTStateError(expected=True)
        if not b.has_flag(FLAGS.MONTGOMERY_STATE):
            raise errors.MontgomeryStateError(expected=True)

        level = a.level
        data = []
        c0 = self.nttCtx.mont_add(a.data[0], b.data[0], level)
        c1 = self.nttCtx.mont_add(a.data[1], b.data[1], level)
        self.nttCtx.reduce_2q(c0, level)
        self.nttCtx.reduce_2q(c1, level)
        data.extend([c0, c1])
        c2 = self.nttCtx.mont_add(a.data[2], b.data[2], level)
        self.nttCtx.reduce_2q(c2, level)
        data.append(c2)

        return CiphertextTriplet(
            data=data,
            flags=FLAGS.MONTGOMERY_STATE
            | FLAGS.NTT_STATE
            | FLAGS.NEED_RELINERIZE,
            level=level,
            # following is metadata (not required args)
            logN=self.ckksCfg.logN,
            creator_hash=self.hash,
        )

    def cc_add(
        self,
        a: Ciphertext | CiphertextTriplet,
        b: Ciphertext | CiphertextTriplet,
    ) -> Ciphertext | CiphertextTriplet:
        if isinstance(a, Ciphertext) and isinstance(b, Ciphertext):
            result = self.cc_add_double(a, b)

        elif isinstance(a, CiphertextTriplet) and isinstance(
            b, CiphertextTriplet
        ):
            result = self.cc_add_triplet(a, b)
        else:
            raise errors.DifferentTypeError(a=type(a), b=type(b))

        return result

    # @strictype # enable when debugging
    def cc_sub_double(self, a: Ciphertext, b: Ciphertext) -> Ciphertext:
        if a.has_flag(FLAGS.NTT_STATE):
            raise errors.NTTStateError(expected=False)
        if a.has_flag(FLAGS.MONTGOMERY_STATE):
            raise errors.MontgomeryStateError(expected=False)
        if b.has_flag(FLAGS.NTT_STATE):
            raise errors.NTTStateError(expected=False)
        if b.has_flag(FLAGS.MONTGOMERY_STATE):
            raise errors.MontgomeryStateError(expected=False)

        level = a.level
        data = []

        c0 = self.nttCtx.mont_sub(a.data[0], b.data[0], level)
        c1 = self.nttCtx.mont_sub(a.data[1], b.data[1], level)
        self.nttCtx.reduce_2q(c0, level)
        self.nttCtx.reduce_2q(c1, level)
        data.extend([c0, c1])

        return Ciphertext(
            data=data,
            level=level,
            # following is metadata (not required args)
            logN=self.ckksCfg.logN,
            creator_hash=self.hash,
        )

    # @strictype # enable when debugging
    def cc_sub_triplet(
        self, a: CiphertextTriplet, b: CiphertextTriplet
    ) -> CiphertextTriplet:
        if not a.has_flag(FLAGS.NTT_STATE):
            raise errors.NTTStateError(expected=True)
        if not a.has_flag(FLAGS.MONTGOMERY_STATE):
            raise errors.MontgomeryStateError(expected=True)
        if not b.has_flag(FLAGS.NTT_STATE):
            raise errors.NTTStateError(expected=True)
        if not b.has_flag(FLAGS.MONTGOMERY_STATE):
            raise errors.MontgomeryStateError(expected=True)

        level = a.level
        data = []
        c0 = self.nttCtx.mont_sub(a.data[0], b.data[0], level)
        c1 = self.nttCtx.mont_sub(a.data[1], b.data[1], level)
        c2 = self.nttCtx.mont_sub(a.data[2], b.data[2], level)
        self.nttCtx.reduce_2q(c0, level)
        self.nttCtx.reduce_2q(c1, level)
        self.nttCtx.reduce_2q(c2, level)
        data.extend([c0, c1, c2])

        return CiphertextTriplet(
            data=data,
            flags=FLAGS.MONTGOMERY_STATE
            | FLAGS.NTT_STATE
            | FLAGS.NEED_RELINERIZE,
            level=level,
            # following is metadata (not required args)
            logN=self.ckksCfg.logN,
            creator_hash=self.hash,
        )

    # @strictype # enable when debugging
    def cc_sub(
        self,
        a: Ciphertext | CiphertextTriplet,
        b: Ciphertext | CiphertextTriplet,
    ) -> Ciphertext | CiphertextTriplet:
        if isinstance(a, Ciphertext) and isinstance(b, Ciphertext):
            ct_sub = self.cc_sub_double(a, b)
        elif isinstance(a, CiphertextTriplet) and isinstance(
            b, CiphertextTriplet
        ):
            ct_sub = self.cc_sub_triplet(a, b)
        else:
            raise errors.DifferentTypeError(a=type(a), b=type(b))
        return ct_sub

    # -------------------------------------------------------------------------------------------
    # Level up.
    # -------------------------------------------------------------------------------------------
    # @strictype # enable when debugging
    def level_up(
        self, ct: Ciphertext, dst_level: int, inplace=True
    ) -> Ciphertext:
        if ct.level == dst_level:
            return ct if inplace else ct.clone()

        current_level = ct.level

        new_ct = self.rescale(ct)

        src_level = current_level + 1

        dst_len_devices = len(self.rnsPart.destination_arrays[dst_level])

        diff_deviation = self.deviations[dst_level] / np.sqrt(
            self.deviations[src_level]
        )

        deviated_delta = round(self.ckksCfg.scale * diff_deviation)

        if dst_level - src_level > 0:
            src_rns_lens = [
                len(d) for d in self.rnsPart.destination_arrays[src_level]
            ]
            dst_rns_lens = [
                len(d) for d in self.rnsPart.destination_arrays[dst_level]
            ]

            diff_rns_lens = [y - x for x, y in zip(dst_rns_lens, src_rns_lens)]

            new_ct_data0 = []
            new_ct_data1 = []

            for device_id in range(dst_len_devices):
                new_ct_data0.append(
                    new_ct.data[0][device_id][diff_rns_lens[device_id] :]
                )
                new_ct_data1.append(
                    new_ct.data[1][device_id][diff_rns_lens[device_id] :]
                )
        else:
            new_ct_data0, new_ct_data1 = new_ct.data

        multipliers = []
        for device_id in range(dst_len_devices):
            dest = self.rnsPart.destination_arrays[dst_level][device_id]
            q = [self.montCtx.q[i] for i in dest]

            multiplier = [(deviated_delta * self.montCtx.R) % qi for qi in q]
            multiplier = torch.tensor(
                multiplier,
                dtype=self.ckksCfg.torch_dtype,
                device=self.nttCtx.devices[device_id],
            )
            multipliers.append(multiplier)

        self.nttCtx.mont_enter_scalar(new_ct_data0, multipliers, dst_level)
        self.nttCtx.mont_enter_scalar(new_ct_data1, multipliers, dst_level)

        self.nttCtx.reduce_2q(new_ct_data0, dst_level)
        self.nttCtx.reduce_2q(new_ct_data1, dst_level)

        new_ct = Ciphertext(
            data=[new_ct_data0, new_ct_data1],
            level=dst_level,
            # following is metadata (not required args)
            logN=self.ckksCfg.logN,
            creator_hash=self.hash,
        )

        return new_ct

    # -------------------------------------------------------------------------------------------
    # Fused enc/dec.
    # -------------------------------------------------------------------------------------------

    # @strictype # enable when debugging
    def encodecrypt(
        self, m, pk: PublicKey = None, *, level: int = 0, padding=True
    ) -> Ciphertext:
        pk = pk or self.pk
        if padding:
            m = codec.padding(m=m, num_slots=self.num_slots)
        deviation = self.deviations[level]
        pt = codec.encode(
            m,
            scale=self.ckksCfg.scale,
            device=self.device0,
            norm=self.norm,
            deviation=deviation,
            rng=self.rng,
            return_without_scaling=self.bias_guard,
        )

        if self.bias_guard:
            dc_integral = pt[0].item() // 1
            pt[0] -= dc_integral

            dc_scale = int(dc_integral) * int(self.ckksCfg.scale)
            dc_rns = []
            for device_id, dest in enumerate(
                self.rnsPart.destination_arrays[level]
            ):
                dci = [dc_scale % self.montCtx.q[i] for i in dest]
                dci = torch.tensor(
                    dci,
                    dtype=self.ckksCfg.torch_dtype,
                    device=self.nttCtx.devices[device_id],
                )
                dc_rns.append(dci)

            pt *= np.float64(self.ckksCfg.scale)
            pt = self.rng.randround(pt)

        encoded = [pt]

        pt_buffer = self.ksk_buffers[0][0][0]
        pt_buffer.copy_(encoded[-1])
        for dev_id in range(1, self.nttCtx.num_devices):
            encoded.append(pt_buffer.cuda(self.nttCtx.devices[dev_id]))

        mult_type = -2 if pk.has_flag(FLAGS.INCLUDE_SPECIAL) else -1

        e0e1 = self.rng.discrete_gaussian(repeats=2)

        e0 = [e[0] for e in e0e1]
        e1 = [e[1] for e in e0e1]

        e0_tiled = self.nttCtx.tile_unsigned(e0, level, mult_type)
        e1_tiled = self.nttCtx.tile_unsigned(e1, level, mult_type)

        pt_tiled = self.nttCtx.tile_unsigned(encoded, level, mult_type)

        if self.bias_guard:
            for device_id, pti in enumerate(pt_tiled):
                pti[:, 0] += dc_rns[device_id]

        self.nttCtx.mont_enter_scale(pt_tiled, level, mult_type)
        self.nttCtx.mont_reduce(pt_tiled, level, mult_type)
        pte0 = self.nttCtx.mont_add(pt_tiled, e0_tiled, level, mult_type)

        start = self.nttCtx.starts[level]
        pk0 = [
            pk.data[0][di][start[di] :] for di in range(self.nttCtx.num_devices)
        ]
        pk1 = [
            pk.data[1][di][start[di] :] for di in range(self.nttCtx.num_devices)
        ]

        v = self.rng.randint(amax=2, shift=0, repeats=1)

        v = self.nttCtx.tile_unsigned(v, level, mult_type)
        self.nttCtx.enter_ntt(v, level, mult_type)

        vpk0 = self.nttCtx.mont_mult(v, pk0, level, mult_type)
        vpk1 = self.nttCtx.mont_mult(v, pk1, level, mult_type)

        self.nttCtx.intt_exit(vpk0, level, mult_type)
        self.nttCtx.intt_exit(vpk1, level, mult_type)

        ct0 = self.nttCtx.mont_add(vpk0, pte0, level, mult_type)
        ct1 = self.nttCtx.mont_add(vpk1, e1_tiled, level, mult_type)

        self.nttCtx.reduce_2q(ct0, level, mult_type)
        self.nttCtx.reduce_2q(ct1, level, mult_type)

        ct = Ciphertext(
            data=[ct0, ct1],
            flags=(
                FLAGS.INCLUDE_SPECIAL
                if pk.has_flag(FLAGS.INCLUDE_SPECIAL)
                else FLAGS(0)
            ),
            level=level,
            # following is metadata (not required args)
            logN=self.ckksCfg.logN,
            creator_hash=self.hash,
        )

        return ct

    def decryptcode(
        self,
        ct: Ciphertext | CiphertextTriplet,
        sk: SecretKey = None,
        *,
        is_real=False,
        final_round=True,
    ):  # todo keep on GPU or not convert back to numpy
        sk = sk or self.sk

        if not sk.has_flag(FLAGS.NTT_STATE):
            raise errors.NTTStateError(expected=True)
        if not sk.has_flag(FLAGS.MONTGOMERY_STATE):
            raise errors.MontgomeryStateError(expected=True)

        level = ct.level
        sk_data = sk.data[0][self.nttCtx.starts[level][0] :]

        if isinstance(ct, CiphertextTriplet):
            if not ct.has_flag(FLAGS.NTT_STATE):
                raise errors.NTTStateError(expected=True)
            if not ct.has_flag(FLAGS.MONTGOMERY_STATE):
                raise errors.MontgomeryStateError(expected=True)

            d0 = [ct.data[0][0].clone()]
            d1 = [ct.data[1][0]]
            d2 = [ct.data[2][0]]

            self.nttCtx.intt_exit_reduce(d0, level)

            sk_data = [sk.data[0][self.nttCtx.starts[level][0] :]]

            d1_s = self.nttCtx.mont_mult(d1, sk_data, level)

            s2 = self.nttCtx.mont_mult(sk_data, sk_data, level)
            d2_s2 = self.nttCtx.mont_mult(d2, s2, level)

            self.nttCtx.intt_exit(d1_s, level)
            self.nttCtx.intt_exit(d2_s2, level)

            pt = self.nttCtx.mont_add(d0, d1_s, level)
            pt = self.nttCtx.mont_add(pt, d2_s2, level)
            self.nttCtx.reduce_2q(pt, level)
        else:
            if not isinstance(ct, Ciphertext):
                logger.warning(
                    f"ct type is mismatched: excepted {Ciphertext} or {CiphertextTriplet}, got {type(ct)}, will try to decode as ct anyway."
                )
            if ct.has_flag(FLAGS.NTT_STATE):
                raise errors.NTTStateError(expected=False)
            if ct.has_flag(FLAGS.MONTGOMERY_STATE):
                raise errors.MontgomeryStateError(expected=False)

            ct0 = ct.data[0][0]
            a = ct.data[1][0].clone()

            self.nttCtx.enter_ntt([a], level)

            sa = self.nttCtx.mont_mult([a], [sk_data], level)
            self.nttCtx.intt_exit(sa, level)

            pt = self.nttCtx.mont_add([ct0], sa, level)
            self.nttCtx.reduce_2q(pt, level)

        base_at = (
            -self.ckksCfg.num_special_primes - 1
            if ct.has_flag(FLAGS.INCLUDE_SPECIAL)
            else -1
        )
        base = pt[0][base_at][None, :]
        scaler = pt[0][0][None, :]

        len_left = len(self.rnsPart.destination_arrays[level][0])

        if (len_left >= 3) and self.bias_guard:
            dc0 = base[0][0].item()
            dc1 = scaler[0][0].item()
            dc2 = pt[0][1][0].item()

            base[0][0] = 0
            scaler[0][0] = 0

            q0_ind = self.rnsPart.destination_arrays[level][0][base_at]
            q1_ind = self.rnsPart.destination_arrays[level][0][0]
            q2_ind = self.rnsPart.destination_arrays[level][0][1]

            q0 = self.montCtx.q[q0_ind]
            q1 = self.montCtx.q[q1_ind]
            q2 = self.montCtx.q[q2_ind]

            Q = q0 * q1 * q2
            Q0 = q1 * q2
            Q1 = q0 * q2
            Q2 = q0 * q1

            Qi0 = pow(Q0, -1, q0)
            Qi1 = pow(Q1, -1, q1)
            Qi2 = pow(Q2, -1, q2)

            dc = (dc0 * Qi0 * Q0 + dc1 * Qi1 * Q1 + dc2 * Qi2 * Q2) % Q

            half_Q = Q // 2
            dc = dc if dc <= half_Q else dc - Q

            dc = (dc + (q1 - 1)) // q1

        final_scalar = self.final_scalar[level]
        scaled = self.nttCtx.mont_sub([base], [scaler], -1)
        self.nttCtx.mont_enter_scalar(scaled, [final_scalar], -1)
        self.nttCtx.reduce_2q(scaled, -1)
        self.nttCtx.make_signed(scaled, -1)

        # Round?
        if final_round:
            # The scaler and the base channels are guaranteed to be in the
            # device 0.
            rounding_prime = self.nttCtx.qlists[0][
                -self.ckksCfg.num_special_primes - 2
            ]
            rounder = (scaler[0] > (rounding_prime // 2)) * 1
            scaled[0] += rounder

        # Decoding.
        correction = self.corrections[level]
        decoded = codec.decode(
            scaled[0][-1],
            scale=self.ckksCfg.scale,
            correction=correction,
            norm=self.norm,
            return_without_scaling=self.bias_guard,
        )
        decoded = decoded[: self.ckksCfg.N // 2].cpu().numpy()

        decoded = decoded / self.ckksCfg.scale * correction

        # Bias guard.
        if (len_left >= 3) and self.bias_guard:
            decoded += dc / self.ckksCfg.scale * correction
        if is_real:
            decoded = decoded.real
        return decoded

    # -------------------------------------------------------------------------------------------
    # Conjugation
    # -------------------------------------------------------------------------------------------

    # @strictype # enable when debugging
    def create_conjugation_key(self, sk: SecretKey = None) -> ConjugationKey:
        sk = sk or self.sk

        if not sk.has_flag(FLAGS.NTT_STATE):
            raise errors.NTTStateError(expected=True)
        if not sk.has_flag(FLAGS.MONTGOMERY_STATE):
            raise errors.MontgomeryStateError(expected=True)

        sk_new_data = [s.clone() for s in sk.data]
        self.nttCtx.intt(sk_new_data)
        sk_new_data = [codec.conjugate(s) for s in sk_new_data]
        self.nttCtx.ntt(sk_new_data)
        sk_rotated = SecretKey(
            data=sk_new_data,
            flags=FLAGS.MONTGOMERY_STATE | FLAGS.NTT_STATE,
            level=0,
            # following is metadata (not required args)
            logN=self.ckksCfg.logN,
            creator_hash=self.hash,
        )
        rotk = ConjugationKey.wrap(
            self.create_key_switching_key(sk_rotated, sk)
        )
        return rotk

    # @strictype # enable when debugging
    def conjugate(self, ct: Ciphertext, conjk: ConjugationKey) -> Ciphertext:
        level = ct.level
        conj_ct_data = [
            [codec.conjugate(d) for d in ct_data] for ct_data in ct.data
        ]

        conj_ct_sk = Ciphertext(
            data=conj_ct_data,
            level=level,
            # following is metadata (not required args)
            logN=self.ckksCfg.logN,
            creator_hash=self.hash,
        )

        conj_ct = self.switch_key(conj_ct_sk, conjk)
        return conj_ct

    # @strictype # enable when debugging
    def negate(self, ct: Ciphertext, inplace: bool = False) -> Ciphertext:
        if not inplace:
            ct = ct.clone()

        for part in ct.data:
            for d in part:
                d *= -1
            self.nttCtx.make_signed(part, ct.level)

        return ct

    # -------------------------------------------------------------------------------------------
    # scalar ops.
    # -------------------------------------------------------------------------------------------

    # @strictype # enable when debugging
    def pc_add(
        self,
        pt: Plaintext,
        ct: Ciphertext,
        inplace: bool = False,
    ):
        # process cache
        if str(self.pc_add) not in pt.cache[ct.level]:
            m = pt.src * math.sqrt(self.deviations[ct.level + 1])
            pt_ = self.encode(m, ct.level, scale=pt.scale)
            pt_ = self.nttCtx.tile_unsigned(pt_, ct.level)
            self.nttCtx.mont_enter_scale(pt_, ct.level)
            pt.cache[ct.level][str(self.pc_add)] = pt_
        pt_ = pt.cache[ct.level][
            str(self.pc_add)
        ]  # todo does rewrite to auto trace impact performance?

        # process ct

        new_ct = ct.clone() if not inplace else ct

        self.nttCtx.mont_enter(new_ct.data[0], ct.level)
        new_d0 = self.nttCtx.mont_add(pt_, new_ct.data[0], ct.level)
        self.nttCtx.mont_reduce(new_d0, ct.level)
        self.nttCtx.reduce_2q(new_d0, ct.level)
        new_ct.data[0] = new_d0
        return new_ct

    # @strictype # enable when debugging
    def pc_mult(
        self,
        pt: Plaintext,
        ct: Ciphertext,
        inplace: bool = False,  # its always inplace if this function is a atomic op, this flag will only control if do ntt/intt on original data
        post_rescale=True,
    ):
        # process cache
        if str(self.pc_mult) not in pt.cache[ct.level]:
            m = pt.src * math.sqrt(self.deviations[ct.level + 1])
            pt_ = self.encode(m, ct.level, scale=pt.scale)
            pt_ = self.nttCtx.tile_unsigned(pt_, ct.level)
            self.nttCtx.enter_ntt(pt_, ct.level)
            pt.cache[ct.level][str(self.pc_mult)] = pt_
        pt_ = pt.cache[ct.level][
            str(self.pc_mult)
        ]  # todo does rewrite to auto trace impact performance?

        # process ct

        new_ct = ct if inplace else ct.clone()

        self.nttCtx.enter_ntt(new_ct.data[0], ct.level)
        self.nttCtx.enter_ntt(new_ct.data[1], ct.level)

        new_d0 = self.nttCtx.mont_mult(pt_, new_ct.data[0], ct.level)
        new_d1 = self.nttCtx.mont_mult(pt_, new_ct.data[1], ct.level)

        self.nttCtx.intt_exit_reduce(new_d0, ct.level)
        self.nttCtx.intt_exit_reduce(new_d1, ct.level)

        new_ct.data[0] = new_d0
        new_ct.data[1] = new_d1

        if post_rescale:
            new_ct = self.rescale(new_ct)
        return new_ct

    # @strictype # enable when debugging
    def mult_int_scalar(self, ct: Ciphertext, scalar) -> Ciphertext:
        device_len = len(ct.data[0])

        int_scalar = int(scalar)
        mont_scalar = [
            (int_scalar * self.montCtx.R) % qi for qi in self.montCtx.q
        ]

        dest = self.rnsPart.destination_arrays[ct.level]

        partitioned_mont_scalar = [
            [mont_scalar[i] for i in desti] for desti in dest
        ]
        tensorized_scalar = []
        for device_id in range(device_len):
            scal_tensor = torch.tensor(
                partitioned_mont_scalar[device_id],
                dtype=self.ckksCfg.torch_dtype,
                device=self.nttCtx.devices[device_id],
            )
            tensorized_scalar.append(scal_tensor)

        new_ct = ct.clone()
        new_data = new_ct.data
        for i in [0, 1]:
            self.nttCtx.mont_enter_scalar(
                new_data[i], tensorized_scalar, ct.level
            )
            self.nttCtx.reduce_2q(new_data[i], ct.level)

        return new_ct

    # @strictype # enable when debugging
    def mult_scalar(
        self, ct: Ciphertext, scalar: ScalarMessageType, inplace: bool = False
    ) -> Ciphertext:
        device_len = len(ct.data[0])

        scaled_scalar = int(
            scalar * self.ckksCfg.scale * np.sqrt(self.deviations[ct.level + 1])
            + 0.5
        )

        mont_scalar = [
            (scaled_scalar * self.montCtx.R) % qi for qi in self.montCtx.q
        ]

        dest = self.rnsPart.destination_arrays[ct.level]

        partitioned_mont_scalar = [
            [mont_scalar[i] for i in dest_i] for dest_i in dest
        ]
        tensorized_scalar = []
        for device_id in range(device_len):
            scal_tensor = torch.tensor(
                partitioned_mont_scalar[device_id],
                dtype=self.ckksCfg.torch_dtype,
                device=self.nttCtx.devices[device_id],
            )
            tensorized_scalar.append(scal_tensor)

        new_ct = ct if inplace else ct.clone()
        new_data = new_ct.data

        for i in [0, 1]:
            self.nttCtx.mont_enter_scalar(
                new_data[i], tensorized_scalar, ct.level
            )
            self.nttCtx.reduce_2q(new_data[i], ct.level)

        return self.rescale(new_ct)

    # @strictype # enable when debugging
    def add_scalar(
        self, ct: Ciphertext, scalar: ScalarMessageType, inplace: bool = False
    ) -> Ciphertext:
        device_len = len(ct.data[0])

        scaled_scalar = int(
            scalar * self.ckksCfg.scale * self.deviations[ct.level] + 0.5
        )

        if self.norm == "backward":
            scaled_scalar *= self.ckksCfg.N

        scaled_scalar *= self.ckksCfg.int_scale

        scaled_scalar = [scaled_scalar % qi for qi in self.montCtx.q]

        dest = self.rnsPart.destination_arrays[ct.level]

        partitioned_mont_scalar = [
            [scaled_scalar[i] for i in desti] for desti in dest
        ]
        tensorized_scalar = []
        for device_id in range(device_len):
            scal_tensor = torch.tensor(
                partitioned_mont_scalar[device_id],
                dtype=self.ckksCfg.torch_dtype,
                device=self.nttCtx.devices[device_id],
            )
            tensorized_scalar.append(scal_tensor)

        new_ct = ct if inplace else ct.clone()
        new_data = new_ct.data

        dc = [d[:, 0] for d in new_data[0]]
        for device_id in range(device_len):
            dc[device_id] += tensorized_scalar[device_id]

        self.nttCtx.reduce_2q(new_data[0], ct.level)

        return new_ct

    # -------------------------------------------------------------------------------------------
    # message ops.
    # -------------------------------------------------------------------------------------------

    # @strictype # enable when debugging
    def mc_mult(
        self, m, ct: Ciphertext, inplace: bool = False, post_rescale=True
    ) -> Ciphertext:
        return self.pc_mult(
            pt=Plaintext(m),
            ct=ct,
            inplace=inplace,
            post_rescale=post_rescale,
        )

    # @strictype # enable when debugging
    def mc_add(self, m, ct: Ciphertext, inplace: bool = False) -> Ciphertext:
        return self.pc_add(pt=Plaintext(m), ct=ct, inplace=inplace)

    # -------------------------------------------------------------------------------------------
    # Misc.
    # -------------------------------------------------------------------------------------------

    def align_level(self, ct0, ct1):
        level_diff = ct0.level - ct1.level
        if level_diff < 0:
            new_ct0 = self.level_up(ct0, ct1.level)
            return new_ct0, ct1
        elif level_diff > 0:
            new_ct1 = self.level_up(ct1, ct0.level)
            return ct0, new_ct1
        else:
            return ct0, ct1

    def refresh(self):
        # Refreshes the rng state.
        self.rng.refresh()

    def reduce_error(self, ct):
        # Reduce the accumulated error in the cipher text.
        return self.mult_scalar(ct, 1.0)

    # @strictype # enable when debugging
    def sum(self, ct: Ciphertext) -> Ciphertext:
        new_ct = ct.clone()
        for roti in range(self.ckksCfg.logN - 1):
            rotk = self.rotk[roti]
            rot_ct = self.rotate_single(new_ct, rotk)
            new_ct = self.cc_add(rot_ct, new_ct)
        return new_ct

    # @strictype # enable when debugging
    def mean(self, ct: Ciphertext, *, alpha=1):
        # Divide by num_slots.
        # The cipher text is refreshed here, and hence
        # doesn't beed to be refreshed at roti=0 in the loop.
        new_ct = self.mc_mult(m=1 / self.num_slots / alpha, ct=ct)
        for roti in range(self.ckksCfg.logN - 1):
            rotk = self.rotk[roti]
            rot_ct = self.rotate_single(new_ct, rotk)
            new_ct = self.cc_add(rot_ct, new_ct)
        return new_ct

    # @strictype # enable when debugging
    def cov(
        self,
        ct_a: Ciphertext,
        ct_b: Ciphertext,
        evk: EvaluationKey = None,
    ) -> Ciphertext:
        evk = evk or self.evk

        cta_mean = self.mean(ct_a)
        ctb_mean = self.mean(ct_b)

        cta_dev = self.cc_sub(ct_a, cta_mean)
        ctb_dev = self.cc_sub(ct_b, ctb_mean)

        ct_cov = self.mc_mult(
            ct=self.cc_mult(cta_dev, ctb_dev, evk), m=1 / (self.num_slots - 1)
        )
        return ct_cov

    # @strictype # enable when debugging
    def pow(
        self, ct: Ciphertext, power: int, evk: EvaluationKey = None
    ) -> Ciphertext:
        evk = evk or self.evk

        current_exponent = 2
        pow_list = [ct]
        while current_exponent <= power:
            current_ct = pow_list[-1]
            new_ct = self.cc_mult(current_ct, current_ct, evk)
            pow_list.append(new_ct)
            current_exponent *= 2

        remaining_exponent = power - current_exponent // 2
        new_ct = pow_list[-1]

        while remaining_exponent > 0:
            pow_ind = math.floor(math.log2(remaining_exponent))
            pow_term = pow_list[pow_ind]
            # align level
            new_ct, pow_term = self.align_level(new_ct, pow_term)
            new_ct = self.cc_mult(new_ct, pow_term, evk)
            remaining_exponent -= 2**pow_ind

        return new_ct

    # @strictype # enable when debugging
    def sqrt(
        self, ct: Ciphertext, evk: EvaluationKey = None, e=0.0001, alpha=0.0001
    ) -> Ciphertext:
        a = ct.clone()
        b = ct.clone()
        evk = evk or self.evk

        while e <= 1 - alpha:
            k = float(np.roots([1 - e**3, -6 + 6 * e**2, 9 - 9 * e])[1])
            t = self.mult_scalar(a, k, evk)
            b0 = self.add_scalar(t, -3)
            b1 = self.mult_scalar(b, (k**0.5) / 2, evk)
            b = self.cc_mult(b0, b1, evk)

            a0 = self.mult_scalar(a, (k**3) / 4)
            t = self.add_scalar(a, -3 / k)
            a1 = self.cc_mult(t, t, evk)
            a = self.cc_mult(a0, a1, evk)
            e = k * (3 - k) ** 2 / 4

        return b

    #### -------------------------------------------------------------------------------------------
    ####  Statistics
    #### -------------------------------------------------------------------------------------------

    # @strictype # enable when debugging
    def randn(
        self,
        amin=-1,
        amax=1,
        decimal_places: int = 10,
        level=0,
        return_src=False,
    ) -> np.array:
        def integral_bits_available(self):
            base_prime = self.base_prime
            max_bits = math.floor(math.log2(base_prime))
            integral_bits = max_bits - self.ckksCfg.scale_bits
            return integral_bits

        if amin is None:
            amin = -(2 ** integral_bits_available())

        if amax is None:
            amax = 2 ** integral_bits_available()

        base = 10**decimal_places
        a = (
            np.random.randint(amin * base, amax * base, self.ckksCfg.N // 2)
            / base
        )
        b = (
            np.random.randint(amin * base, amax * base, self.ckksCfg.N // 2)
            / base
        )

        sample = a + b * 1j

        encrypted = self.encodecrypt(
            m=sample,
            level=level,
        )

        return (encrypted, sample) if return_src else encrypted

    # @strictype # enable when debugging
    def var(
        self,
        ct: Ciphertext,
        evk: EvaluationKey = None,
        *,
        post_relin=False,
    ) -> Ciphertext:
        evk = evk or self.evk
        ct_mean = self.mean(ct=ct)
        dev = self.cc_sub(ct, ct_mean)
        dev = self.square(ct=dev, evk=evk, post_relin=post_relin)
        if not post_relin:
            dev = self.relinearize(ct_triplet=dev, evk=evk)
        ct_var = self.mean(ct=dev)
        return ct_var

    # @strictype # enable when debugging
    def std(
        self,
        ct: Ciphertext,
        evk: EvaluationKey = None,
        post_relin=False,
    ) -> Ciphertext:
        evk = evk or self.evk
        ct_var = self.var(ct=ct, evk=evk, post_relin=post_relin)
        ct_std = self.sqrt(ct=ct_var, evk=evk)
        return ct_std
