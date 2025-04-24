import torch
from vdtoys.mvc import strictype

from tiberate.fhe.encdec import rotate
from tiberate.fhe.engine import CkksEngine, errors
from tiberate.typing import *


class CkksEngineMPCExtension(CkksEngine):
    # -------------------------------------------------------------------------------------------
    # Multiparty.
    # -------------------------------------------------------------------------------------------

    def multiparty_public_crs(self, pk: PublicKey):
        crs = self.clone(pk).data[1]
        return crs

    @strictype
    def multiparty_create_public_key(
        self, sk: SecretKey, a=None, include_special=False
    ) -> PublicKey:
        if include_special and not sk.has_flag(FLAGS.INCLUDE_SPECIAL):
            raise errors.SecretKeyNotIncludeSpecialPrime()
        mult_type = -2 if include_special else -1

        level = 0
        e = self.rng.discrete_gaussian(repeats=1)
        e = self.nttCtx.tile_unsigned(e, level, mult_type)

        self.nttCtx.enter_ntt(e, level, mult_type)
        repeats = (
            self.ckksCtx.num_special_primes
            if sk.has_flag(FLAGS.INCLUDE_SPECIAL)
            else 0
        )

        if a is None:
            a = self.rng.randint(
                self.nttCtx.q_prepack[mult_type][level][0], repeats=repeats
            )

        sa = self.nttCtx.mont_mult(a, sk.data, 0, mult_type)
        pk0 = self.nttCtx.mont_sub(e, sa, 0, mult_type)
        pk = PublicKey(
            data=[pk0, a],
            flags=(FLAGS.INCLUDE_SPECIAL if include_special else FLAGS(0))
            | FLAGS.NTT_STATE
            | FLAGS.MONTGOMERY_STATE,
            level=level,
            hash=self.hash,
        )
        return pk

    def multiparty_create_collective_public_key(
        self, pks: list[DataStruct]
    ) -> PublicKey:
        (
            data,
            include_special,
            ntt_state,
            montgomery_state,
            origin,
            level,
            hash_,
            version,
        ) = pks[0]
        mult_type = -2 if include_special else -1
        b = [b.clone() for b in data[0]]  # num of gpus
        a = [a.clone() for a in data[1]]

        for pk in pks[1:]:
            b = self.nttCtx.mont_add(b, pk.data[0], lvl=0, mult_type=mult_type)

        build_flags = FLAGS(0)
        if include_special:
            build_flags |= FLAGS.INCLUDE_SPECIAL
        if ntt_state:
            build_flags |= FLAGS.NTT_STATE
        if montgomery_state:
            build_flags |= FLAGS.MONTGOMERY_STATE
        cpk = PublicKey(
            data=(b, a),
            flags=build_flags,
            level=level,
            hash=self.hash,
        )
        return cpk

    @strictype
    def multiparty_decrypt_head(self, ct: Ciphertext, sk: SecretKey):
        if ct.has_flag(FLAGS.NTT_STATE):
            raise errors.NTTStateError(expected=False)
        if ct.has_flag(FLAGS.MONTGOMERY_STATE):
            raise errors.MontgomeryStateError(expected=False)
        if not sk.has_flag(FLAGS.NTT_STATE):
            raise errors.NTTStateError(expected=True)
        if not sk.has_flag(FLAGS.MONTGOMERY_STATE):
            raise errors.MontgomeryStateError(expected=True)

        level = ct.level

        ct0 = ct.data[0][0]
        a = ct.data[1][0].clone()

        self.nttCtx.enter_ntt([a], level)

        sk_data = sk.data[0][self.nttCtx.starts[level][0] :]

        sa = self.nttCtx.mont_mult([a], [sk_data], level)
        self.nttCtx.intt_exit(sa, level)

        pt = self.nttCtx.mont_add([ct0], sa, level)

        return pt

    @strictype
    def multiparty_decrypt_partial(
        self, ct: Ciphertext, sk: SecretKey
    ) -> DataStruct:
        if ct.has_flag(FLAGS.NTT_STATE):
            raise errors.NTTStateError(expected=False)
        if ct.has_flag(FLAGS.MONTGOMERY_STATE):
            raise errors.MontgomeryStateError(expected=False)
        if not sk.has_flag(FLAGS.NTT_STATE):
            raise errors.NTTStateError(expected=True)
        if not sk.has_flag(FLAGS.MONTGOMERY_STATE):
            raise errors.MontgomeryStateError(expected=True)

        a = ct.data[1][0].clone()

        self.nttCtx.enter_ntt([a], ct.level)

        sk_data = sk.data[0][self.nttCtx.starts[ct.level][0] :]

        sa = self.nttCtx.mont_mult([a], [sk_data], ct.level)
        self.nttCtx.intt_exit(sa, ct.level)

        return sa

    def multiparty_decrypt_fusion(
        self, pcts: list, level=0, include_special=False
    ):
        pt = [x.clone() for x in pcts[0]]
        for pct in pcts[1:]:
            pt = self.nttCtx.mont_add(pt, pct, level)

        self.nttCtx.reduce_2q(pt, level)

        base_at = (
            -self.ckksCtx.num_special_primes - 1 if include_special else -1
        )

        base = pt[0][base_at][None, :]
        scaler = pt[0][0][None, :]

        final_scalar = self.final_scalar[level]
        scaled = self.nttCtx.mont_sub([base], [scaler], -1)
        self.nttCtx.mont_enter_scalar(scaled, [final_scalar], -1)
        self.nttCtx.reduce_2q(scaled, -1)
        self.nttCtx.make_signed(scaled, -1)

        m = self.decode(m=scaled, level=level)

        return m

    #### -------------------------------------------------------------------------------------------
    #### Multiparty. ROTATION
    #### -------------------------------------------------------------------------------------------

    @strictype
    def multiparty_create_key_switching_key(
        self, sk_src: SecretKey, sk_dst: SecretKey, a=None
    ) -> KeySwitchKey:
        if not sk_src.has_flag(FLAGS.NTT_STATE):
            raise errors.NTTStateError(expected=True)
        if not sk_src.has_flag(FLAGS.MONTGOMERY_STATE):
            raise errors.MontgomeryStateError(expected=True)
        if not sk_dst.has_flag(FLAGS.NTT_STATE):
            raise errors.NTTStateError(expected=True)
        if not sk_dst.has_flag(FLAGS.MONTGOMERY_STATE):
            raise errors.MontgomeryStateError(expected=True)

        level = 0

        stops = self.nttCtx.stops[-1]
        Psk_src = [
            sk_src.data[di][: stops[di]].clone()
            for di in range(self.nttCtx.num_devices)
        ]

        self.nttCtx.mont_enter_scalar(Psk_src, self.mont_PR, level)

        ksk = [[] for _ in range(self.nttCtx.rnsPart.num_partitions + 1)]
        for device_id in range(self.nttCtx.num_devices):
            for part_id, part in enumerate(
                self.nttCtx.rnsPart.p[level][device_id]
            ):
                global_part_id = self.nttCtx.rnsPart.part_allocations[
                    device_id
                ][part_id]

                crs = a[global_part_id] if a else None
                pk = self.multiparty_create_public_key(
                    sk_dst, include_special=True, a=crs
                )
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

                # Name the pk.
                pk_name = f"key switch key part index {global_part_id}"
                pk = pk._replace(origin=pk_name)

                ksk[global_part_id] = pk

        return KeySwitchKey(
            data=ksk,
            flags=FLAGS.NTT_STATE
            | FLAGS.MONTGOMERY_STATE
            | FLAGS.INCLUDE_SPECIAL,
            level=level,
            hash=self.hash,
        )

    @strictype
    def multiparty_create_rotation_key(
        self, sk: SecretKey, delta: int, a=None
    ) -> RotationKey:
        sk_new_data = [s.clone() for s in sk.data]
        self.nttCtx.intt(sk_new_data)
        sk_new_data = [rotate(s, delta) for s in sk_new_data]
        self.nttCtx.ntt(sk_new_data)
        sk_rotated = DataStruct(
            data=sk_new_data,
            flags=FLAGS.NTT_STATE | FLAGS.MONTGOMERY_STATE,
            level=0,
            hash=self.hash,
        )
        rotk = RotationKey.wrap(
            self.multiparty_create_key_switching_key(sk_rotated, sk, a=a),
            delta=delta,
        )
        return rotk

    @strictype
    def multiparty_generate_rotation_key(
        self, rotks: list[RotationKey]
    ) -> RotationKey:
        crotk = self.clone(rotks[0])
        for rotk in rotks[1:]:
            for ksk_idx in range(len(rotk.data)):
                update_parts = self.nttCtx.mont_add(
                    crotk.data[ksk_idx].data[0], rotk.data[ksk_idx].data[0]
                )
                crotk.data[ksk_idx].data[0][0].copy_(
                    update_parts[0], non_blocking=True
                )
        return crotk

    @strictype
    def generate_rotation_crs(self, rotk: RotationKey | KeySwitchKey):
        crss = []
        for ksk in rotk.data:
            crss.append(ksk.data[1])
        return crss

    #### -------------------------------------------------------------------------------------------
    #### Multiparty. GALOIS
    #### -------------------------------------------------------------------------------------------

    @strictype
    def generate_galois_crs(self, galk: GaloisKey):
        crs_s = []
        for rotk in galk.data:
            crss = [ksk.data[1] for ksk in rotk.data]
            crs_s.append(crss)
        return crs_s

    @strictype
    def multiparty_create_galois_key(self, sk: SecretKey, a: list) -> GaloisKey:
        galois_deltas = [2**i for i in range(self.ckksCtx.logN - 1)]
        galois_key_parts = [
            self.multiparty_create_rotation_key(
                sk, galois_deltas[idx], a=a[idx]
            )
            for idx in range(len(galois_deltas))
        ]

        galois_key = GaloisKey(
            data=galois_key_parts,
            flags=FLAGS.NTT_STATE
            | FLAGS.MONTGOMERY_STATE
            | FLAGS.INCLUDE_SPECIAL,
            level=0,
            hash=self.hash,
        )
        return galois_key

    def multiparty_generate_galois_key(
        self, galks: list[DataStruct]
    ) -> DataStruct:
        cgalk = self.clone(galks[0])
        for galk in galks[1:]:  # galk
            for rotk_idx in range(len(galk.data)):  # rotk
                for ksk_idx in range(len(galk.data[rotk_idx].data)):  # ksk
                    update_parts = self.nttCtx.mont_add(
                        cgalk.data[rotk_idx].data[ksk_idx].data[0],
                        galk.data[rotk_idx].data[ksk_idx].data[0],
                    )
                    cgalk.data[rotk_idx].data[ksk_idx].data[0][0].copy_(
                        update_parts[0], non_blocking=True
                    )
        return cgalk

    #### -------------------------------------------------------------------------------------------
    #### Multiparty. Evaluation Key
    #### -------------------------------------------------------------------------------------------

    def multiparty_sum_evk_share(self, evks_share: list[DataStruct]):
        evk_sum = self.clone(evks_share[0])
        for evk_share in evks_share[1:]:
            for ksk_idx in range(len(evk_sum.data)):
                update_parts = self.nttCtx.mont_add(
                    evk_sum.data[ksk_idx].data[0],
                    evk_share.data[ksk_idx].data[0],
                )
                for dev_id in range(len(update_parts)):
                    evk_sum.data[ksk_idx].data[0][dev_id].copy_(
                        update_parts[dev_id], non_blocking=True
                    )

        return evk_sum

    @strictype
    def multiparty_mult_evk_share_sum(
        self, evk_sum: DataStruct, sk: SecretKey
    ) -> DataStruct:
        evk_sum_mult = self.clone(evk_sum)

        for ksk_idx in range(len(evk_sum.data)):
            update_part_b = self.nttCtx.mont_mult(
                evk_sum_mult.data[ksk_idx].data[0], sk.data
            )
            update_part_a = self.nttCtx.mont_mult(
                evk_sum_mult.data[ksk_idx].data[1], sk.data
            )
            for dev_id in range(len(update_part_b)):
                evk_sum_mult.data[ksk_idx].data[0][dev_id].copy_(
                    update_part_b[dev_id], non_blocking=True
                )
                evk_sum_mult.data[ksk_idx].data[1][dev_id].copy_(
                    update_part_a[dev_id], non_blocking=True
                )

        return evk_sum_mult

    def multiparty_sum_evk_share_mult(
        self, evk_sum_mult: list[DataStruct]
    ) -> DataStruct:
        cevk = self.clone(evk_sum_mult[0])
        for evk in evk_sum_mult[1:]:
            for ksk_idx in range(len(cevk.data)):
                update_part_b = self.nttCtx.mont_add(
                    cevk.data[ksk_idx].data[0], evk.data[ksk_idx].data[0]
                )
                update_part_a = self.nttCtx.mont_add(
                    cevk.data[ksk_idx].data[1], evk.data[ksk_idx].data[1]
                )
                for dev_id in range(len(update_part_b)):
                    cevk.data[ksk_idx].data[0][dev_id].copy_(
                        update_part_b[dev_id], non_blocking=True
                    )
                    cevk.data[ksk_idx].data[1][dev_id].copy_(
                        update_part_a[dev_id], non_blocking=True
                    )
        return cevk
