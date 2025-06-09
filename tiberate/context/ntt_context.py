import numpy as np
import torch
from loguru import logger

from tiberate.config.ckks_config import CkksConfig
from tiberate.context.mont_context import MontgomeryContext
from tiberate.context.rns_partition import RnsPartition

# ------------------------------------------------------------------------------------------
# NTT pre-compute.
# ------------------------------------------------------------------------------------------


def primitive_root_2N(q, N):
    _2N = 2 * N
    K = (q - 1) // _2N
    for x in range(2, N):
        g = pow(x, K, q)
        h = pow(g, N, q)
        if h != 1:
            break
    return g


def psi_power_series(psi, N, q):
    series = [1]
    for i in range(N - 1):
        series.append(series[-1] * psi % q)
    return series


def bit_rev_psi(q, logN):
    N = 2**logN
    psi = [primitive_root_2N(qi, N) for qi in q]
    # Bit-reverse index.
    ind = range(N)
    brind = [bit_reverse(i, logN) for i in ind]
    # The psi power and the indices are the same.
    return [pow(psi, brpower, q) for brpower in brind]


def psi_bank(q, logN):
    N = 2**logN
    psi = [primitive_root_2N(qi, N) for qi in q]
    ipsi = [pow(psii, -1, qi) for psii, qi in zip(psi, q)]
    psi_series = [psi_power_series(psii, N, qi) for psii, qi in zip(psi, q)]
    ipsi_series = [psi_power_series(ipsii, N, qi) for ipsii, qi in zip(ipsi, q)]
    return psi_series, ipsi_series


def bit_reverse(a, nbits):
    format_string = f"0{nbits}b"
    binary_string = f"{a:{format_string}}"
    reverse_binary_string = binary_string[::-1]
    return int(reverse_binary_string, 2)


def bit_reverse_order_index(logN):
    N = 2**logN
    # Note that for a bit reversing, forward and backward permutations are the same.
    # i.e., don't worry about which direction.
    revi = np.array([bit_reverse(i, logN) for i in range(N)], dtype=np.int32)
    return revi


def get_psi(q, logN, my_dtype):
    np_dtype_dict = {
        np.int32: np.int32,
        np.int64: np.int64,
        30: np.int32,
        62: np.int64,
    }
    dtype = np_dtype_dict[my_dtype]
    psi, ipsi = psi_bank(q, logN)
    bit_reverse_index = bit_reverse_order_index(logN)
    psi = np.array(psi, dtype=dtype)[:, bit_reverse_index]
    ipsi = np.array(ipsi, dtype=dtype)[:, bit_reverse_index]
    return psi, ipsi


def paint_butterfly_forward(logN):
    N = 2**logN
    t = N
    painted_even = np.zeros((logN, N), dtype=np.bool_)
    painted_odd = np.zeros((logN, N), dtype=np.bool_)
    painted_psi = np.zeros((logN, N // 2), dtype=np.int32)
    for logm in range(logN):
        m = 2**logm
        t //= 2
        psi_ind = 0
        for i in range(m):
            j1 = 2 * i * t
            j2 = j1 + t - 1
            Sind = m + i
            for j in range(j1, j2 + 1):
                Uind = j
                Vind = j + t
                painted_even[logm, Uind] = True
                painted_odd[logm, Vind] = True
                painted_psi[logm, psi_ind] = Sind
                psi_ind += 1
    painted_eveni = np.where(painted_even)[1].reshape(logN, -1)
    painted_oddi = np.where(painted_odd)[1].reshape(logN, -1)
    return painted_eveni, painted_oddi, painted_psi


def paint_butterfly_backward(logN):
    N = 2**logN
    t = 1
    painted_even = np.zeros((logN, N), dtype=np.bool_)
    painted_odd = np.zeros((logN, N), dtype=np.bool_)
    painted_psi = np.zeros((logN, N // 2), dtype=np.int32)
    for logm in range(logN, 0, -1):
        level = logN - logm
        m = 2**logm
        j1 = 0
        h = m // 2
        psi_ind = 0
        for i in range(h):
            j2 = j1 + t - 1
            Sind = h + i
            for j in range(j1, j2 + 1):
                Uind = j
                Vind = j + t
                # Paint
                painted_even[level, Uind] = True
                painted_odd[level, Vind] = True
                painted_psi[level, psi_ind] = Sind
                psi_ind += 1
            j1 += 2 * t
        t *= 2
    painted_eveni = np.where(painted_even)[1].reshape(logN, -1)
    painted_oddi = np.where(painted_odd)[1].reshape(logN, -1)
    return painted_eveni, painted_oddi, painted_psi


def copy_to_devices(variable, dtype, devices):

    return [
        torch.tensor(variable, dtype=dtype, device=device) for device in devices
    ]


class NTTContext:
    def __init__(
        self,
        ckks_config: CkksConfig,
        index_type=torch.int32,
        devices=None,
    ):
        # Set devices first.
        if devices is None:
            devices = ["cuda:0"]
            logger.info(f"Device not specified. Using default {devices}.")

        self.devices = devices
        self.num_devices = len(self.devices)

        # Transfer input parameters.
        self.index_type = index_type

        self.ckksCfg = ckks_config
        self.montCtx = MontgomeryContext.from_ckks_config(ckks_config)
        self.num_ordinary_primes = self.ckksCfg.num_scales + 1
        self.num_special_primes = self.ckksCfg.num_special_primes

        self.rnsPart = RnsPartition(
            self.num_ordinary_primes, self.num_special_primes, self.num_devices
        )

        # ==============================================
        # generate_paints()
        self.N_inv = [pow(self.ckksCfg.N, -1, qi) for qi in self.ckksCfg.q]

        # psi and psi_inv.
        psi, psi_inv = get_psi(
            self.ckksCfg.q, self.ckksCfg.logN, self.ckksCfg.buffer_bit_length
        )

        # Paints.
        (
            self.forward_even_indices,
            self.forward_odd_indices,
            forward_psi_paint,
        ) = paint_butterfly_forward(self.ckksCfg.logN)
        (
            self.backward_even_indices,
            self.backward_odd_indices,
            backward_psi_paint,
        ) = paint_butterfly_backward(self.ckksCfg.logN)

        # Pre-painted psi and ipsi.
        self.forward_psi = psi[..., forward_psi_paint.ravel()].reshape(
            -1, *forward_psi_paint.shape
        )
        self.backward_psi_inv = psi_inv[
            ..., backward_psi_paint.ravel()
        ].reshape(-1, *backward_psi_paint.shape)

        # =============================================

        self.prepare_parameters()

        self.qlists = [qi.tolist() for qi in self.q]

        astop_special = [
            len(d) for d in self.rnsPart.destination_arrays_with_special[0]
        ]
        astop_ordinary = [len(d) for d in self.rnsPart.destination_arrays[0]]
        self.starts = self.rnsPart.diff

        self.stops = [astop_special, astop_ordinary]

        self.generate_parts_pack()
        self.pre_package()

    @property
    def num_levels(self) -> int:
        return self.ckksCfg.num_scales + 1

    # -------------------------------------------------------------------------------------------------
    # Arrange according to partitioning scheme input variables, and copy to GPUs for fast access.
    # -------------------------------------------------------------------------------------------------

    def partition_variable(self, variable):
        np_v = np.array(variable, dtype=self.ckksCfg.numpy_dtype)

        v_special = []
        dest = self.rnsPart.d_special
        for dev_id in range(self.num_devices):
            d = dest[dev_id]
            parted_v = np_v[d]
            v_special.append(
                torch.from_numpy(parted_v).to(self.devices[dev_id])
            )

        return v_special

    def psi_enter(self):
        Rs = self.Rs
        ql = self.ql
        qh = self.qh
        kl = self.kl
        kh = self.kh

        p = self.psi

        a = [psi.view(psi.size(0), -1) for psi in p]

        torch.ops.tiberate_ntt_ops.mont_enter(a, Rs, ql, qh, kl, kh)

        p = self.ipsi
        a = [psi.view(psi.size(0), -1) for psi in p]
        torch.ops.tiberate_ntt_ops.mont_enter(a, Rs, ql, qh, kl, kh)

    def Ninv_enter(self):
        self.Ninv = [
            (self.N_inv[i] * self.montCtx.R) % self.ckksCfg.q[i]
            for i in range(len(self.ckksCfg.q))
        ]

    def prepare_parameters(self):
        scale = 2**self.ckksCfg.scale_bits
        self.Rs_scale = self.partition_variable(
            [
                (Rs * scale) % q
                for Rs, q in zip(self.montCtx.R_square, self.ckksCfg.q)
            ]
        )

        self.Rs = self.partition_variable(self.montCtx.R_square)

        self.q = self.partition_variable(self.montCtx.q)
        self._2q = self.partition_variable(self.montCtx.q_double)
        self.ql = self.partition_variable(self.montCtx.q_lower_bits)
        self.qh = self.partition_variable(self.montCtx.q_higher_bits)
        self.kl = self.partition_variable(self.montCtx.k_lower_bits)
        self.kh = self.partition_variable(self.montCtx.k_higher_bits)

        self.even = copy_to_devices(
            self.forward_even_indices,
            dtype=self.index_type,
            devices=self.devices,
        )
        self.odd = copy_to_devices(
            self.forward_odd_indices,
            dtype=self.index_type,
            devices=self.devices,
        )
        self.ieven = copy_to_devices(
            self.backward_even_indices,
            dtype=self.index_type,
            devices=self.devices,
        )
        self.iodd = copy_to_devices(
            self.backward_odd_indices,
            dtype=self.index_type,
            devices=self.devices,
        )

        self.psi = self.partition_variable(self.forward_psi)
        self.ipsi = self.partition_variable(self.backward_psi_inv)

        self.Ninv_enter()
        self.Ninv = self.partition_variable(self.Ninv)

        self.psi_enter()

        self.mont_pack0 = [self.ql, self.qh, self.kl, self.kh]

        self.ntt_pack0 = [
            self.even,
            self.odd,
            self.psi,
            self._2q,
            self.ql,
            self.qh,
            self.kl,
            self.kh,
        ]

        self.intt_pack0 = [
            self.ieven,
            self.iodd,
            self.ipsi,
            self.Ninv,
            self._2q,
            self.ql,
            self.qh,
            self.kl,
            self.kh,
        ]

    def param_pack(self, param, astart, astop, remove_empty=True):
        pack = [
            param[dev_id][astart[dev_id] : astop[dev_id]]
            for dev_id in range(self.num_devices)
        ]

        def remove_empty_f(x):
            return [xi for xi in x if len(xi) > 0]

        if remove_empty:
            pack = remove_empty_f(pack)
        return pack

    def mont_pack(self, astart, astop, remove_empty=True):
        return [
            self.param_pack(param, astart, astop, remove_empty)
            for param in self.mont_pack0
        ]

    def ntt_pack(self, astart, astop, remove_empty=True):
        def remove_empty_f_x(x):
            return [xi for xi in x if len(xi) > 0]

        def remove_empty_f_xy(x, y):
            return [xi for xi, yi in zip(x, y) if len(yi) > 0]

        even_odd = self.ntt_pack0[:2]
        rest = [
            self.param_pack(param, astart, astop, remove_empty=False)
            for param in self.ntt_pack0[2:]
        ]

        if remove_empty:
            even_odd = [remove_empty_f_xy(eo, rest[0]) for eo in even_odd]
            rest = [remove_empty_f_x(r) for r in rest]

        return even_odd + rest

    def intt_pack(self, astart, astop, remove_empty=True):
        def remove_empty_f_x(x):
            return [xi for xi in x if len(xi) > 0]

        def remove_empty_f_xy(x, y):
            return [xi for xi, yi in zip(x, y) if len(yi) > 0]

        even_odd = self.intt_pack0[:2]
        rest = [
            self.param_pack(param, astart, astop, remove_empty=False)
            for param in self.intt_pack0[2:]
        ]

        if remove_empty:
            even_odd = [remove_empty_f_xy(eo, rest[0]) for eo in even_odd]
            rest = [remove_empty_f_x(r) for r in rest]

        return even_odd + rest

    def start_stop(self, lvl, mult_type):
        return self.starts[lvl], self.stops[mult_type]

    # -------------------------------------------------------------------------------------------------
    # Package by parts.
    # -------------------------------------------------------------------------------------------------

    def params_pack_device(self, device_id, astart, astop):
        starts = [0] * self.num_devices
        stops = [0] * self.num_devices

        starts[device_id] = astart
        stops[device_id] = astop + 1

        stst = [starts, stops]

        item = {}

        item["mont_pack"] = self.mont_pack(*stst)
        item["ntt_pack"] = self.ntt_pack(*stst)
        item["intt_pack"] = self.intt_pack(*stst)
        item["Rs"] = self.param_pack(self.Rs, *stst)
        item["Rs_scale"] = self.param_pack(self.Rs_scale, *stst)
        item["_2q"] = self.param_pack(self._2q, *stst)
        item["qlist"] = self.param_pack(self.qlists, *stst)

        return item

    def generate_parts_pack(self):
        blank_L_enter = [None] * self.num_devices

        self.parts_pack = []

        for device_id in range(self.num_devices):
            self.parts_pack.append({})

            for i in range(
                len(self.rnsPart.destination_arrays_with_special[0][device_id])
            ):
                self.parts_pack[device_id][i,] = self.params_pack_device(
                    device_id, i, i
                )

            for level in range(self.num_levels):
                for mult_type in [-1, -2]:
                    starts, stops = self.start_stop(level, mult_type)
                    astart = starts[device_id]

                    astop = stops[device_id] - 1

                    key = tuple(range(astart, astop + 1))

                    if len(key) > 0:
                        if key not in self.parts_pack[device_id]:
                            self.parts_pack[device_id][key] = (
                                self.params_pack_device(
                                    device_id, astart, astop
                                )
                            )

                for p in self.rnsPart.p_special[level][device_id]:
                    key = tuple(p)
                    if key not in self.parts_pack[device_id].keys():
                        astart = p[0]
                        astop = p[-1]
                        self.parts_pack[device_id][key] = (
                            self.params_pack_device(device_id, astart, astop)
                        )

        for device_id in range(self.num_devices):
            for level in range(self.num_levels):
                # We do basis extension for only ordinary parts.
                for part_index, part in enumerate(
                    self.rnsPart.destination_parts[level][device_id]
                ):
                    key = tuple(self.rnsPart.p[level][device_id][part_index])

                    # Check if Y and L are already calculated for this part.
                    if "Y_scalar" not in self.parts_pack[device_id][key].keys():
                        alpha = len(part)
                        m = [self.montCtx.q[idx] for idx in part]
                        L = [m[0]]

                        for i in range(1, alpha - 1):
                            L.append(L[-1] * m[i])

                        Y_scalar = []
                        L_scalar = []
                        for i in range(alpha - 1):
                            L_inv = pow(L[i], -1, m[i + 1])
                            L_inv_R = (L_inv * self.montCtx.R) % m[i + 1]
                            Y_scalar.append(L_inv_R)

                            if (i + 2) < alpha:
                                L_scalar.append([])
                                for j in range(i + 2, alpha):
                                    L_scalar[i].append(
                                        (L[i] * self.montCtx.R) % m[j]
                                    )

                        L_enter_devices = []
                        for target_device_id in range(self.num_devices):
                            dest = self.rnsPart.destination_arrays_with_special[
                                0
                            ][target_device_id]
                            q = [self.montCtx.q[idx] for idx in dest]
                            Rs = [self.montCtx.R_square[idx] for idx in dest]

                            L_enter = []
                            for i in range(alpha - 1):
                                L_enter.append([])
                                for j in range(len(dest)):
                                    L_Rs = (L[i] * Rs[j]) % q[j]
                                    L_enter[i].append(L_Rs)
                            L_enter_devices.append(L_enter)

                        device = self.devices[device_id]

                        if len(Y_scalar) > 0:
                            Y_scalar = torch.tensor(
                                Y_scalar,
                                dtype=self.ckksCfg.torch_dtype,
                                device=device,
                            )
                            self.parts_pack[device_id][key][
                                "Y_scalar"
                            ] = Y_scalar

                            for target_device_id in range(self.num_devices):
                                target_device = self.devices[target_device_id]

                                L_enter_devices[target_device_id] = [
                                    torch.tensor(
                                        Li,
                                        dtype=self.ckksCfg.torch_dtype,
                                        device=target_device,
                                    )
                                    for Li in L_enter_devices[target_device_id]
                                ]

                            self.parts_pack[device_id][key][
                                "L_enter"
                            ] = L_enter_devices

                        else:
                            self.parts_pack[device_id][key]["Y_scalar"] = None
                            self.parts_pack[device_id][key][
                                "L_enter"
                            ] = blank_L_enter

                        if len(L_scalar) > 0:
                            L_scalar = [
                                torch.tensor(
                                    Li,
                                    dtype=self.ckksCfg.torch_dtype,
                                    device=device,
                                )
                                for Li in L_scalar
                            ]
                            self.parts_pack[device_id][key][
                                "L_scalar"
                            ] = L_scalar
                        else:
                            self.parts_pack[device_id][key]["L_scalar"] = None

    # -------------------------------------------------------------------------------------------------
    # Pre-packaging.
    # -------------------------------------------------------------------------------------------------

    # Structure of self.ntt_prepack[mult_type][lvl][part]:
    #   Part Index | Description                      | Shape                              | Example(cuda:0, logN15, N=32768, 16 scales, 15 levels)
    # -------------|----------------------------------|------------------------------------|-------------------------------------------------------
    #       0      | Even values (duplicated)         | [n_dev, logN, N/2]                 | [1, 15, 16384]
    #       1      | Odd values (duplicated)          | [n_dev, logN, N/2]                 | [1, 15, 16384]
    #       2      | Psi (nth roots of unity)         | [n_dev, n_scales+1-lvl, logN, N/2] | [1, 16~1, 15, 16384]
    #       3      | 2q values                        | [n_dev, n_scales+1-lvl]            | [1, 16~1]
    #       4      | Lower bits of q (ql)             | [n_dev, n_scales+1-lvl]            | [1, 16~1]
    #       5      | Higher bits of q (qh)            | [n_dev, n_scales+1-lvl]            | [1, 16~1]
    #       6      | Lower bits of k (kl)             | [n_dev, n_scales+1-lvl]            | [1, 16~1]
    #       7      | Higher bits of k (kh)            | [n_dev, n_scales+1-lvl]            | [1, 16~1]

    def pre_package(self):

        self.mont_prepack = []
        self.ntt_prepack = []
        self.intt_prepack = []
        self.Rs_prepack = []
        self.Rs_scale_prepack = []
        self._2q_prepack = []

        # q_prepack is a list of lists, not tensors.
        # We need this for generating uniform samples.
        self.q_prepack = []

        for device_id in range(self.num_devices):
            mont_prepack = []
            ntt_prepack = []
            intt_prepack = []
            Rs_prepack = []
            Rs_scale_prepack = []
            _2q_prepack = []
            q_prepack = []
            for lvl in range(self.num_levels):
                mont_prepack_part = []
                ntt_prepack_part = []
                intt_prepack_part = []
                Rs_prepack_part = []
                Rs_scale_prepack_part = []
                _2q_prepack_part = []
                q_prepack_part = []
                for part in self.rnsPart.p_special[lvl][device_id]:
                    key = tuple(part)
                    item = self.parts_pack[device_id][key]

                    mont_prepack_part.append(item["mont_pack"])
                    ntt_prepack_part.append(item["ntt_pack"])
                    intt_prepack_part.append(item["intt_pack"])
                    Rs_prepack_part.append(item["Rs"])
                    Rs_scale_prepack_part.append(item["Rs_scale"])
                    _2q_prepack_part.append(item["_2q"])
                    q_prepack_part.append(item["qlist"])

                for mult_type in [-2, -1]:
                    starts, stops = self.start_stop(lvl, mult_type)
                    astart = starts[device_id]

                    astop = stops[device_id] - 1

                    key = tuple(range(astart, astop + 1))

                    if len(key) > 0:
                        item = self.parts_pack[device_id][key]

                        mont_prepack_part.append(item["mont_pack"])
                        ntt_prepack_part.append(item["ntt_pack"])
                        intt_prepack_part.append(item["intt_pack"])
                        Rs_prepack_part.append(item["Rs"])
                        Rs_scale_prepack_part.append(item["Rs_scale"])
                        _2q_prepack_part.append(item["_2q"])
                        q_prepack_part.append(item["qlist"])

                    else:
                        mont_prepack_part.append(None)
                        ntt_prepack_part.append(None)
                        intt_prepack_part.append(None)
                        Rs_prepack_part.append(None)
                        Rs_scale_prepack_part.append(None)
                        _2q_prepack_part.append(None)
                        q_prepack_part.append(None)

                mont_prepack.append(mont_prepack_part)
                ntt_prepack.append(ntt_prepack_part)
                intt_prepack.append(intt_prepack_part)
                Rs_prepack.append(Rs_prepack_part)
                Rs_scale_prepack.append(Rs_scale_prepack_part)
                _2q_prepack.append(_2q_prepack_part)
                q_prepack.append(q_prepack_part)

            self.mont_prepack.append(mont_prepack)
            self.ntt_prepack.append(ntt_prepack)
            self.intt_prepack.append(intt_prepack)
            self.Rs_prepack.append(Rs_prepack)
            self.Rs_scale_prepack.append(Rs_scale_prepack)
            self._2q_prepack.append(_2q_prepack)
            self.q_prepack.append(q_prepack)

        for mult_type in [-2, -1]:
            mont_prepack = []
            ntt_prepack = []
            intt_prepack = []
            Rs_prepack = []
            Rs_scale_prepack = []
            _2q_prepack = []
            q_prepack = []
            for lvl in range(self.num_levels):
                stst = self.start_stop(lvl, mult_type)
                mont_prepack.append([self.mont_pack(*stst)])
                ntt_prepack.append([self.ntt_pack(*stst)])
                intt_prepack.append([self.intt_pack(*stst)])
                Rs_prepack.append([self.param_pack(self.Rs, *stst)])
                Rs_scale_prepack.append([self.param_pack(self.Rs_scale, *stst)])
                _2q_prepack.append([self.param_pack(self._2q, *stst)])
                q_prepack.append([self.param_pack(self.qlists, *stst)])
            self.mont_prepack.append(mont_prepack)
            self.ntt_prepack.append(ntt_prepack)
            self.intt_prepack.append(intt_prepack)
            self.Rs_prepack.append(Rs_prepack)
            self.Rs_scale_prepack.append(Rs_scale_prepack)
            self._2q_prepack.append(_2q_prepack)
            self.q_prepack.append(q_prepack)

    # -------------------------------------------------------------------------------------------------
    # Helper functions to do the Montgomery and NTT operations.
    # -------------------------------------------------------------------------------------------------

    def mont_enter(self, a, lvl=0, mult_type=-1, part=0):
        torch.ops.tiberate_ntt_ops.mont_enter(
            a,
            self.Rs_prepack[mult_type][lvl][part],
            *self.mont_prepack[mult_type][lvl][part],
        )

    def mont_enter_scale(self, a, lvl=0, mult_type=-1, part=0):
        torch.ops.tiberate_ntt_ops.mont_enter(
            a,
            self.Rs_scale_prepack[mult_type][lvl][part],
            *self.mont_prepack[mult_type][lvl][part],
        )

    def mont_enter_scalar(self, a, b, lvl=0, mult_type=-1, part=0):
        torch.ops.tiberate_ntt_ops.mont_enter(
            a, b, *self.mont_prepack[mult_type][lvl][part]
        )

    def mont_mult(self, a, b, lvl=0, mult_type=-1, part=0):
        return torch.ops.tiberate_ntt_ops.mont_mult(
            a, b, *self.mont_prepack[mult_type][lvl][part]
        )

    def ntt(self, a, lvl=0, mult_type=-1, part=0):
        torch.ops.tiberate_ntt_ops.ntt(
            a, *self.ntt_prepack[mult_type][lvl][part]
        )

    def enter_ntt(self, a, lvl=0, mult_type=-1, part=0):
        torch.ops.tiberate_ntt_ops.enter_ntt(
            a,
            self.Rs_prepack[mult_type][lvl][part],
            *self.ntt_prepack[mult_type][lvl][part],
        )

    def intt(self, a, lvl=0, mult_type=-1, part=0):
        torch.ops.tiberate_ntt_ops.intt(
            a, *self.intt_prepack[mult_type][lvl][part]
        )

    def mont_reduce(self, a, lvl=0, mult_type=-1, part=0):
        torch.ops.tiberate_ntt_ops.mont_reduce(
            a, *self.mont_prepack[mult_type][lvl][part]
        )

    def intt_exit(self, a, lvl=0, mult_type=-1, part=0):
        torch.ops.tiberate_ntt_ops.intt_exit(
            a, *self.intt_prepack[mult_type][lvl][part]
        )

    def intt_exit_reduce(self, a, lvl=0, mult_type=-1, part=0):
        torch.ops.tiberate_ntt_ops.intt_exit_reduce(
            a, *self.intt_prepack[mult_type][lvl][part]
        )

    def intt_exit_reduce_signed(self, a, lvl=0, mult_type=-1, part=0):
        torch.ops.tiberate_ntt_ops.intt_exit_reduce_signed(
            a, *self.intt_prepack[mult_type][lvl][part]
        )

    def reduce_2q(self, a, lvl=0, mult_type=-1, part=0):
        torch.ops.tiberate_ntt_ops.reduce_2q(
            a, self._2q_prepack[mult_type][lvl][part]
        )

    def make_signed(self, a, lvl=0, mult_type=-1, part=0):
        torch.ops.tiberate_ntt_ops.make_signed(
            a, self._2q_prepack[mult_type][lvl][part]
        )

    def make_unsigned(self, a, lvl=0, mult_type=-1, part=0):
        torch.ops.tiberate_ntt_ops.make_unsigned(
            a, self._2q_prepack[mult_type][lvl][part]
        )

    def mont_add(self, a, b, lvl=0, mult_type=-1, part=0):
        return torch.ops.tiberate_ntt_ops.mont_add(
            a, b, self._2q_prepack[mult_type][lvl][part]
        )

    def mont_sub(self, a, b, lvl=0, mult_type=-1, part=0):
        return torch.ops.tiberate_ntt_ops.mont_sub(
            a, b, self._2q_prepack[mult_type][lvl][part]
        )

    def tile_unsigned(self, a, lvl=0, mult_type=-1, part=0):
        return torch.ops.tiberate_ntt_ops.tile_unsigned(
            a, self._2q_prepack[mult_type][lvl][part]
        )

    # ===========================================
    # fused ops
    # ===========================================

    def mont_add_reduce_2q(self, a, b, lvl=0, mult_type=-1, part=0):
        return torch.ops.tiberate_fused_ops.mont_add_reduce_2q(
            a, b, self._2q_prepack[mult_type][lvl][part]
        )

    def mont_pc_add_fused(self, ct_data, pt_data, lvl=0, mult_type=-1, part=0):
        return torch.ops.tiberate_fused_ops.pc_add_fused(
            ct_data,
            pt_data,
            self._2q_prepack[mult_type][lvl][part],
            self.Rs_prepack[mult_type][lvl][part],
            *self.mont_prepack[mult_type][lvl][part],
        )

    def __repr__(self):
        pass

    # def __str__(self):
    #     what_is_this = f"{self.__class__}"
    #     what_is_this += f"""
    #     Using CKKS Context:
    #     {textwrap.indent(str(self.ckksCtx), ' '*8)} # todo)) continue refactor
    #     Using devices = {self.devices}
    #     Available levels = {self.num_levels}
    #     Ordinary primes = {self.num_ordinary_primes}
    #     Special primes = {self.num_special_primes}
    #     """
    #     return what_is_this
