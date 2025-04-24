#
# Author: GavinGong aka VisualDust
# Github: github.com/visualDust

import math
from typing import Any

import torch
from loguru import logger
from vdtoys.cache import CachedDict
from vdtoys.framing import get_caller_info_traceback
from vdtoys.registry import Registry

from tiberate import CkksEngine
from tiberate.typing import *  # noqa: F403

from ..packing.feature_wise_compact import (
    FeatureWise_PackedCT,
    FeatureWise_PTPacking,
)
from ..packing.interface import PackedCT
from .interface import HELinear


@Registry(str(HELinear)).register()
class HELinear_FeatureWiseCTInput_ColMajorPTSquareWeight_FeatureWiseCTOutput(
    HELinear
):
    def __init__(
        self,
        weight: list[Any],
        bias: list[Any],
        engine: CkksEngine,
    ):
        if get_caller_info_traceback().func_name != "fromWeight":
            logger.warning(
                f"Please use class methods .fromTorch or .fromWeight to create an instance of {self.__class__.__name__}, DO NOT USE THIS init METHOD DIRECTLY, unless you know what you are doing."
            )

        super().__init__()

        # encode weight and bias

        plaintext_weight = []
        for i in range(len(weight)):
            plaintext_weight.append([])
            for j in range(len(weight[i])):
                plaintext_weight[-1].append(Plaintext(weight[i][j]))

        self.weight = plaintext_weight

        if bias:
            plaintext_bias = []
            for i in range(
                len(bias)
            ):  # only one row in bias actually, but for general case, still use loop
                plaintext_bias.append([])
                for j in range(len(bias[i])):
                    plaintext_bias[-1].append(Plaintext(bias[i][j]))
        self.bias = plaintext_bias if bias else None

        self.engine = engine

        def encode_mask_for_every_logical_num_slots_th(logical_num_slots):
            # mask when logical_num_slots is factor of physical num_slots
            mask_for_every_logical_num_slots_th = [0] * self.engine.num_slots
            # i*logical_num_slots-th element set to 1
            mask_for_every_logical_num_slots_th[0] = 1
            for i in range(self.engine.num_slots // logical_num_slots):
                mask_for_every_logical_num_slots_th[i * logical_num_slots] = 1
            return Plaintext(mask_for_every_logical_num_slots_th)

        self.mask_out_others_except_every_logical_num_slots_th_element = (
            CachedDict(encode_mask_for_every_logical_num_slots_th)
        )

        self.mask_zeros = Plaintext([0] * self.engine.num_slots)

    @classmethod
    def fromTorch(cls, linear: torch.nn.Linear, engine: CkksEngine):
        return cls.fromWeight(linear.weight, linear.bias, engine)

    @classmethod
    def fromWeight(
        cls,
        weight: torch.Tensor | torch.nn.Parameter,
        bias: torch.Tensor | torch.nn.Parameter,
        engine: CkksEngine,
    ) -> (
        "HELinear_FeatureWiseCTInput_ColMajorPTSquareWeight_FeatureWiseCTOutput"
    ):
        if cls.debug:
            logger.debug(f"packing weight and bias for {cls.__name__}")
        # check weight shape
        assert (
            len(weight.shape) == 2
        ), f"weight shape must be 2D, got {weight.shape}"
        assert (
            weight.shape[0] == weight.shape[1]
        ), "this is a linear layer only for squre weight, weight shape must be square"
        if bias is not None:
            assert (
                len(bias.shape) == 1
            ), f"bias shape must be 1D, got {bias.shape}"
            assert (
                weight.shape[0] == bias.shape[0]
            ), f"weight shape {weight.shape} must match bias shape {bias.shape}"
            bias = bias.detach()
            # expand bias with an extra dim
            bias = bias.unsqueeze(0)
        weight = weight.detach()  # weight here is already transposed
        logical_num_slots = FeatureWise_PTPacking.find_logical_num_slots(
            engine.num_slots, last_dim=weight.shape[-1]
        )
        if (
            logical_num_slots % engine.num_slots == 0
        ):  # logical_num_slots is multiple of physical num_slots
            if cls.debug:
                logger.debug(
                    f"logical_num_slots: {logical_num_slots}, the weight will be padded to align with the logical_num_slots."
                )
            # no repeat, just pad
            weight_padded = FeatureWise_PTPacking.pad_tensor_to_align_logical_num_slots_on_last_dim(
                x=weight, logical_num_slots=logical_num_slots
            )
            weight_fit_with_num_slots = FeatureWise_PTPacking.fit_padded_tensor_into_num_slots_on_last_dim(
                x=weight_padded, num_slots=engine.num_slots
            )
            # fold
            fold_factor = logical_num_slots // engine.num_slots
            weight_fit_with_num_slots = [
                weight_fit_with_num_slots[i : i + fold_factor]
                for i in range(0, len(weight_fit_with_num_slots), fold_factor)
            ]

            weight = weight_fit_with_num_slots

            # do again on bias
            if bias is not None:
                bias_padded = FeatureWise_PTPacking.pad_tensor_to_align_logical_num_slots_on_last_dim(
                    x=bias, logical_num_slots=logical_num_slots
                )
                bias_fit_with_num_slots = FeatureWise_PTPacking.fit_padded_tensor_into_num_slots_on_last_dim(
                    x=bias_padded, num_slots=engine.num_slots
                )
                bias_fit_with_num_slots = [
                    bias_fit_with_num_slots[i : i + fold_factor]
                    for i in range(0, len(bias_fit_with_num_slots), fold_factor)
                ]
                bias = bias_fit_with_num_slots

        elif (
            engine.num_slots % logical_num_slots == 0
        ):  # logical_num_slots is factor of physical num_slots
            repeat_factor = engine.num_slots // logical_num_slots
            if cls.debug:
                logger.debug(
                    f"logical_num_slots: {logical_num_slots}, the weight will be repeated for {repeat_factor} times."
                )
            weight_repeated = weight.repeat_interleave(repeat_factor, dim=0)
            weight_padded = FeatureWise_PTPacking.pad_tensor_to_align_logical_num_slots_on_last_dim(
                x=weight_repeated, logical_num_slots=logical_num_slots
            )
            weight_fit_with_num_slots = FeatureWise_PTPacking.fit_padded_tensor_into_num_slots_on_last_dim(
                x=weight_padded, num_slots=engine.num_slots
            )
            # fold (actually no fold, just add dim to align dims)
            weight_fit_with_num_slots = [
                [weight_fit_with_num_slots[i]]
                for i in range(len(weight_fit_with_num_slots))
            ]
            weight = weight_fit_with_num_slots

            # do again on bias
            if bias is not None:
                bias_repeated = bias.repeat_interleave(repeat_factor, dim=0)
                bias_padded = FeatureWise_PTPacking.pad_tensor_to_align_logical_num_slots_on_last_dim(
                    x=bias_repeated, logical_num_slots=logical_num_slots
                )
                bias_fit_with_num_slots = FeatureWise_PTPacking.fit_padded_tensor_into_num_slots_on_last_dim(
                    x=bias_padded, num_slots=engine.num_slots
                )
                bias_fit_with_num_slots = [
                    [bias_fit_with_num_slots[i]]
                    for i in range(len(bias_fit_with_num_slots))
                ]
                bias = bias_fit_with_num_slots

        else:
            raise ValueError(  # well, this should not happen
                f"logical_num_slots {logical_num_slots} is neither a multiple nor a factor of physical num_slots {engine.num_slots}"
            )

        # create instance
        instance = cls(weight=weight, bias=bias, engine=engine)

        return instance

    # todo only accept specific kind of packing for input
    def forward(
        self, ct_in: PackedCT, memory_save: bool = False
    ) -> FeatureWise_PackedCT:
        if memory_save:
            return self.forward_memory_save(ct_in)
        else:
            return self.forward_non_memory_save(ct_in)

    def forward_non_memory_save(self, ct_in: PackedCT) -> FeatureWise_PackedCT:
        input_level = ct_in.cts[0][0][0].level
        if self.__class__.debug:
            logger.debug(
                f"Doing forward(non memory save) on ct_in with original shape {ct_in.metadata.original_shape}, level={input_level}"
            )

        logical_num_slots = ct_in.metadata.logical_num_slots
        mask_logical_num_slots = (
            self.mask_out_others_except_every_logical_num_slots_th_element[
                logical_num_slots
            ]
        )

        result = (
            []
        )  # since the weight is square, the num of rows of output should be the same as the num of rows of input

        for batch in range(len(ct_in.cts)):  # for each batch
            if self.__class__.debug:
                logger.debug(f"=> Processing batch: {batch}/{len(ct_in.cts)-1}")
            result.append([])
            for row in range(len(ct_in.cts[batch])):  # for each row in input
                # c should be a list with 1 or more ct that represent logical ct
                intermediate_cts = []
                if self.__class__.debug:
                    logger.debug(
                        f"\t=> Processing row: {row}/{len(ct_in.cts[batch])-1}"
                    )
                    logger.debug(
                        f"\t=> ct_in.cts[b][c] has {len(ct_in.cts[batch][row])} elements, each of them will be multiplied with {len(self.weight)} row in weight, and produce {len(self.weight)} intermediate results"
                    )

                for weight_row in range(
                    len(self.weight)
                ):  # for each row(logically col) in weight
                    assert len(self.weight[weight_row]) == len(
                        ct_in.cts[batch][row]
                    ), f"input row {row} must match weight col {weight_row} in length"  # debug check
                    intermediate_cts.append([])
                    for ict_idx in range(len(self.weight[weight_row])):
                        intermediate_cts[-1].append(
                            self.engine.pc_mult(
                                pt=self.weight[weight_row][ict_idx],
                                ct=ct_in.cts[batch][row][ict_idx],
                            )
                        )

                    # reduce sum for this row on intermediate_cts
                    if (
                        ct_in.metadata.logical_num_slots % self.engine.num_slots
                        == 0
                    ):  # logical_num_slots is multiple of physical num_slots, this include the case that logical_num_slots == physical num_slots
                        if self.__class__.debug:
                            logger.debug(
                                f"\t=> logical_num_slots is multiple of physical num_slots, doing sum on each logical row of intermediate_cts, there are {len(intermediate_cts[0])} cts in each of them"
                            )
                        # sum each cipher in intermediate_cts[row_idx]

                    for i in range(1, len(intermediate_cts[-1])):
                        intermediate_cts[-1][0] = [
                            self.engine.cc_add(
                                intermediate_cts[-1][0],
                                intermediate_cts[-1][i],
                                self.engine.evk,
                            )
                        ]

                result_for_this_row = None

                # elif self.engine.num_slots % ct_in.metadata.logical_num_slots == 0:
                # following procedure share the same logic with the above one
                # each intermediate_ct should only contain 1 ct. rotate for log2 logical_num_slots times.
                rotate_deltas = int(
                    math.log2(
                        min(
                            ct_in.metadata.logical_num_slots,
                            self.engine.num_slots,
                        )
                    )
                )
                mask = mask_logical_num_slots
                if self.__class__.debug:
                    logger.debug(
                        f"\t\t=> doing rotate and sum, rotate_deltas = {range(rotate_deltas)}"
                    )
                for ict_idx in range(len(intermediate_cts)):
                    tmp_ct = intermediate_cts[ict_idx][0]
                    for roti in range(rotate_deltas):
                        rot_ct = self.engine.rotate_single(
                            tmp_ct, self.engine.rotk[-(2**roti)]
                        )
                        tmp_ct = self.engine.cc_add(rot_ct, tmp_ct)

                    intermediate_cts[ict_idx][0] = tmp_ct
                    # now every `logical_num_slots`-th element in intermediate_cts[:][0] are the result of the corresponding element in output row c, all of them are needed.
                    # # for each intermediate result
                    # for ict_idx in range(
                    #     len(intermediate_cts)
                    # ):  # len(intermediate_cts) equals to logical_num_slots in this case
                    # mask it, each `logical_num_slots`-th element is effective, others should be zero
                    intermediate_cts[ict_idx][0] = self.engine.pc_mult(
                        pt=mask,
                        ct=intermediate_cts[ict_idx][0],
                    )
                    # rotate it with delta `i`
                    intermediate_cts[ict_idx][0] = self.engine.rotate_single(
                        ct=intermediate_cts[ict_idx][0],
                        rotk=self.engine.rotk[ict_idx],
                    )
                    if result_for_this_row is None:
                        assert (
                            ict_idx == 0
                        ), f"result_for_this_row should be None at the beginning, but got {ict_idx}"  # debug check
                        result_for_this_row = intermediate_cts[ict_idx]
                    else:
                        result_for_this_row[
                            ict_idx // self.engine.num_slots
                        ] = self.engine.cc_add(
                            result_for_this_row[
                                ict_idx // self.engine.num_slots
                            ],
                            intermediate_cts[ict_idx][0],
                        )
                    if ict_idx != 0:
                        intermediate_cts[ict_idx][0] = None  # release memory

                # # apply bias
                if self.bias is not None:
                    if self.__class__.debug:
                        logger.debug(
                            f"\t=> adding bias to the result_for_this_row (level={result_for_this_row[0].level})"
                        )
                    for ict_idx in range(len(result_for_this_row)):
                        result_for_this_row[ict_idx] = self.engine.pc_add(
                            pt=self.bias[0][ict_idx],
                            ct=result_for_this_row[ict_idx],
                        )

                # append the result for this row to the result
                result[-1].append(result_for_this_row)

        return FeatureWise_PackedCT(cts=result, metadata=ct_in.metadata)

    def forward_memory_save(self, ct_in: PackedCT) -> FeatureWise_PackedCT:
        input_level = ct_in.cts[0][0][0].level
        if self.__class__.debug:
            logger.debug(
                f"Doing forward(memory save) on ct_in with original shape {ct_in.metadata.original_shape}, level={input_level}"
            )

        logical_num_slots = ct_in.metadata.logical_num_slots
        mask_logical_num_slots = (
            self.mask_out_others_except_every_logical_num_slots_th_element[
                logical_num_slots
            ]
        )
        mask_zeros = self.mask_zeros
        result = (
            []
        )  # since the weight is square, the num of rows of output should be the same as the num of rows of input

        for batch in range(len(ct_in.cts)):  # for each batch
            if self.__class__.debug:
                logger.debug(f"=> Processing batch: {batch}/{len(ct_in.cts)-1}")
            result.append([])
            for row in range(len(ct_in.cts[batch])):  # for each row in input
                # c should be a list with 1 or more ct that represent logical ct
                intermediate_cts = None
                if self.__class__.debug:
                    logger.debug(
                        f"\t=> Processing row: {row}/{len(ct_in.cts[batch])-1}"
                    )
                    logger.debug(
                        f"\t=> ct_in.cts[b][c] has {len(ct_in.cts[batch][row])} elements, each of them will be multiplied with {len(self.weight)} row in weight, and produce {len(self.weight)} intermediate results"
                    )
                for weight_row in range(
                    len(self.weight)
                ):  # for each row(logically col) in weight
                    assert len(self.weight[weight_row]) == len(
                        ct_in.cts[batch][row]
                    ), f"input row {row} must match weight col {weight_row} in length"  # debug check
                    intermediate_cts = []
                    for ict_idx in range(len(self.weight[weight_row])):
                        intermediate_cts.append(
                            self.engine.pc_mult(
                                pt=self.weight[weight_row][ict_idx],
                                ct=ct_in.cts[batch][row][ict_idx],
                            )
                        )

                    # reduce sum for this row on intermediate_cts
                    if (
                        ct_in.metadata.logical_num_slots % self.engine.num_slots
                        == 0
                    ):  # logical_num_slots is multiple of physical num_slots, this include the case that logical_num_slots == physical num_slots
                        if self.__class__.debug:
                            logger.debug(
                                f"\t=> logical_num_slots is multiple of physical num_slots, doing sum on each logical row of intermediate_cts, there are {len(intermediate_cts)} cts in each of them"
                            )
                        # sum each cipher in intermediate_cts
                    for i in range(1, len(intermediate_cts)):
                        intermediate_cts[0] = [
                            self.engine.cc_add(
                                intermediate_cts[0],
                                intermediate_cts[i],
                                self.engine.evk,
                            )
                        ]
                    # do I have to manually release memory here?
                    tmp_ct = intermediate_cts[0]
                    rotate_deltas = int(
                        math.log2(
                            min(
                                ct_in.metadata.logical_num_slots,
                                self.engine.num_slots,
                            )
                        )
                    )
                    mask = mask_logical_num_slots
                    for roti in range(rotate_deltas):
                        rot_ct = self.engine.rotate_single(
                            tmp_ct, self.engine.rotk[-(2**roti)]
                        )
                        tmp_ct = self.engine.cc_add(rot_ct, tmp_ct)
                    intermediate_cts[0] = tmp_ct

                    intermediate_cts[0] = self.engine.pc_mult(
                        pt=mask,
                        ct=intermediate_cts[0],
                    )

                    if len(result[-1]) <= row:
                        result[-1].append(intermediate_cts)
                        for i in range(1, len(result[-1][-1])):
                            result[-1][-1][i] = self.engine.pc_mult(
                                pt=mask_zeros, ct=result[-1][-1][i]
                            )
                    else:
                        result[-1][-1][weight_row // self.engine.num_slots] = (
                            self.engine.rotate_single(
                                result[-1][-1][
                                    weight_row // self.engine.num_slots
                                ],
                                rotk=self.engine.rotk[-1],
                            )
                        )
                        result[-1][-1][weight_row // self.engine.num_slots] = (
                            self.engine.cc_add(
                                result[-1][-1][
                                    weight_row // self.engine.num_slots
                                ],
                                intermediate_cts[0],
                            )
                        )

                result[-1][-1] = [
                    self.engine.rotate_single(
                        result[-1][-1][i],
                        rotk=self.engine.rotk[len(self.weight) - 1],
                    )
                    for i in range(len(result[-1][-1]))
                ]

                # # apply bias
                if self.bias is not None:
                    if self.__class__.debug:
                        logger.debug(
                            f"\t=> adding bias to the result[-1][-1] (level={result[-1][-1][0].level})"
                        )
                    for ict_idx in range(len(result[-1][-1])):
                        result[-1][-1][ict_idx] = self.engine.pc_add(
                            pt=self.bias[0][ict_idx],
                            ct=result[-1][-1][ict_idx],
                        )

        return FeatureWise_PackedCT(cts=result, metadata=ct_in.metadata)
