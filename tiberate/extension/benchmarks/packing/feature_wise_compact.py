#
# Author: GavinGong aka VisualDust
# Github: github.com/visualDust

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from loguru import logger
from vdtoys.registry import Registry

from tiberate import CkksEngine
from tiberate.typing import *
from tiberate.utils.massive import next_power_of_2

from ..packing.interface import PackedCT, PTPacking
from .interface import CTEncoding, PackedCT, PackingMetadata

if TYPE_CHECKING:
    import numpy as np


class FeatureWise_PTPacking(PTPacking):
    @classmethod
    def pack(
        cls,
        *,
        x: torch.Tensor,
        num_slots: int,
        logical_num_slots: int,
    ) -> PackedCT:
        x = cls.pad_tensor_to_align_logical_num_slots_on_last_dim(
            x, logical_num_slots
        )
        x = cls.fit_padded_tensor_into_num_slots_on_last_dim(x, num_slots)
        return x

    @classmethod
    def unpack(
        cls,
        *,
        x: torch.Tensor,
        logical_num_slots: int,
        original_shape: tuple,
    ) -> torch.Tensor:
        x = cls.undo_fit_padded_tensor_into_num_slots_on_last_dim(
            x, logical_num_slots
        )
        x = cls.undo_pad_tensor_to_align_logical_num_slots_on_last_dim(
            x, original_shape
        )
        return x

    @classmethod
    def find_logical_num_slots(cls, num_slots: int, last_dim: int):
        if last_dim > num_slots:
            # Find the smallest multiple of num_slots greater than length
            logical_num_slots = num_slots * math.ceil(last_dim / num_slots)
        else:
            # pad to next power of 2
            logical_num_slots = next_power_of_2(last_dim)
        return logical_num_slots

    @classmethod
    def pad_tensor_to_align_logical_num_slots_on_last_dim(
        cls, x: torch.Tensor, logical_num_slots: int
    ):
        """
        Do padding on the last dimension of the input tensor to align with the logical_num_slots.
        """
        length = x.shape[-1]
        pad_size = logical_num_slots - length

        if pad_size > 0:
            if isinstance(x, np.ndarray):
                # Create a pad width for each dimension
                pad_width = [(0, 0)] * x.ndim
                # Pad only the last dimension
                pad_width[-1] = (0, pad_size)
                # Apply padding
                x = np.pad(
                    x,
                    pad_width=pad_width,
                    mode="constant",
                    constant_values=0,
                )
            elif isinstance(x, torch.Tensor):
                # Create a tensor of zeros to concatenate
                pad_shape = list(x.shape)
                pad_shape[-1] = pad_size
                pad_tensor = torch.zeros(
                    *pad_shape, dtype=x.dtype, device=x.device
                )
                # Concatenate along the last dimension
                x = torch.cat((x, pad_tensor), dim=-1)
            else:
                raise TypeError(
                    "Input tensor must be a NumPy array or a PyTorch tensor."
                )
        return x

    @classmethod
    def undo_pad_tensor_to_align_logical_num_slots_on_last_dim(
        cls, x: torch.Tensor, original_shape: Union[List[int], tuple[int]]
    ):
        """
        This is the inverse operation of @pad_tensor_to_align_logical_num_slots_on_last_dim,
        which transforms the input tensor back to its original shape.

        Please call @undo_fit_padded_tensor_into_num_slots_on_last_dim before calling this function.

        Parameters:
        - tensor (torch.Tensor): The input tensor.
        - original_shape (List[int]): The original shape of the tensor before padding.

        Returns:
        - torch.Tensor: The restored tensor in shape of original_shape.
        """
        # the last dim is bigger than the last dim of original shape, the extra part should be removed
        # print(x.shape)
        x = x[..., : original_shape[-1]]
        # reshape the tensor to the original shape
        x = x.reshape(original_shape)  # todo fix error for size like 1,512
        return x

    @classmethod
    def fit_padded_tensor_into_num_slots_on_last_dim(
        cls, x: torch.Tensor, num_slots: int
    ):
        """
        This function fits the padded tensor into num_slots. Please make sure the input tensor is the output of @pad_tensor_to_align_logical_num_slots_on_last_dim.

        Parameters:
        - input_tensor (torch.Tensor): The input tensor.
        - num_slots (int): The target size for the last dimension should be num_slots.

        Returns:
        - torch.Tensor: The transformed tensor with the last dimension of size num_slots.
        """
        logical_num_slots = last_dim = x.shape[
            -1
        ]  # note that last dim here should be equal to logical_num_slots, this ensured by @pad_tensor_to_align_logical_num_slots_on_last_dim
        rest_shape = x.shape[:-1]  # All dimensions except the last one

        if logical_num_slots % num_slots == 0:
            # Case 1: logical_num_slots is a multiple of num_slots, this include the case that logical_num_slots == num_slots
            factor = logical_num_slots // num_slots
            # Flatten the tensor except the last dimension
            output = x.reshape(-1, logical_num_slots)
            # Split each row into 'factor' rows
            output = output.reshape(-1, factor, num_slots)
            output = output.reshape(-1, num_slots)
            # Restore the original batch dimensions with increased rows
            output_shape = rest_shape[:-1] + (-1, num_slots)
            output = output.reshape(output_shape)
            return output
        elif num_slots % logical_num_slots == 0:
            # Case 2: logical_num_slots is a factor of num_slots
            factor = num_slots // logical_num_slots
            # Flatten the tensor except the last dimension
            output = x.reshape(-1, logical_num_slots)
            num_rows = output.shape[0]
            if num_rows % factor != 0:
                if num_rows == 1 and output.shape[1] < num_slots:
                    # pad the last dimension to num_slots
                    output = F.pad(
                        output,
                        (0, num_slots - output.shape[1]),
                        mode="constant",
                        value=0,
                    )
            # Combine 'factor' rows into one
            output = output.reshape(
                max(num_rows // factor, 1), factor * logical_num_slots
            )
            # Reshape to get the last dimension size num_slots
            output = output.reshape(-1, num_slots)
            # Restore the original batch dimensions
            output_shape = rest_shape[:-1] + (-1, num_slots)
            output = output.reshape(output_shape)
            return output

        else:
            raise ValueError(
                "The last dimension must be a factor or multiple of num_slots."
            )

    @classmethod
    def undo_fit_padded_tensor_into_num_slots_on_last_dim(
        cls, x: torch.Tensor, logical_num_slots: int
    ):
        """
        This is the inverse operation of @fit_padded_tensor_into_num_slots_on_last_dim,
        which transforms the input tensor back to its original shape.

        Parameters:
        - input_tensor (torch.Tensor): The input tensor.
        - logical_num_slots (int): The logical number of slots used for padding.


        Returns:
        - torch.Tensor: The restored tensor in shape of original shape after padding. It's not the original tensor, but the restored tensor after padding. Please also call @undo_pad_tensor_to_align_logical_num_slots_on_last_dim to get the original tensor.
        """

        # the last dim of input tensor should be either a factor or multiple of logical_num_slots
        num_slots = x.shape[
            -1
        ]  # note, last dim here is the last dim of tensor after fit into num_slots, which means it should be equal to num_slots
        if (
            logical_num_slots % num_slots == 0
        ):  # Case 1: logical_num_slots is multiple of num_slots, this include the case that logical_num_slots == num_slots
            factor = logical_num_slots // num_slots
            # Flatten the tensor except the last dimension
            x = x.reshape(-1, num_slots)
            # Split each row into 'factor' rows
            x = x.reshape(-1, factor, num_slots)
            x = x.reshape(-1, logical_num_slots)
            return x
        elif (
            num_slots % logical_num_slots == 0
        ):  # Case 2: logical_num_slots is a factor of num_slots
            factor = num_slots // logical_num_slots
            # Flatten the tensor except the last dimension
            x = x.reshape(-1, num_slots)
            # num_rows = output.shape[0]
            # if num_rows % factor != 0:
            #     raise ValueError(
            #         "Cannot unpack rows evenly: the number of rows is not divisible by factor."
            #     )
            # Combine 'factor' rows into one
            x = x.reshape(-1, factor * logical_num_slots)
            # Reshape to get the last dimension size logical_num_slots
            x = x.reshape(-1, logical_num_slots)
            # print(x.shape)

            return x
        else:
            raise ValueError(
                "The last dimension must be a factor or multiple of logical_num_slots."
            )


@Registry(str(PackedCT)).register()
class FeatureWise_PackedCT(PackedCT):
    pass  # todo add operator overloading(s)


@Registry(str(CTEncoding)).register()
class FeatureWise_CTEncoding(CTEncoding):
    @classmethod
    def encodecrypt(
        cls, *, src: torch.Tensor, engine: CkksEngine, level=0
    ) -> PackedCT:
        origional_shape = src.shape
        if cls.debug:
            logger.debug(
                f"doing row major packing on last two dims with input shape {src.shape}"
            )
        assert (
            len(src.shape) > 2
        ), f"input shape must be at least 3D, got {src.shape}"
        # for example, with input shape B,C,H,W, we pack H,W into N
        # we do packing on each B,C separately
        src = src.view(-1, *src.shape[-2:])
        src = [
            src[i] for i in range(src.shape[0])
        ]  # convert to list of tensors
        num_slots = engine.num_slots
        logical_num_slots = FeatureWise_PTPacking.find_logical_num_slots(
            num_slots, src[0].shape[-1]
        )  # find the logical num_slots for padding

        for i in range(len(src)):  # do on each B,C separately
            src[i] = FeatureWise_PTPacking.pack(
                x=src[i],
                num_slots=num_slots,
                logical_num_slots=logical_num_slots,
            )
        # do encryption
        cts = []
        for bc_dim in range(len(src)):
            ct_per_bcdim = []
            for pt in src[bc_dim]:
                ct_per_bcdim.append(
                    engine.encodecrypt(
                        pt.cpu(), engine.pk, padding=False, level=level
                    )
                )
            cts.append(ct_per_bcdim)

        if (
            logical_num_slots % num_slots == 0
        ):  # check if logical_num_slots is multiple of num_slots, this include the case that logical_num_slots == num_slots
            fold_factor = logical_num_slots // num_slots
            # every fold_factor element in cts will be put into a list
            # do this for each element in cts
            for i in range(len(cts)):
                cts[i] = [
                    cts[i][j : j + fold_factor]
                    for j in range(0, len(cts[i]), fold_factor)
                ]
        elif (
            num_slots % logical_num_slots == 0
        ):  # check if logical_num_slots is a factor of num_slots
            for i in range(len(cts)):
                cts[i] = [[cts[i][j]] for j in range(len(cts[i]))]

        return FeatureWise_PackedCT(
            cts=cts,
            metadata=PackingMetadata(
                original_shape=origional_shape,
                logical_num_slots=logical_num_slots,
                packed_by=FeatureWise_PTPacking,
                encoded_by=cls,
            ),
        )

    @classmethod
    def decryptcode(
        cls,
        *,
        packed_ct: FeatureWise_PackedCT,
        sk: SecretKey,
        engine: CkksEngine,
    ) -> torch.Tensor:
        """
        This is the inverse operation of encodecrypt.

        Args:
            packed_ct: FeatureWise_Compact_PackedCT
            engine: CkksEngine
            sk: SecretKey

        Returns:
            torch.Tensor
        """
        assert isinstance(
            packed_ct, FeatureWise_PackedCT
        ), f"invalid input type {type(packed_ct)}, expected {FeatureWise_PackedCT}"
        assert (
            packed_ct.metadata.packed_by == FeatureWise_PTPacking
            and packed_ct.metadata.encoded_by == cls
        ), "Cannot do decodecrypt on something packed by another packing method"
        cts = packed_ct.cts
        logical_num_slots = packed_ct.metadata.logical_num_slots
        # undo encryption on each B,C separately
        src = []
        for bc_dim in range(len(cts)):
            src_per_bcdim = []
            for ct in cts[bc_dim]:
                for i in range(len(ct)):
                    ct[i] = engine.decryptcode(ct[i], sk)
                    ct[i] = torch.tensor(ct[i])
                src_per_bcdim.append(torch.stack(ct))
            src.append(torch.stack(src_per_bcdim))

        # undo packing on each B,C separately
        for i in range(len(src)):
            src[i] = FeatureWise_PTPacking.unpack(
                x=src[i],
                logical_num_slots=logical_num_slots,
                original_shape=packed_ct.metadata.original_shape[-2:],
            )

        # restore the original shape
        src = torch.stack(src)
        if len(packed_ct.metadata.original_shape) > 2:
            src = src.view(packed_ct.metadata.original_shape)

        return src
