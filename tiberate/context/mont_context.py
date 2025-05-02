from dataclasses import dataclass

from tiberate.config import CkksConfig


@dataclass
class MontgomeryContext:
    """
    Montgomery context for CKKS context.
    """

    R: int
    R_square: list[int]
    half_buffer_bit_length: int
    lower_bits_mask: int
    full_bits_mask: int
    q_lower_bits: list[int]
    q_higher_bits: list[int]
    q_double: list[int]
    R_inv: list[int]
    k: list[int]
    k_lower_bits: list[int]
    k_higher_bits: list[int]
    q: list[int]

    @classmethod
    def from_ckks_config(cls, ckks_config: CkksConfig):
        R = 2**ckks_config.buffer_bit_length
        R_square = [R**2 % qi for qi in ckks_config.q]
        half_buffer_bit_length = ckks_config.buffer_bit_length // 2
        lower_bits_mask = (1 << half_buffer_bit_length) - 1
        full_bits_mask = (1 << ckks_config.buffer_bit_length) - 1

        q_lower_bits = [qi & lower_bits_mask for qi in ckks_config.q]
        q_higher_bits = [qi >> half_buffer_bit_length for qi in ckks_config.q]
        q_double = [qi << 1 for qi in ckks_config.q]

        R_inv = [pow(R, -1, qi) for qi in ckks_config.q]
        k = [(R * R_invi - 1) // qi for R_invi, qi in zip(R_inv, ckks_config.q)]
        k_lower_bits = [ki & lower_bits_mask for ki in k]
        k_higher_bits = [ki >> half_buffer_bit_length for ki in k]

        return cls(
            R=R,
            R_square=R_square,
            half_buffer_bit_length=half_buffer_bit_length,
            lower_bits_mask=lower_bits_mask,
            full_bits_mask=full_bits_mask,
            q_lower_bits=q_lower_bits,
            q_higher_bits=q_higher_bits,
            q_double=q_double,
            R_inv=R_inv,
            k=k,
            k_lower_bits=k_lower_bits,
            k_higher_bits=k_higher_bits,
            q=ckks_config.q,
        )
