from abc import ABC, abstractmethod
from typing import List, Union


class RandNumGen(ABC):
    @abstractmethod
    def randbytes(self, shares: List[int] = None, repeats: int = 0, reshape: bool = False):
        """
        Generate random bytes.

        Args:
            shares (List[int]): Number of channels per device.
            repeats (int): Number of repeating channels.
            reshape (bool): Whether to reshape the output.

        Returns:
            List[torch.Tensor]: List of random byte tensors.
        """
        pass

    @abstractmethod
    def randint(
        self,
        amax: Union[int, List[List[int]]],
        shift: int = 0,
        repeats: int = 0,
    ):
        """
        Generate random integers.

        Args:
            amax (int or List[List[int]]): Maximum values per channel.
            shift (int): Optional shift to apply to results.
            repeats (int): Number of repeating channels.

        Returns:
            List[torch.Tensor]: List of random integer tensors.
        """
        pass

    @abstractmethod
    def discrete_gaussian(self, non_repeats: Union[int, List[int]] = 0, repeats: int = 1):
        """
        Generate discrete Gaussian samples.

        Args:
            non_repeats (int or List[int]): Channels without repetition.
            repeats (int): Number of repeating channels.

        Returns:
            List[torch.Tensor]: List of sampled tensors.
        """
        pass

    @abstractmethod
    def randround(self, coef):
        """
        Randomly round floating-point coefficients.

        Args:
            coef (torch.Tensor): Tensor to be randomly rounded.

        Returns:
            torch.Tensor: Rounded values.
        """
        pass
