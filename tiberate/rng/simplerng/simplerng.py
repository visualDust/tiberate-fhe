import os

import torch

from tiberate.rng.interface import RandNumGen  # Keep as per your structure


class SimpleRNG(RandNumGen):
    def __init__(
        self,
        num_coefs=2**15,
        num_channels=[8],
        num_repeating_channels=2,
        sigma=3.2,
        devices=None,
        seed=None,
        nonce=None,
    ):
        self.num_coefs = num_coefs
        self.num_channels = num_channels
        self.num_repeating_channels = num_repeating_channels
        self.sigma = sigma

        # Set devices
        if devices is None:
            self.devices = [
                f"cuda:{i}" for i in range(torch.cuda.device_count())
            ] or ["cpu"]
        else:
            self.devices = devices

        self.num_devices = len(self.devices)

        if len(self.num_channels) == 1:
            self.shares = [self.num_channels[0]] * self.num_devices
        elif len(self.num_channels) == self.num_devices:
            self.shares = self.num_channels
        else:
            raise Exception("Mismatch between num_channels and devices.")

        self.total_num_channels = sum(self.shares)
        self.L = self.num_coefs // 4

        # Initialize random number generators per device
        self.generators = []
        for dev in self.devices:
            gen = torch.Generator(device=dev)
            seed_val = (
                seed
                if seed is not None
                else int.from_bytes(os.urandom(8), "big")
            )
            gen.manual_seed(seed_val)
            self.generators.append(gen)

    def randbytes(
        self, shares: list[int] = None, repeats: int = 0, reshape: bool = False
    ):
        if shares is None:
            shares = self.shares

        rand_out = []
        for i, (dev, gen) in enumerate(zip(self.devices, self.generators)):
            num_ch = shares[i] + repeats
            shape = (
                (num_ch, self.L * 4, 16)
                if reshape
                else (num_ch, self.L * 4 * 16)
            )
            rb = torch.randint(
                0, 256, shape, dtype=torch.uint8, generator=gen, device=dev
            )
            rand_out.append(rb)
        return rand_out

    def randint(
        self,
        amax: int | list[list[int]],
        shift: int = 0,
        repeats: int = 0,
    ):
        if isinstance(amax, int):
            amax = [
                [amax] * (self.shares[i] + repeats)
                for i in range(self.num_devices)
            ]

        result = []
        for i, (dev, gen) in enumerate(zip(self.devices, self.generators)):
            per_device = []
            for q in amax[i]:
                r = torch.randint(
                    low=0,
                    high=q,
                    size=(self.num_coefs,),
                    generator=gen,
                    device=dev,
                )
                per_device.append(r + shift)
            result.append(torch.stack(per_device))
        return [result[0][:1]]

    def discrete_gaussian(
        self, non_repeats: int | list[int] = 0, repeats: int = 1
    ):
        if isinstance(non_repeats, int):
            shares = [non_repeats] * self.num_devices
        else:
            shares = non_repeats

        result = []
        for i, (dev, gen) in enumerate(zip(self.devices, self.generators)):
            num_ch = shares[i] + repeats
            samples = torch.normal(
                mean=0.0,
                std=self.sigma,
                size=(num_ch, self.num_coefs),
                generator=gen,
                device=dev,
            )
            result.append(torch.round(samples).to(torch.int64))
        return result

    def randround(self, coef):
        # coef must be on the first device
        rand = torch.rand_like(coef)
        frac = coef - torch.floor(coef)
        rounded = torch.floor(coef) + (rand < frac).float()
        return rounded.to(int)
