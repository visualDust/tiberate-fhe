# -*- coding: utf-8 -*-
#
# Author: GavinGong aka VisualDust
# Github: github.com/visualDust

import torch


class HEModule:
    debug = False

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class HELinear(HEModule):
    debug = False

    @classmethod
    def fromTorch(cls, *args, **kwargs):
        raise NotImplementedError()


class HELayerNorm(HEModule):
    debug = False

    @classmethod
    def fromTorch(cls, *args, **kwargs):
        raise NotImplementedError()
