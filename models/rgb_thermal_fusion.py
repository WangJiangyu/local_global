# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import torch.nn as nn

from models.model_utils import SqueezeAndExcitation


class SqueezeAndExciteFusionAdd(nn.Module):
    def __init__(self, channels_in, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExciteFusionAdd, self).__init__()

        self.se_rgb = SqueezeAndExcitation(channels_in,
                                           activation=activation)
        self.se_thermal = SqueezeAndExcitation(channels_in,
                                             activation=activation)

    def forward(self, rgb, thermal):
        rgb = self.se_rgb(rgb)
        thermal = self.se_thermal(thermal)
        out = rgb + thermal
        return out
