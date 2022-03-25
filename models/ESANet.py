# -*- coding: utf-8 -*-
"""
.. codeauthor:: Mona Koehler <mona.koehler@tu-ilmenau.de>
.. codeauthor:: Daniel Seichter <daniel.seichter@tu-ilmenau.de>
"""
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet import ResNet18, ResNet34, ResNet50
from models.rgb_thermal_fusion import SqueezeAndExciteFusionAdd
from models.context_modules import get_context_module
from models.resnet import BasicBlock, NonBottleneck1D
from models.model_utils import ConvBNAct, Swish, Hswish
from . import mix_transformer


class ESANet(nn.Module):
    def __init__(self,
                 height=480,
                 width=640,
                 n_class=37,
                 encoder_rgb='resnet18',
                 encoder_thermal='resnet18',
                 encoder_block='BasicBlock',
                 channels_decoder=[512, 256, 128],  # default: [128, 128, 128]
                 pretrained_on_imagenet=True,
                 pretrained_dir='./trained_models/imagenet',
                 activation='relu',
                 encoder_decoder_fusion='add',
                 context_module='ppm',
                 nr_decoder_blocks=[1, 2, 3],  # default: [1, 1, 1]
                 fuse_thermal_in_rgb_encoder='SE-add',
                 upsampling='bilinear'):

        super(ESANet, self).__init__()

        if channels_decoder is None:
            channels_decoder = [128, 128, 128]
        if nr_decoder_blocks is None:
            nr_decoder_blocks = [1, 1, 1]

        self.fuse_thermal_in_rgb_encoder = fuse_thermal_in_rgb_encoder

        # set activation function
        if activation.lower() == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation.lower() in ['swish', 'silu']:
            self.activation = Swish()
        elif activation.lower() == 'hswish':
            self.activation = Hswish()
        else:
            raise NotImplementedError(
                'Only relu, swish and hswish as activation function are '
                'supported so far. Got {}'.format(activation))

        if encoder_rgb == 'resnet50' or encoder_thermal == 'resnet50':
            warnings.warn('Parameter encoder_block is ignored for ResNet50. '
                          'ResNet50 always uses Bottleneck')

        # rgb encoder
        if encoder_rgb == 'mit_b0':
            self.encoder_rgb = getattr(mix_transformer, encoder_rgb)()
        elif encoder_rgb == 'mit_b1':
            self.encoder_rgb = getattr(mix_transformer, encoder_rgb)()
        elif encoder_rgb == 'mit_b3':
            self.encoder_rgb = getattr(mix_transformer, encoder_rgb)()
        else:
            raise NotImplementedError(
                'Only ResNets are supported for '
                'encoder_rgb. Got {}'.format(encoder_rgb))

        # thermal encoder
        if encoder_thermal == 'resnet18':
            self.encoder_thermal = ResNet18(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation,
                input_channels=1)
        elif encoder_thermal == 'resnet34':
            self.encoder_thermal = ResNet34(
                block=encoder_block,
                pretrained_on_imagenet=pretrained_on_imagenet,
                pretrained_dir=pretrained_dir,
                activation=self.activation,
                input_channels=1)
        elif encoder_thermal == 'resnet50':
            self.encoder_thermal = ResNet50(
                pretrained_on_imagenet=pretrained_on_imagenet,
                activation=self.activation,
                input_channels=1)
        else:
            raise NotImplementedError(
                'Only ResNets are supported for '
                'encoder_thermal. Got {}'.format(encoder_rgb))

        self.channels_decoder_in = self.encoder_thermal.down_32_channels_out

        if fuse_thermal_in_rgb_encoder == 'SE-add':
            self.se_layer0 = SqueezeAndExciteFusionAdd(
                64, activation=self.activation)
            self.se_layer1 = SqueezeAndExciteFusionAdd(
                self.encoder_thermal.down_4_channels_out,
                activation=self.activation)
            self.se_layer2 = SqueezeAndExciteFusionAdd(
                self.encoder_thermal.down_8_channels_out,
                activation=self.activation)
            self.se_layer3 = SqueezeAndExciteFusionAdd(
                self.encoder_thermal.down_16_channels_out,
                activation=self.activation)
            self.se_layer4 = SqueezeAndExciteFusionAdd(
                self.encoder_thermal.down_32_channels_out,
                activation=self.activation)
        elif fuse_thermal_in_rgb_encoder == 'fusion':
            self.se_layer0 = SqueezeAndExciteFusionAdd(
                64, activation=self.activation)
            self.se_layer1 = FusionModule(channel=self.encoder_rgb.down_4_channels_out)
            self.se_layer2 = FusionModule(channel=self.encoder_rgb.down_8_channels_out)
            self.se_layer3 = FusionModule(channel=self.encoder_rgb.down_16_channels_out)
            self.se_layer4 = FusionModule(channel=self.encoder_rgb.down_32_channels_out)

        if encoder_decoder_fusion == 'add':
            layers_skip1 = list()
            if self.encoder_thermal.down_4_channels_out != channels_decoder[2]:
                layers_skip1.append(ConvBNAct(
                    self.encoder_thermal.down_4_channels_out,
                    channels_decoder[2],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer1 = nn.Sequential(*layers_skip1)

            layers_skip2 = list()
            if self.encoder_thermal.down_8_channels_out != channels_decoder[1]:
                layers_skip2.append(ConvBNAct(
                    self.encoder_thermal.down_8_channels_out,
                    channels_decoder[1],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer2 = nn.Sequential(*layers_skip2)

            layers_skip3 = list()
            if self.encoder_thermal.down_16_channels_out != channels_decoder[0]:
                layers_skip3.append(ConvBNAct(
                    self.encoder_thermal.down_16_channels_out,
                    channels_decoder[0],
                    kernel_size=1,
                    activation=self.activation))
            self.skip_layer3 = nn.Sequential(*layers_skip3)

        elif encoder_decoder_fusion == 'None':
            self.skip_layer0 = nn.Identity()
            self.skip_layer1 = nn.Identity()
            self.skip_layer2 = nn.Identity()
            self.skip_layer3 = nn.Identity()

        # context module
        if 'learned-3x3' in upsampling:
            warnings.warn('for the context module the learned upsampling is '
                          'not possible as the feature maps are not upscaled '
                          'by the factor 2. We will use nearest neighbor '
                          'instead.')
            upsampling_context_module = 'nearest'
        else:
            upsampling_context_module = upsampling
        self.context_module, channels_after_context_module = \
            get_context_module(
                context_module,
                self.channels_decoder_in,
                channels_decoder[0],
                input_size=(height // 32, width // 32),
                activation=self.activation,
                upsampling_mode=upsampling_context_module
            )

        # decoder
        self.decoder = Decoder(
            channels_in=channels_after_context_module,
            channels_decoder=channels_decoder,
            activation=self.activation,
            nr_decoder_blocks=nr_decoder_blocks,
            encoder_decoder_fusion='concat',
            upsampling_mode=upsampling,
            num_classes=n_class
        )
        self.decoder_thermal = DecoderThermal(
            channels_in=channels_after_context_module,
            channels_decoder=channels_decoder,
            activation=self.activation,
            nr_decoder_blocks=nr_decoder_blocks,
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling,
            num_classes=n_class
        )

    def forward(self, input_img):
        B = input_img.shape[0]
        rgb = input_img[:, :3]
        thermal = input_img[:, 3:]

        # rgb = self.encoder_rgb.forward_first_conv(rgb)
        thermal = self.encoder_thermal.forward_first_conv(thermal)

        # if self.fuse_thermal_in_rgb_encoder == 'add':
        #     fuse = rgb + thermal
        # else:
        #     fuse = self.se_layer0(rgb, thermal)

        # rgb = F.max_pool2d(fuse, kernel_size=3, stride=2, padding=1)
        thermal = F.max_pool2d(thermal, kernel_size=3, stride=2, padding=1)

        # block 1
        rgb = self.encoder_rgb.forward_layer1(rgb, B)
        thermal = self.encoder_thermal.forward_layer1(thermal)
        if self.fuse_thermal_in_rgb_encoder == 'add':
            fuse = rgb + thermal
        else:
            fuse = self.se_layer1(rgb, thermal)
        skip1 = self.skip_layer1(fuse)

        # block 2
        rgb = self.encoder_rgb.forward_layer2(fuse, B)
        thermal = self.encoder_thermal.forward_layer2(thermal)
        if self.fuse_thermal_in_rgb_encoder == 'add':
            fuse = rgb + thermal
        else:
            fuse = self.se_layer2(rgb, thermal)
        skip2 = self.skip_layer2(fuse)

        # block 3
        rgb = self.encoder_rgb.forward_layer3(fuse, B)
        thermal = self.encoder_thermal.forward_layer3(thermal)
        if self.fuse_thermal_in_rgb_encoder == 'add':
            fuse = rgb + thermal
        else:
            fuse = self.se_layer3(rgb, thermal)
        skip3 = self.skip_layer3(fuse)

        # block 4
        rgb = self.encoder_rgb.forward_layer4(fuse, B)
        thermal = self.encoder_thermal.forward_layer4(thermal)
        if self.fuse_thermal_in_rgb_encoder == 'add':
            fuse = rgb + thermal
        else:
            fuse = self.se_layer4(rgb, thermal)

        out = self.context_module(fuse)
        out = self.decoder(enc_outs=[out, skip3, skip2, skip1])

        thermal = self.decoder_thermal(enc_outs=thermal)

        return out, thermal


class Decoder(nn.Module):
    def __init__(self,
                 channels_in,
                 channels_decoder,
                 activation=nn.ReLU(inplace=True),
                 nr_decoder_blocks=1,
                 encoder_decoder_fusion='add',
                 upsampling_mode='bilinear',
                 num_classes=37):
        super().__init__()

        self.decoder_module_1 = DecoderModule(
            channels_in=channels_in,
            channels_dec=channels_decoder[0],
            activation=activation,
            nr_decoder_blocks=nr_decoder_blocks[0],
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling_mode,
            num_classes=num_classes
        )

        self.decoder_module_2 = DecoderModule(
            channels_in=channels_decoder[0],
            channels_dec=channels_decoder[1],
            activation=activation,
            nr_decoder_blocks=nr_decoder_blocks[1],
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling_mode,
            num_classes=num_classes
        )

        self.decoder_module_3 = DecoderModule(
            channels_in=channels_decoder[1],
            channels_dec=channels_decoder[2],
            activation=activation,
            nr_decoder_blocks=nr_decoder_blocks[2],
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling_mode,
            num_classes=num_classes
        )
        out_channels = channels_decoder[2]

        self.conv_out = nn.Conv2d(out_channels,
                                  num_classes, kernel_size=3, padding=1)

        # upsample twice with factor 2
        self.upsample1 = Upsample(mode=upsampling_mode,
                                  channels=num_classes)
        self.upsample2 = Upsample(mode=upsampling_mode,
                                  channels=num_classes)
        self.upsample_32 = nn.Upsample(scale_factor=32, mode=upsampling_mode, align_corners=True)
        self.upsample_16 = nn.Upsample(scale_factor=16, mode=upsampling_mode, align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode=upsampling_mode, align_corners=True)

    def forward(self, enc_outs):
        enc_out, enc_skip_down_16, enc_skip_down_8, enc_skip_down_4 = enc_outs

        out, out_down_32 = self.decoder_module_1(enc_out, enc_skip_down_16)
        out, out_down_16 = self.decoder_module_2(out, enc_skip_down_8)
        out, out_down_8 = self.decoder_module_3(out, enc_skip_down_4)

        out = self.conv_out(out)
        out = self.upsample1(out)
        out = self.upsample2(out)

        if self.training:
            return [out, self.upsample_8(out_down_8), self.upsample_16(out_down_16), self.upsample_32(out_down_32)]
        return out


class DecoderThermal(nn.Module):
    def __init__(self,
                 channels_in,
                 channels_decoder,
                 activation=nn.ReLU(inplace=True),
                 nr_decoder_blocks=1,
                 encoder_decoder_fusion='add',
                 upsampling_mode='bilinear',
                 num_classes=37):
        super().__init__()

        self.decoder_module_1 = DecoderThermalModule(
            channels_in=channels_in,
            channels_dec=channels_decoder[0],
            activation=activation,
            nr_decoder_blocks=nr_decoder_blocks[0],
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling_mode,
            num_classes=num_classes
        )

        self.decoder_module_2 = DecoderThermalModule(
            channels_in=channels_decoder[0],
            channels_dec=channels_decoder[1],
            activation=activation,
            nr_decoder_blocks=nr_decoder_blocks[1],
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling_mode,
            num_classes=num_classes
        )

        self.decoder_module_3 = DecoderThermalModule(
            channels_in=channels_decoder[1],
            channels_dec=channels_decoder[2],
            activation=activation,
            nr_decoder_blocks=nr_decoder_blocks[2],
            encoder_decoder_fusion=encoder_decoder_fusion,
            upsampling_mode=upsampling_mode,
            num_classes=num_classes
        )
        out_channels = channels_decoder[2]

        self.conv_out = nn.Conv2d(out_channels,
                                  num_classes, kernel_size=3, padding=1)

        # upsample twice with factor 2
        self.upsample1 = Upsample(mode=upsampling_mode,
                                  channels=num_classes)
        self.upsample2 = Upsample(mode=upsampling_mode,
                                  channels=num_classes)
        self.upsample_32 = nn.Upsample(scale_factor=32, mode=upsampling_mode, align_corners=True)
        self.upsample_16 = nn.Upsample(scale_factor=16, mode=upsampling_mode, align_corners=True)
        self.upsample_8 = nn.Upsample(scale_factor=8, mode=upsampling_mode, align_corners=True)

    def forward(self, enc_outs):
        enc_out = enc_outs

        out, out_down_32 = self.decoder_module_1(enc_out)
        out, out_down_16 = self.decoder_module_2(out)
        out, out_down_8 = self.decoder_module_3(out)

        out = self.conv_out(out)
        out = self.upsample1(out)
        out = self.upsample2(out)

        if self.training:
            return [out, self.upsample_8(out_down_8), self.upsample_16(out_down_16), self.upsample_32(out_down_32)]
        return out


class DecoderModule(nn.Module):
    def __init__(self,
                 channels_in,
                 channels_dec,
                 activation=nn.ReLU(inplace=True),
                 nr_decoder_blocks=1,
                 encoder_decoder_fusion='add',
                 upsampling_mode='bilinear',
                 num_classes=37):
        super().__init__()
        self.upsampling_mode = upsampling_mode
        self.encoder_decoder_fusion = encoder_decoder_fusion

        self.conv3x3 = ConvBNAct(channels_in, channels_dec, kernel_size=3,
                                 activation=activation)

        blocks = []
        for _ in range(nr_decoder_blocks):
            blocks.append(NonBottleneck1D(channels_dec,
                                          channels_dec,
                                          activation=activation)
                          )
        self.decoder_blocks = nn.Sequential(*blocks)

        self.upsample = Upsample(mode=upsampling_mode,
                                 channels=channels_dec)

        # for pyramid supervision
        self.side_output = nn.Conv2d(channels_dec,
                                     num_classes,
                                     kernel_size=1)
        self.conv1x1 = nn.Conv2d(2*channels_dec, channels_dec, kernel_size=1)

    def forward(self, decoder_features, encoder_features):
        out = self.conv3x3(decoder_features)
        out = self.decoder_blocks(out)

        if self.training:
            out_side = self.side_output(out)
        else:
            out_side = None

        out = self.upsample(out)

        if self.encoder_decoder_fusion == 'add':
            out += encoder_features
        else:
            out = self.conv1x1(torch.cat([out, encoder_features], dim=1))

        return out, out_side


class DecoderThermalModule(nn.Module):
    def __init__(self,
                 channels_in,
                 channels_dec,
                 activation=nn.ReLU(inplace=True),
                 nr_decoder_blocks=1,
                 encoder_decoder_fusion='add',
                 upsampling_mode='bilinear',
                 num_classes=37):
        super().__init__()
        self.upsampling_mode = upsampling_mode
        self.encoder_decoder_fusion = encoder_decoder_fusion

        self.conv3x3 = ConvBNAct(channels_in, channels_dec, kernel_size=3,
                                 activation=activation)

        blocks = []
        for _ in range(nr_decoder_blocks):
            blocks.append(NonBottleneck1D(channels_dec,
                                          channels_dec,
                                          activation=activation)
                          )
        self.decoder_blocks = nn.Sequential(*blocks)

        self.upsample = Upsample(mode=upsampling_mode,
                                 channels=channels_dec)

        # for pyramid supervision
        self.side_output = nn.Conv2d(channels_dec,
                                     num_classes,
                                     kernel_size=1)
        self.conv1x1 = nn.Conv2d(2 * channels_dec, channels_dec, kernel_size=1)

    def forward(self, decoder_features):
        out = self.conv3x3(decoder_features)
        out = self.decoder_blocks(out)

        if self.training:
            out_side = self.side_output(out)
        else:
            out_side = None

        out = self.upsample(out)

        return out, out_side


class Upsample(nn.Module):
    def __init__(self, mode, channels=None):
        super(Upsample, self).__init__()
        self.interp = nn.functional.interpolate

        if mode == 'bilinear':
            self.align_corners = False
        else:
            self.align_corners = None

        if 'learned-3x3' in mode:
            # mimic a bilinear interpolation by nearest neigbor upscaling and
            # a following 3x3 conv. Only works as supposed when the
            # feature maps are upscaled by a factor 2.

            if mode == 'learned-3x3':
                self.pad = nn.ReplicationPad2d((1, 1, 1, 1))
                self.conv = nn.Conv2d(channels, channels, groups=channels,
                                      kernel_size=3, padding=0)
            elif mode == 'learned-3x3-zeropad':
                self.pad = nn.Identity()
                self.conv = nn.Conv2d(channels, channels, groups=channels,
                                      kernel_size=3, padding=1)

            # kernel that mimics bilinear interpolation
            w = torch.tensor([[[
                [0.0625, 0.1250, 0.0625],
                [0.1250, 0.2500, 0.1250],
                [0.0625, 0.1250, 0.0625]
            ]]])

            self.conv.weight = torch.nn.Parameter(torch.cat([w] * channels))

            # set bias to zero
            with torch.no_grad():
                self.conv.bias.zero_()

            self.mode = 'nearest'
        else:
            # define pad and conv just to make the forward function simpler
            self.pad = nn.Identity()
            self.conv = nn.Identity()
            self.mode = mode

    def forward(self, x):
        size = (int(x.shape[2]*2), int(x.shape[3]*2))
        x = self.interp(x, size, mode=self.mode,
                        align_corners=self.align_corners)
        x = self.pad(x)
        x = self.conv(x)
        return x


class FusionModule(nn.Module):
    def __init__(self, channel, ksize=1):
        super(FusionModule, self).__init__()
        self.query = nn.Conv2d(2*channel, channel, kernel_size=ksize, bias=False)
        self.key = nn.Conv2d(2*channel, channel, kernel_size=ksize, bias=False)
        self.value = nn.Conv2d(2*channel, channel, kernel_size=ksize, bias=False)
        self.conv = nn.Conv2d(channel, channel, kernel_size=1, stride=1, bias=False)

    def forward(self, rgb, thermal):
        fused = torch.cat([rgb, thermal], dim=1)
        query = self.query(fused)
        key = self.key(fused)
        value = self.value(fused)

        N, C, H, W = query.shape
        query = query.permute(0, 2, 3, 1).view((N, H * W, C))
        key = key.view(N, C, H * W)
        value = value.permute(0, 2, 3, 1).view((N, H * W, C))
        score = torch.matmul(query, key)
        score = F.softmax(score, dim=2)
        out = torch.matmul(score, value)
        out = out.permute(0, 2, 1).view(N, -1, H, W)
        out = self.conv(out)

        return out


def main():
    height = 480
    width = 640

    model = ESANet(height=height, width=width)

    print(model)

    model.eval()
    rgb_image = torch.randn(1, 3, height, width)
    thermal_image = torch.randn(1, 1, height, width)

    with torch.no_grad():
        output = model(rgb_image, thermal_image)
    print(output.shape)


if __name__ == '__main__':
    main()
