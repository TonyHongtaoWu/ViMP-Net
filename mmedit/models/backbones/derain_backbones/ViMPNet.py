import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init
from mmcv.runner import load_checkpoint
from mmcv.cnn import constant_init, kaiming_init
from mmcv.utils.parrots_wrapper import _BatchNorm
from einops import rearrange
from mmedit.models.backbones.derain_backbones.derain_net import (
    ResidualBlocksWithInputConv, SPyNet)
from mmedit.models.common import flow_warp
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger
from modules.Dualmaskformer import Dual_Raindrop_gnostic_Transformer
from modules.DeformableAlignment import SecondOrderDeformableAlignment


@BACKBONES.register_module()
class ViMPNet(nn.Module):
    def __init__(self,
                 mid_channels=64,
                 num_blocks=9,
                 max_residue_magnitude=10,
                 spynet_pretrained=None
                 ):
        super().__init__()
        self.mid_channels = mid_channels

        # optical flow
        self.spynet = SPyNet(pretrained=spynet_pretrained)

        # feature extraction module

        self.feat_extract = nn.Sequential(
                nn.Conv2d(3, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                ResidualBlocksWithInputConv(mid_channels, mid_channels, 5))
                
        # propagation branches
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        modules = ['backward_1', 'forward_1']
        for i, module in enumerate(modules):
            self.deform_align[module] = SecondOrderDeformableAlignment(
                2 * mid_channels,
                mid_channels,
                3,
                padding=1,
                deform_groups=16,
                max_residue_magnitude=max_residue_magnitude)
            self.backbone[module] = ResidualBlocksWithInputConv(
                (2 + i) * mid_channels, mid_channels, num_blocks)

        # upsampling module
        self.reconstruction = ResNet3D(mid_channels, 5)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv_last = nn.Conv3d(16, 3, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        # check if the sequence is augmented by flipping
        self.is_mirror_extended = False

        # detect rain streak Intensity
        self.S_mask_streak = nn.Sequential(nn.Conv3d(mid_channels, mid_channels // 4, 3, 1, 1),
                                           nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                           nn.Conv3d(mid_channels // 4, mid_channels // 4, 1, 1, 0),
                                           nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                           nn.Conv3d(mid_channels // 4, 1, 3, 1, 1))
        # detect raindrop mask
        self.S_mask_drop = nn.Sequential(nn.Conv3d(16, 16 // 2, 3, 1, 1),
                                         nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                         nn.Conv3d(16 // 2, 16 // 2, 1, 1, 0),
                                         nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                         nn.Conv3d(16 // 2, 1, 3, 1, 1))

        # mask-guide raindrop removal
        self.Dual_Raindrop_gnostic_Transformer = Dual_Raindrop_gnostic_Transformer()

    def check_if_mirror_extended(self, lqs):
        """Check whether the input is a mirror-extended sequence.

        If mirror-extended, the i-th (i=0, ..., t-1) frame is equal to the
        (t-1-i)-th frame.

        Args:
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
        """

        if lqs.size(1) % 2 == 0:
            lqs_1, lqs_2 = torch.chunk(lqs, 2, dim=1)
            if torch.norm(lqs_1 - lqs_2.flip(1)) == 0:
                self.is_mirror_extended = True

    def compute_flow(self, lqs):
        """Compute optical flow using SPyNet for feature alignment.
            lqs (tensor): Input low quality (LQ) sequence with
                shape (n, t, c, h, w).
        """
        n, t, c, h, w = lqs.size()
        lqs_1 = lqs[:, :-1, :, :, :].reshape(-1, c, h, w)
        lqs_2 = lqs[:, 1:, :, :, :].reshape(-1, c, h, w)

        flows_backward = self.spynet(lqs_1, lqs_2).view(n, t - 1, 2, h, w)

        if self.is_mirror_extended:  # flows_forward = flows_backward.flip(1)
            flows_forward = None
        else:
            flows_forward = self.spynet(lqs_2, lqs_1).view(n, t - 1, 2, h, w)

        return flows_forward, flows_backward

    def propagate(self, feats, flows, module_name, streakmask):
        """Propagate the latent features throughout the sequence.

        Args:
            feats dict(list[tensor]): Features from previous branches. Each
                component is a list of tensors with shape (n, c, h, w).
            flows (tensor): Optical flows with shape (n, t - 1, 2, h, w).
            module_name (str): The name of the propagation branches. Can either
                be 'backward_1', 'forward_1', 'backward_2', 'forward_2'.

        Return:
            dict(list[tensor]): A dictionary containing all the propagated
                features. Each key in the dictionary corresponds to a
                propagation branch, which is represented by a list of tensors.
        """

        n, t, _, h, w = flows.size()

        frame_idx = range(0, t + 1)
        flow_idx = range(-1, t)
        mapping_idx = list(range(0, len(feats['spatial'])))
        mapping_idx += mapping_idx[::-1]

        if 'backward' in module_name:
            frame_idx = frame_idx[::-1]
            flow_idx = frame_idx

        feat_prop = flows.new_zeros(n, self.mid_channels, h, w)
        for i, idx in enumerate(frame_idx):
            feat_current = feats['spatial'][mapping_idx[idx]]

            # second-order deformable alignment
            if i > 0:
                flow_n1 = flows[:, flow_idx[i], :, :, :]
                streakmask_n1 = streakmask[:, flow_idx[i], :, :, :]

                cond_n1 = flow_warp(feat_prop, flow_n1.permute(0, 2, 3, 1))

                # initialize second-order features
                feat_n2 = torch.zeros_like(feat_prop)
                flow_n2 = torch.zeros_like(flow_n1)
                cond_n2 = torch.zeros_like(cond_n1)

                if i > 1:  # second-order features
                    feat_n2 = feats[module_name][-2]
                    flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                    flow_n2 = flow_n1 + flow_warp(flow_n2,
                                                  flow_n1.permute(0, 2, 3, 1))
                    cond_n2 = flow_warp(feat_n2, flow_n2.permute(0, 2, 3, 1))

                # flow-guided deformable convolution
                cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                feat_prop = self.deform_align[module_name](feat_prop, cond,
                                                           flow_n1, flow_n2, streakmask_n1)

            # concatenate and residual blocks
            feat = [feat_current] + [
                feats[k][idx]
                for k in feats if k not in ['spatial', module_name]
            ] + [feat_prop]

            feat = torch.cat(feat, dim=1)
            feat_prop = feat_prop + self.backbone[module_name](feat)
            feats[module_name].append(feat_prop)

        if 'backward' in module_name:
            feats[module_name] = feats[module_name][::-1]

        return feats



    def forward(self, lqs):

        n, t, c, h, w = lqs.size()
        
        lqs_downsample = F.interpolate(
                lqs.view(-1, c, h, w), scale_factor=0.25,
                mode='bicubic').view(n, t, c, h // 4, w // 4)

        # check whether the input is an extended sequence
        self.check_if_mirror_extended(lqs)

        feats = {}

        # compute spatial features
        feats_ = self.feat_extract(lqs.view(-1, c, h, w))

        h, w = feats_.shape[2:]
        feats_ = feats_.view(n, t, -1, h, w)

        # predicate rain streak intensity
        feats_streks = rearrange(feats_, 'n d c h w ->  n c d h w')
        streakmask = self.S_mask_streak(feats_streks)
        streakmask = rearrange(streakmask, 'n c d h w ->  n d c h w')
        
        feats['spatial'] = [feats_[:, i, :, :, :] for i in range(0, t)]
        # compute optical flow using the low-res inputs
        assert lqs_downsample.size(3) >= 64 and lqs_downsample.size(4) >= 64, (
            'The height and width of low-res inputs must be at least 64, '
            f'but got {h} and {w}.')
        flows_forward, flows_backward = self.compute_flow(lqs_downsample)

        for iter_ in [1]:
            for direction in ['backward', 'forward']:
                module = f'{direction}_{iter_}'
                feats[module] = []
                if direction == 'backward':
                    flows = flows_backward
                elif flows_forward is not None:
                    flows = flows_forward
                else:
                    flows = flows_backward.flip(1)
                feats = self.propagate(feats, flows, module, streakmask)
        
        streakmask = streakmask.view(n*t, 1, h, w)
        streakmask = F.interpolate(streakmask, size=lqs.size()[3:], mode='bilinear', align_corners=True)
        streakmask = streakmask.view(n, t, 1, h*4, w*4)

        refined = []
        num_outputs = len(feats['spatial'])

        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        for i in range(0, lqs.size(1)):
            hr = [feats[k].pop(0) for k in feats if k != 'spatial']
            hr.insert(0, feats['spatial'][mapping_idx[i]])
            hr = torch.cat(hr, dim=1)
            refined.append(hr)

        # stage 2 raindrop restore
        refinedvideo = torch.stack(refined, dim=1)
        restored = self.reconstruction(refinedvideo)
        dropmask = self.S_mask_drop(restored)
        dropmask = torch.sigmoid(dropmask)
        restored = self.conv_last(restored)
        restored = rearrange(restored, 'n c d h w ->  n d c h w')
        dropmask = rearrange(dropmask, 'n c d h w ->  n d c h w')

        stage1 = restored + lqs
        final = self.Dual_Raindrop_gnostic_Transformer(stage1, dropmask)
        output = stage1 + final

        return stage1, dropmask,  streakmask, output


    def init_weights(self, pretrained=None, strict=True):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')


class ResNet3D(nn.Module):
    def __init__(self, mid_channels=64, num_blocks=5):
        super(ResNet3D, self).__init__()
        self.input_conv = nn.Conv3d(3 * mid_channels, mid_channels, kernel_size=1, padding=0, bias=True)
        self.res_blocks = nn.Sequential(
            *[ResidualBlock3DNoBN(mid_channels=mid_channels) for _ in range(num_blocks)]
        )
        self.conv_before_upsample1 = nn.Conv3d(64, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.upsample1 = nn.PixelShuffle(2)
        self.conv_before_upsample2 = nn.Conv3d(32, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.upsample2 = nn.PixelShuffle(2)

    def forward(self, x):
        x = rearrange(x, 'n d c h w ->  n c d h w')
        x = self.input_conv(x)
        x = self.res_blocks(x)  
        x = self.conv_before_upsample1(x)
        x = rearrange(x, 'n c d h w -> n d c h w')
        x = self.upsample1(x)
        x = rearrange(x, 'n d c h w -> n c d h w')
        x = self.conv_before_upsample2(x)
        x = rearrange(x, 'n c d h w -> n d c h w')
        x = self.upsample2(x)
        x = rearrange(x, 'n d c h w -> n c d h w')
        return x


def default_init_weights(module, scale=1):

    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, nn.Linear):
            kaiming_init(m, a=0, mode='fan_in', bias=0)
            m.weight.data *= scale
        elif isinstance(m, _BatchNorm):
            constant_init(m.weight, val=1, bias=0)


class ResidualBlock3DNoBN(nn.Module):

    def __init__(self, mid_channels=64, res_scale=1.0):
        super().__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv3d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=True)
        self.conv2 = nn.Conv3d(mid_channels, mid_channels, kernel_size=3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if res_scale == 1.0:
            self.init_weights()

    def init_weights(self):
        default_init_weights(self, 0.1)

    def forward(self, x):

        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale
