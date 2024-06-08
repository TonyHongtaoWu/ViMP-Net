"""
    This code is based on:
    FuseFormer: Fusing Fine-Grained Information in Transformers for Video Inpainting, ICCV 2021
"""

import numpy as np
import time
import math
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class BaseNetwork(nn.Module):
    def __init__(self):
        super(BaseNetwork, self).__init__()

    def print_network(self):
        if isinstance(self, list):
            self = self[0]
        num_params = 0
        for param in self.parameters():
            num_params += param.numel()
        print('Network [%s] was created. Total number of parameters: %.1f million. '
              'To see the architecture, do print(network).' % (type(self).__name__, num_params / 1000000))

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('InstanceNorm2d') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight.data, 1.0)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    nn.init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError(
                        'initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

        # propagate to children
        for m in self.children():
            if hasattr(m, 'init_weights'):
                m.init_weights(init_type, gain)


class downdropbg(nn.Module):
    def __init__(self):
        super(downdropbg, self).__init__()
        self.conv_first = nn.Conv3d(4, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.conv1 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2),
                               padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
       
    def forward(self, x):
        out = F.leaky_relu(self.conv_first(x), 0.2, inplace=True)
        out = F.leaky_relu(self.conv1(out), 0.2, inplace=True)
        out = F.leaky_relu(self.conv2(out), 0.2, inplace=True)

        return out

class downdrop(nn.Module):
    def __init__(self):
        super(downdrop, self).__init__()
        self.conv_first = nn.Conv3d(3, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.conv1 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2),
                               padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1))
      

    def forward(self, x):
        out = F.leaky_relu(self.conv_first(x), 0.2, inplace=True)
        out = F.leaky_relu(self.conv1(out), 0.2, inplace=True)
        out = F.leaky_relu(self.conv2(out))

        return out

class Dual_Raindrop_gnostic_Transformer(BaseNetwork):
    def __init__(self, init_weights=True):
        super(Dual_Raindrop_gnostic_Transformer, self).__init__()
        self.channel = 512
        hidden = 512
        num_head = 4
        kernel_size = (7, 7)
        padding = (3, 3)
        stride = (3, 3)
        output_size = (64, 64)
        dropout = 0.
        t2t_params = {'kernel_size': kernel_size, 'stride': stride, 'padding': padding, 'output_size': output_size}
        n_vecs = 1
        for i, d in enumerate(kernel_size):
            n_vecs *= int((output_size[i] + 2 * padding[i] - (d - 1) - 1) / stride[i] + 1)

        self.transformer = TransformerBlock(hidden=hidden, num_head=num_head, dropout=dropout)
        self.transformer1 = TransformerBlock(hidden=hidden, num_head=num_head, dropout=dropout)
      
        self.Fusion_Transformer = TransformerBlock1(hidden=hidden, num_head=num_head, dropout=dropout)

        self.ss = SoftSplit(self.channel // 2,
                            hidden,
                            kernel_size,
                            stride,
                            padding,
                            t2t_param=t2t_params)
        self.sc = SoftComp(self.channel // 2, hidden, kernel_size, stride, padding)
        self.TokenLearner = TokenLearner()
        self.downdropbg = downdropbg()
        self.downdrop1 = downdrop()
        self.downdrop2= downdrop()
        self.conv_before_upsample1 = nn.Conv3d(self.channel // 2, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.upsample1 = nn.PixelShuffle(2)
        self.conv_before_upsample2 = nn.Conv3d(32, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.upsample2 = nn.PixelShuffle(2)
        self.conv_last = nn.Conv3d(16, 3, kernel_size=(1, 3, 3), padding=(0, 1, 1))




        if init_weights:
            self.init_weights()

    def up(self, x):
        x = self.conv_before_upsample1(x)
        # print(x.shape, 'xxxxxxxx')
        x = rearrange(x, 'n c d h w -> n d c h w')
        x = self.upsample1(x)
        x = rearrange(x, 'n d c h w -> n c d h w')
        x = self.conv_before_upsample2(x)
        x = rearrange(x, 'n c d h w -> n d c h w')
        x = self.upsample2(x)
        x = rearrange(x, 'n d c h w -> n c d h w')
        final = self.conv_last(x).transpose(1, 2)
        return final


    def forward(self, dropbg, masks):

        b, t, c, h, w = dropbg.size()
        enc_feator = self.downdrop1(dropbg.view(b, t, c, h, w).transpose(1, 2))
        enc_featdropbg = self.downdropbg(torch.cat((dropbg.view(b , t, c, h, w), masks.view(b, t, 1, h, w)), dim=2).transpose(1, 2))

        _,  ch, _, h1, w1 = enc_featdropbg.size()
        fold_output_size = (h1, w1)
        enc_feator =  enc_feator.transpose(1, 2)
        enc_featdropbg = enc_featdropbg.transpose(1, 2)
        masked_frames = (dropbg * (1 - masks).float())
        enc_featdrop = self.downdrop2(masked_frames.transpose(1, 2))
        enc_featdrop = enc_featdrop.transpose(1, 2)
        trans_feat = self.ss(enc_featdropbg.view(b * t, ch, h1, w1), b)
        trans_feator = self.ss(enc_feator.view(b * t, ch, h1, w1), b)
        trans_feat1 = self.ss(enc_featdrop.view(b * t, ch, h1, w1), b)
        trans_featcat = torch.cat((trans_feator, trans_feat1), dim=2)
        trans_feat1 = self.TokenLearner(trans_featcat)
        trans_feat, trans_feat1 = self.transformer(trans_feat, trans_feat1)
        trans_feat, trans_feat1 = self.transformer(trans_feat, trans_feat1)
        trans_feat, trans_feat1 = self.transformer1(trans_feat, trans_feat1)
        trans_feat, trans_feat1 = self.transformer1(trans_feat, trans_feat1)
        trans_Fusion = self.Fusion_Transformer(trans_feat, trans_feat1)
        trans_feat_out = self.sc(trans_Fusion, t, fold_output_size)
        trans_feat_out = trans_feat_out.view(b, t, ch, h1, w1)
        trans_feat_out = rearrange(trans_feat_out, 'n d c h w ->  n c d h w')
        output = self.up(trans_feat_out)
       
        return output


class deconv(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=3, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel,
                              kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear',
                          align_corners=True)
        return self.conv(x)


class TokenLearner(nn.Module):
    def __init__(self):
        super(TokenLearner, self).__init__()
        self.conv_layer = nn.Conv1d(1024, 512, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # print(x.shape)
        x = x.permute(0, 2, 1)
        x = self.conv_layer(x)
        x = x.permute(0, 2, 1)
     #   print(x.shape)
        return x


class TokenLearnerxxx(nn.Module):
    def __init__(self):

        super().__init__()
        self.conv_layer = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        print(x.shape)
        #input_tensor = x.permute(0, 2, 1)
        output_tensor = self.conv_layer(input_tensor)
        #output_tensor = output_tensor.permute(0, 2, 1)
        return output_tensor

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def __init__(self, p=0.1):
        super(Attention, self).__init__()
        self.dropout = nn.Dropout(p=p)

    def forward(self, query, key, value, m=None):
        scores = torch.matmul(query, key.transpose(-2, -1)
                              ) / math.sqrt(query.size(-1))
        if m is not None:
            scores.masked_fill_(m, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.dropout(p_attn)
        p_val = torch.matmul(p_attn, value)
        return p_val, p_attn

class AddPosEmb(nn.Module):
    def __init__(self, n, c):
        super(AddPosEmb, self).__init__()
        self.pos_emb = nn.Parameter(torch.zeros(1, 1, n, c).float().normal_(mean=0, std=0.02), requires_grad=True)
        self.num_vecs = n

    def forward(self, x):
        b, n, c = x.size()
        x = x.view(b, -1, self.num_vecs, c)
        x = x + self.pos_emb
        x = x.view(b, n, c)
        return x



class SoftSplit(nn.Module):
    def __init__(self, channel, hidden, kernel_size, stride, padding,
                 t2t_param):
        super(SoftSplit, self).__init__()
        self.kernel_size = kernel_size
        self.t2t = nn.Unfold(kernel_size=kernel_size,
                             stride=stride,
                             padding=padding)
        c_in = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(c_in, hidden)

        self.t2t_param = t2t_param

    def forward(self, x, b):
        feat = self.t2t(x)
        feat = feat.permute(0, 2, 1)
        feat = self.embedding(feat)
        feat = feat.view(b, -1, feat.size(2))
        
        return feat





class SoftComp(nn.Module):
    def __init__(self, channel, hidden, kernel_size, stride, padding):
        super(SoftComp, self).__init__()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        c_out = reduce((lambda x, y: x * y), kernel_size) * channel
        self.embedding = nn.Linear(hidden, c_out)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.bias_conv = nn.Conv2d(channel,
                                   channel,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)

    def forward(self, x, t, output_size):
        feat = self.embedding(x)
        b, _, c = feat.size()
        feat = feat.view(b * t, -1, c).permute(0, 2, 1)
        feat = F.fold(feat,
                      output_size=output_size,
                      kernel_size=self.kernel_size,
                      stride=self.stride,
                      padding=self.padding)
        feat = self.bias_conv(feat)
        return feat

class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, d_model, head, p=0.1):
        super().__init__()
        self.query_embedding = nn.Linear(d_model, d_model)
        self.value_embedding = nn.Linear(d_model, d_model)
        self.key_embedding = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(p=p)
        self.head = head

    def forward(self, x):
        b, n, c = x.size()
        c_h = c // self.head
        key = self.key_embedding(x)
        key = key.view(b, n, self.head, c_h).permute(0, 2, 1, 3)
        query = self.query_embedding(x)
        query = query.view(b, n, self.head, c_h).permute(0, 2, 1, 3)
        value = self.value_embedding(x)
        value = value.view(b, n, self.head, c_h).permute(0, 2, 1, 3)
        att, _ = self.attention(query, key, value)
        att = att.permute(0, 2, 1, 3).contiguous().view(b, n, c)
        output = self.output_linear(att)
        return output

class MultiHeadedAttention2(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, d_model, head, p=0.1):
        super().__init__()
        self.query_embedding = nn.Linear(d_model, d_model)
        self.value_embedding = nn.Linear(d_model, d_model)
        self.key_embedding = nn.Linear(d_model, d_model)
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention(p=p)
        self.head = head

    def forward(self, x, y):
        b, n, c = x.size()
        c_h = c // self.head
        key = self.key_embedding(y)
        key = key.view(b, n, self.head, c_h).permute(0, 2, 1, 3)
        query = self.query_embedding(x)
        query = query.view(b, n, self.head, c_h).permute(0, 2, 1, 3)
        value = self.value_embedding(y)
        value = value.view(b, n, self.head, c_h).permute(0, 2, 1, 3)
        att, _ = self.attention(query, key, value)
        att = att.permute(0, 2, 1, 3).contiguous().view(b, n, c)
        output = self.output_linear(att)
        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, p=0.1):
        super(FeedForward, self).__init__()
        self.conv = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=p),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(p=p))

    def forward(self, x):
        x = self.conv(x)
        return x


class TransformerBlock(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden=128, num_head=4, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadedAttention(d_model=hidden, head=num_head, p=dropout)
        self.ffn1 = FeedForward(hidden, p=dropout)
        self.ffn2 = FeedForward(hidden, p=dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, input1, input2):
        x1 = self.norm1(input1)
        x1 = input1 + self.dropout(self.attention(x1))
        x2 = self.norm1(input2)
        x2 = input2 + self.dropout(self.attention(x2))
        y1 = self.norm2(x1)
        y2 = self.norm2(x2)
        x1 = x1 + self.ffn1(y1)
        x2 = x2 + self.ffn2(y2)
        return x1, x2


class TransformerBlock1(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden=128, num_head=4, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadedAttention2(d_model=hidden, head=num_head, p=dropout)
        self.ffn = FeedForward(hidden, p=dropout)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(p=dropout)


    def forward(self, x, y):

        m = self.norm1(x)
        d = self.norm1(y)
        x = m + self.dropout(self.attention(m, d))

        y = self.norm2(x)
        x = x + self.ffn(y)
        return x


