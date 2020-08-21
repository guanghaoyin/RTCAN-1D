import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

class _NonLocalBlockND(nn.Module):
    '''
    class _NonLocalBlockND:

    :param in_channels:     input channel
    :param inter_channels:  internel channel
    :param dimension:       the dimension of convolutional/BatchNorm operation in [1,2,3]
    :param mode:            the mode of non-local opeartion in ['embedded_gaussian', 'gaussian', 'dot_product', 'concatenation']
    :param sub_sample:      the bool of using sub_sample operation
    :param bn_layer:        the bool of using BatchNorm operation
    '''

    def __init__(self, in_channels, inter_channels=None, dimension=2, mode='embedded_gaussian', sub_sample=True,
                 bn_layer=False):
        super(_NonLocalBlockND, self).__init__()
        assert dimension in [1, 2, 3]
        assert mode in ['embedded_gaussian', 'gaussian', 'dot_product', 'concatenation']

        self.mode = mode
        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool = nn.MaxPool3d
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool = nn.MaxPool2d
            sub_sample = nn.Upsample
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool = nn.MaxPool1d
            bn = nn.BatchNorm1d

        self.origin_conv = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                   kernel_size=1, stride=1, padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                        kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                             kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = None
        self.phi = None
        self.concat_project = None

        if mode in ['embedded_gaussian', 'dot_product', 'concatenation']:
            self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                                 kernel_size=1, stride=1, padding=0)
            self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

            if mode == 'embedded_gaussian':
                self.operation_function = self._embedded_gaussian
            elif mode == 'dot_product':
                self.operation_function = self._dot_product
            elif mode == 'concatenation':
                self.operation_function = self._concatenation
                self.concat_project = nn.Sequential(
                    nn.Conv2d(self.inter_channels * 2, 1, 1, 1, 0, bias=False),
                    nn.ReLU()
                )
        elif mode == 'gaussian':
            self.operation_function = self._gaussian

        if sub_sample:
            self.origin_conv = nn.Sequential(self.origin_conv, max_pool(kernel_size=2))
            if self.phi is None:
                self.phi = max_pool(kernel_size=2)
            else:
                self.phi = nn.Sequential(self.phi, max_pool(kernel_size=2))

    def forward(self, x):
        '''
        :param x: (b, c, h, w)
        :return:
        '''

        output = self.operation_function(x)
        return output

    def _embedded_gaussian(self, x):
        batch_size = x.size(0)

        # origin_conv => (b, c, h, w) -> (b, 0.5c, h, w) -> (b, h*w, 0.5c)
        f_x = self.origin_conv(x).view(batch_size, self.inter_channels, -1)
        f_x = f_x.permute(0, 2, 1)

        # theta  => (b, c, h, w) -> (b, 0.5c, h, w) -> (b, h*w, 0.5c)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        # phi => (b, c, h, w) -> (b, 0.5c, h, w) -> (b, 0.5c, h*w)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        # f => (b, h*w, 0.5c)dot(b, 0.5c, h*w) -> (b, h*w, h*w)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        # (b, h*w, h*w)dot(b, h*w, 0.5c) = (b, h*w, 0.5c)->(b, 0.5c, h, w)->(b, c, h, w)
        y = torch.matmul(f_div_C, f_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)

        z = W_y + x
        return z

    def _gaussian(self, x):
        batch_size = x.size(0)

        # origin_conv => (b, c, h, w) -> (b, 0.5c, h, w) -> (b, h*w, 0.5c)
        f_x = self.origin_conv(x).view(batch_size, self.inter_channels, -1)
        f_x = f_x.permute(0, 2, 1)

        # theta_x => (b, c, h, w) -> (b, c, h*w) -> (b, h*w, c)
        theta_x = x.view(batch_size, self.in_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        # phi => (b, c, h, w) -> (b, c, h*w)
        phi_x = x.view(batch_size, self.in_channels, -1)

        # f_div_C => (b, h*w, c)dot(b, c, h*w) -> (b, h*w, h*w)
        f = torch.matmul(theta_x, phi_x)
        f_div_C = F.softmax(f, dim=-1)

        # (b, h*w, h*w)dot(b, h*w, 0.5c) = (b, h*w, 0.5c)->(b, 0.5c, h, w)->(b, c, h, w)
        y = torch.manmul(f_div_C, f_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)

        z = W_y + x
        return z

    def _dot_product(self, x):
        batch_size = x.size(0)

        # origin_conv => (b, c, h, w) -> (b, 0.5c, h, w) -> (b, h*w, 0.5c)
        f_x = self.origin_conv(x).view(batch_size, self.inter_channels, -1)
        f_x = f_x.permute(0, 2, 1)

        # theta  => (b, c, h, w) -> (b, 0.5c, h, w) -> (b, h*w, 0.5c)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)

        # phi => (b, c, h, w) -> (b, 0.5c, h, w) -> (b, 0.5c, h*w)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)

        # f => (b, h*w, 0.5c)dot(b, 0.5c, h*w) -> (b, h*w, h*w)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        # for element in f_div_C: element / h*w
        f_div_C = f / N

        # (b, h*w, h*w)dot(b, h*w, 0.5c) = (b, h*w, 0.5c)->(b, 0.5c, h, w)->(b, c, h, w)
        y = torch.matmul(f_div_C, f_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)

        z = W_y + x
        return z

    def _concatenation(self, x):
        batch_size = x.size(0)

        # origin_conv => (b, c, h, w) -> (b, 0.5c, h, w) -> (b, h*w, 0.5c)
        f_x = self.origin_conv(x).view(batch_size, self.inter_channels, -1)
        f_x = f_x.permute(0, 2, 1)

        # theta  => (b, c, h, w) -> (b, 0.5c, h, w) -> (b, 0.5c, h*w, 1)
        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)

        # phi  => (b, c, h, w) -> (b, 0.5c, h, w) -> (b, 0.5c, 1, h*w)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)

        # -> (b, 0,5c, h*w, h*w)
        h = theta_x.size(2)
        w = phi_x.size(3)
        theta_x = theta_x.repeat(1, 1, 1, w)
        phi_x = phi_x.repeat(1, 1, h, 1)

        # (b, 0.5c, h*w, h*w)cat(b, 0.5c, h*w, h*w)-> (b,c, h*w, h*w) -> (b, 1, h*w, h*w)-> (b, h*w, h*w)
        concat_feature = torch.cat([theta_x, phi_x], dim=1)
        f = self.concat_project(concat_feature)
        b, _, h, w = f.size()
        f = f.view(b, h, w)
        N = f.size(-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, f_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z


class NONLocalBlock1D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample=True, bn_layer=True):
        super(NONLocalBlock1D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=1, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self, in_channels, inter_channels=None, mode='embedded_gaussian', sub_sample=True, bn_layer=True):
        super(NONLocalBlock2D, self).__init__(in_channels,
                                              inter_channels=inter_channels,
                                              dimension=2, mode=mode,
                                              sub_sample=sub_sample,
                                              bn_layer=bn_layer)


class NONLocal2D(nn.Module):
    def __init__(self, in_feat=64, inter_feat=32, sub_sample=False, bn_layer=True):
        super(NONLocal2D, self).__init__()

        self.non_local = (
            NONLocalBlock2D(in_channels=in_feat, inter_channels=inter_feat, sub_sample=sub_sample, bn_layer=bn_layer))

    def forward(self, x):
        ## divide feature map into 4 part
        batch_size, C, H, W = x.shape
        H1 = int(H / 2)
        W1 = int(W / 2)
        nonlocal_feat = torch.zeros_like(x)

        feat_sub_lu = x[:, :, :H1, :W1]
        feat_sub_ld = x[:, :, H1:, :W1]
        feat_sub_ru = x[:, :, :H1, W1:]
        feat_sub_rd = x[:, :, H1:, W1:]

        nonlocal_lu = self.non_local(feat_sub_lu)
        nonlocal_ld = self.non_local(feat_sub_ld)
        nonlocal_ru = self.non_local(feat_sub_ru)
        nonlocal_rd = self.non_local(feat_sub_rd)
        nonlocal_feat[:, :, :H1, :W1] = nonlocal_lu
        nonlocal_feat[:, :, H1:, :W1] = nonlocal_ld
        nonlocal_feat[:, :, :H1, W1:] = nonlocal_ru
        nonlocal_feat[:, :, H1:, W1:] = nonlocal_rd

        return nonlocal_feat


class NONLocal1D(nn.Module):
    def __init__(self, in_feat=64, inter_feat=32, sub_sample=True, bn_layer=True):
        super(NONLocal1D, self).__init__()

        self.non_local = (
            NONLocalBlock1D(in_channels=in_feat, inter_channels=inter_feat, sub_sample=sub_sample, bn_layer=bn_layer))

    def forward(self, x):
        batch_size, C, L = x.shape
        L1 = int(L / 4)
        L2 = int(L / 2)
        L3 = int(3 * L / 4)
        nonlocal_feature = torch.zeros_like(x)

        feat_sub_l1 = x[:, :, :L1]
        feat_sub_l2 = x[:, :, L1: L2]
        feat_sub_l3 = x[:, :, L2: L3]
        feat_sub_l4 = x[:, :, L3:]

        nonlocal_l1 = self.non_local(feat_sub_l1)
        nonlocal_l2 = self.non_local(feat_sub_l2)
        nonlocal_l3 = self.non_local(feat_sub_l3)
        nonlocal_l4 = self.non_local(feat_sub_l4)

        nonlocal_feature[:, :, :L1] = nonlocal_l1
        nonlocal_feature[:, :, L1: L2] = nonlocal_l2
        nonlocal_feature[:, :, L2: L3] = nonlocal_l3
        nonlocal_feature[:, :, L3:] = nonlocal_l4

        return nonlocal_feature

class NONLocalG1D(nn.Module):
    def __init__(self, in_feat=64, inter_feat=32, sub_sample=True, bn_layer=True):
        super(NONLocalG1D, self).__init__()

        self.non_local = (
            NONLocalBlock1D(in_channels=in_feat, inter_channels=inter_feat, sub_sample=sub_sample, bn_layer=bn_layer))
    def forward(self,x):
        return self.non_local(x)

## 2D Channel Attention (CA) Layer
class CALayer2D(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CALayer2D, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        _,_,h,w = x.shape
        y_ave = self.avg_pool(x)
        # y_max = self.max_pool(x)
        y_ave = self.conv_du(y_ave)
        # y_max = self.conv_du(y_max)
        # y = y_ave + y_max
        # expand y to C*H*W
        # expand_y = y.expand(-1,-1,h,w)
        return y_ave*x

## 1D Channel Attention (CA) Layer
class CALayer1D(nn.Module):
    def __init__(self, channel, reduction=8):
        super(CALayer1D, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv1d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv1d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        _,_,l = x.shape
        y_ave = self.avg_pool(x)
        # y_max = self.max_pool(x)
        y_ave = self.conv_du(y_ave)
        # y_max = self.conv_du(y_max)
        # y = y_ave + y_max
        # expand y to C*H*W
        # expand_y = y.expand(-1,-1,h,w)
        return y_ave*x