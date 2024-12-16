import functools
import torch
import torch.nn as nn
from torch import nn
import torch.nn.functional as F

def psr(cup, chnls):
    return nn.Sequential(*(cup() for _ in range(chnls)))

class cl1(nn.Module):
    def __init__(self, ll=64, ci=32, bias=1):
        super(cl1, self).__init__()
        lyr = lambda adr, odr: nn.Conv2d(adr, odr, 3, 1, 1, bias=bias)

        self.conv1 = lyr(ll, ci)
        self.conv2 = lyr(ll + ci, ci)
        self.conv3 = lyr(ll + 2 * ci, ci)
        self.conv4 = lyr(ll + 3 * ci, ci)
        self.conv5 = lyr(ll + 4 * ci, ll)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, a):
        kth = [a]
        for x in range(1, 5):
            kth.append(self.lrelu(getattr(self, f'conv{x}')(torch.cat(kth, 1))))

        hgt = self.conv5(torch.cat(kth, 1))
        return 0.2 * hgt + a

class xmre(nn.Module):


    def __init__(self, nf, gc=32):
        super(xmre, self).__init__()
        self.RDB1 = cl1(nf, gc)
        self.RDB2 = cl1(nf, gc)
        self.RDB3 = cl1(nf, gc)

    def forward(self, a):
        # Forward pass through each xmre block
        ut = self.RDB3(self.RDB2(self.RDB1(a)))
        
        # Adding the original input (a) and scaling the output by 0.2
        ut = 0.2 * ut + a
        
        return ut

class network(nn.Module):
    def __init__(self, rmx, mrx, ll, xc, ci=32):
        super(network, self).__init__()

        jsr = functools.partial(xmre, nf=ll, gc=ci)

        # It is the Initial conv layer
        self.conv_first = self.ssr(rmx, ll)

        # It is for trunk
        self.RRDB_trunk = self.utr(jsr, xc, ll)

        # For the Trunk conv
        self.trunk_conv = self.ssr(ll, ll)

        # We are Upsampling the layers here
        self.upconv1 = self.ssr(ll, ll)
        self.upconv2 = self.ssr(ll, ll)

        # Now High-res conv
        self.HRconv = self.ssr(ll, ll)

        # it is Final  layer
        self.conv_last = self.ssr(ll, mrx)

        # for activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def ssr(self, ttr, iop):
        return nn.Conv2d(ttr, iop, kernel_size=3, stride=1, padding=1, bias=1)

    def utr(self, yui, nb, ll):
        return psr(yui, nb)

    def forward(self, a):
        # for the first convolution
        utopia = self.conv_first(a)
        # for processing
        jam = self.RRDB_trunk(utopia)
        jam = self.trunk_conv(jam)
        utopia = utopia + jam
        # it is for Upsampling
        utopia = F.interpolate(self.lrelu(self.upconv1(utopia)), scale_factor=2, mode='nearest')
        utopia = F.interpolate(self.lrelu(self.upconv2(utopia)), scale_factor=2, mode='nearest')
        # last conv
        xzt = self.conv_last(self.lrelu(self.HRconv(utopia)))

        return xzt
