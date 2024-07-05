"""
Generalization from https://github.com/JianghaoWu/SIFA-pytorch
Use repository's parameters's values as default here (except cin|cout|dimensions
"""
import torch
import torch.nn as nn

from net_utils import CONV, CONVT, MAXPOOL, BNORM, INORM



class InsResBlock(nn.Module):
    def __init__(self, cin, dimensions=3, kernel_size=3, stride=1, pad=1):
        super(InsResBlock, self).__init__()
        conv, inorm = CONV[dimensions], INORM[dimensions]
        self.layer = nn.Sequential(conv(cin, cin, kernel_size, stride, pad),
                                   inorm(cin),
                                   nn.ReLU(inplace=True),
                                   conv(cin, cin, kernel_size, stride, pad),
                                   inorm(cin))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.layer(x))


class BasicBlock(nn.Module):
    def __init__(self, cin, cout, dimensions=3, kernel_size=3, stride=1, pad=1,
                 norm=None, dropout=None, do_relu=True):
        super(BasicBlock, self).__init__()
        layer = [ CONV[dimensions](cin, cout, kernel_size, stride, pad) ]
        if dropout is not None:
            layer += [ nn.Dropout(dropout) ]
        if norm is not None:
            if norm == 'BN':
                layer += [ BNORM[dimensions](cout) ]
            if norm == 'IN':
                layer += [ INORM[dimensions](cout)]
        if do_relu:
            layer += [ nn.ReLU(inplace=True) ]
        self.layer = nn.Sequential(*layer)

    def forward(self, x):
        return self.layer(x)


class InsDeconv(nn.Module):
    def __init__(self, cin, cout, dimensions=3, kernel_size=3, stride=2, pad=1, opad=1):
        super(InsDeconv, self).__init__()
        self.layer = nn.Sequential(CONVT[dimensions](cin, cout, kernel_size,
                                                     stride, pad, opad),
                                   INORM[dimensions](cout),
                                   nn.ReLU(inplace=True))

    def forward(self, x):
        return self.layer(x)


class DilateBlock(nn.Module):
    def __init__(self, cin, cout, dimensions=3, kernel_size=3, stride=1, pad=2,
                 dilatation=2, dropout=0.25):
        super(DilateBlock, self).__init__()
        conv, norm = CONV[dimensions], BNORM[dimensions]
        self.layer = nn.Sequential(conv(cin, cout, kernel_size, stride, pad,
                                        dilation=dilatation),
                                   nn.Dropout(dropout),
                                   norm(cout),
                                   nn.ReLU(inplace=True),
                                   conv(cin, cout, kernel_size, stride, pad,
                                        dilatation),
                                   nn.Dropout(dropout),
                                   norm(cout))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.layer(x))


class BNResBlock(nn.Module):
    def __init__(self, cin, cout, dimensions=3, kernel_size=3, stride=1, pad=1, dropout=0.25):
        super(BNResBlock, self).__init__()
        # cin == cout or cout = 2 * cin
        self.cin, self.cout = cin, cout
        conv, norm = CONV[dimensions], BNORM[dimensions]
        self.layer = nn.Sequential(conv(cin, cout, kernel_size, stride, pad),
                                   nn.Dropout(dropout),
                                   norm(cout),
                                   nn.ReLU(inplace=True),
                                   conv(cout, cout, kernel_size, stride, pad),
                                   nn.Dropout(dropout),
                                   norm(cout))
        self.relu = nn.ReLU(inplace=True)

    def expand_channels(self, x):
        # expand channels [B, C, H, W] to [B, 2C, H, W]
        all0 = torch.zeros_like(x)
        z1, z2 = torch.split(all0, x.size(1) // 2, dim=1)
        return torch.cat((z1, x, z2), dim=1)

    def forward(self, x):
        if self.cin == self.cout:
            return self.relu(x + self.layer(x))
        else:
            out0 = self.layer(x)
            out1 = self.expand_channels(x)
            return self.relu(out0 + out1)



class SIFAGenerator(nn.Module):
    """G from cyclegan"""
    def __init__(self, cin=1, channels=(32, 64, 128), dimensions=3,
                 kernel_size=(7, 3, 3), strides=(1, 2, 2), pads=(3, 1, 1),
                 norm="IN", skip=True):
        super(SIFAGenerator, self).__init__()
        self._skip = skip
        model, i = [], cin
        for o, k, s, p in zip(channels, kernel_size, strides, pads):
            model.append(BasicBlock(i, o, dimensions, k, s, p, norm))
            i = o
        model.extend([ InsResBlock(channels[-1], dimensions) ] * 9)
        for j in range(len(channels) -1, 0, -1):
            model.append(InsDeconv(channels[j], channels[j - 1], dimensions))
        model.append(BasicBlock(channels[0], cin, dimensions, kernel_size[0],
                                strides[0], pads[0], do_relu=False))
        self.model = nn.Sequential(*model)
        self.tanh = nn.Tanh()

    def forward(self, inputgen, inputimg):
        out = self.model(inputgen)
        if self._skip is True:
            out = self.tanh(out + inputimg)
        else:
            out = self.tanh(out)
        return out


class SIFAEncoder(nn.Module):
    def __init__(self, cin=1, channels=(16, 32, 64, 128, 256, 512), dimensions=3,
                 kernel_size=7, stride=1, pad=3, norm="BN", pool_kernel_size=2,
                 dropout=0.25):
        super(SIFAEncoder, self).__init__()
        #BasicBlock dim, cin, cout, ksize, stride, pad, norm, dropout, do_relu
        #BNResBlock cin, cout, ksize, stride, pad, dropout
        pool = MAXPOOL[dimensions]
        self.model = nn.Sequential(
            BasicBlock(cin, channels[0], dimensions, kernel_size, stride, pad,
                       norm, dropout),
            BNResBlock(channels[0], channels[0], dimensions),
            pool(pool_kernel_size),
            BNResBlock(channels[0], channels[1], dimensions),
            pool(pool_kernel_size),
            BNResBlock(channels[1], channels[2], dimensions),
            BNResBlock(channels[2], channels[2], dimensions),
            pool(pool_kernel_size),
            BNResBlock(channels[2], channels[3], dimensions),
            BNResBlock(channels[3], channels[3], dimensions),
            BNResBlock(channels[3], channels[4], dimensions),
            BNResBlock(channels[4], channels[4], dimensions),
            BNResBlock(channels[4], channels[4], dimensions),
            BNResBlock(channels[4], channels[4], dimensions),
            BNResBlock(channels[4], channels[5], dimensions),
            BNResBlock(channels[5], channels[5], dimensions),
            DilateBlock(channels[5], channels[5], dimensions),
            DilateBlock(channels[5], channels[5], dimensions),
            BasicBlock(channels[5], channels[5], dimensions, 3, stride, 1,
                       norm, dropout),
            BasicBlock(channels[5], channels[5], dimensions, 3, stride, 1,
                       norm, dropout)
        )

    def forward(self, x):
        out = self.model(x)
        return out


class SIFADecoder(nn.Module):
    def __init__(self, channels=(512, 128, 64, 64, 32, 1), dimensions=3,
                 kernel_size=(3, 7), stride=(1, 1), pad=(1, 3), norm="IN", skip=True):
        super(SIFADecoder, self).__init__()
        self._skip = skip
        model = [BasicBlock(channels[0], channels[1], dimensions,
                             kernel_size[0], stride[0], pad[0], norm)]
        model.extend([InsResBlock(channels[1], dimensions)] * 4)
        for j in range(1, len(channels) - 2):
            model.append(InsDeconv(channels[j], channels[j + 1], dimensions))
        model.append(BasicBlock(channels[-2], channels[-1], dimensions,
                                kernel_size[-1], stride[-1], pad[-1], do_relu=False))
        self.model = nn.Sequential(*model)
        self.tanh = nn.Tanh()

    def forward(self, inputde, inputimg):
        out = self.model(inputde)
        if self._skip is True:
            out = self.tanh(out + inputimg)
        else:
            out = self.tanh(out)
        return out


class SIFASeg(nn.Module):
    def __init__(self, cin=512, cout=3, dimensions=3):
        super(SIFASeg, self).__init__()
        self.model = nn.Sequential(BasicBlock(cin, cout, dimensions, 1, 1, 0, do_relu=False),
                                   nn.Upsample(scale_factor=8, mode='trilinear'))

    def forward(self, x):
        x = self.model(x)
        return x


class SIFADis(nn.Module):
    def __init__(self, cin=1, channels=(64, 128, 256, 512), dimensions=3,
                 kernel_size=4, stride=2, pad=1, leaky_relu=0.2):
        super(SIFADis, self).__init__()
        conv, norm = CONV[dimensions], INORM[dimensions]
        model, i = [], cin
        for do_norm, o in enumerate(channels):
            #FIXME: Last stride should be one
            model.append(conv(i, o, kernel_size, stride, pad))
            if do_norm:
                model.append(norm(o))
            model.append(nn.LeakyReLU(leaky_relu, inplace=True))
            i = o
        model.append(conv(channels[-1], 1, kernel_size, padding=pad))
        self.model = nn.Sequential(*model)

    def forward(self, x):
        x = self.model(x)
        return x


class SIFADisAux(nn.Module):
    def __init__(self, cin=1, channels=(64, 128, 256, 512), dimensions=3,
                 kernel_size=4, stride=2, pad=1, leaky_relu=0.2):
        super(SIFADisAux, self).__init__()
        conv, norm = CONV[dimensions], INORM[dimensions]
        model, i = [], cin
        for do_norm, o in enumerate(channels):
            #FIXME: Last stride should be one
            model.append(conv(i, o, kernel_size, stride, pad))
            if do_norm:
                model.append(norm(o))
            model.append(nn.LeakyReLU(leaky_relu, inplace=True))
            i = o
        self.share = nn.Sequential(*model)
        self.model = nn.Sequential(conv(channels[-1], 1, kernel_size, padding=pad))
        self.model_aux = nn.Sequential(conv(channels[-1], 1, kernel_size, padding=pad))


    def forward(self, x):
        x = self.share(x)
        x = self.model(x)
        return x

    def forward_aux(self,x):
        x = self.share(x)
        x = self.model_aux(x)
        return x
