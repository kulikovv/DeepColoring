import math

import torch
import torch.nn as nn


def clip_align(x, y):
    deltax = (y.size(2) - x.size(2)) / 2
    deltay = (y.size(3) - x.size(3)) / 2

    if deltax > 0 and deltay > 0:
        y = y[:, :, deltax:-deltax, deltay:-deltay]
    return y


class DownModule(nn.Module):
    """
    Downscale module
    """

    def __init__(self, in_dims, out_dims, repeats=1, padding=0, non_linearity=nn.ELU, use_dropout=False, use_bn=False):
        super(DownModule, self).__init__()
        layers = [nn.Conv2d(in_dims, out_dims, 3, padding=padding), non_linearity(inplace=True)]

        for i in range(repeats):
            layers += [nn.Conv2d(out_dims, out_dims, 3, padding=padding)]
            if use_bn:
                layers += [nn.BatchNorm2d(out_dims)]
            layers += [non_linearity(inplace=True)]

        if use_dropout:
            layers += [nn.Dropout2d(p=0.1)]

        self.convs = nn.Sequential(*layers)
        self.pool = nn.MaxPool2d(2, 2)
        self.non_ln = non_linearity(inplace=True)

    def forward(self, x):
        return self.pool(self.convs(x))


class UpModule(nn.Module):
    """
    Upscale module
    """

    def __init__(self, in_dims, out_dims, repeats=1, padding=0, non_linearity=nn.ELU):
        super(UpModule, self).__init__()
        self.conv = nn.ConvTranspose2d(in_dims, out_dims, 2, stride=2)
        layers = [nn.Conv2d(2 * out_dims, out_dims, 3, padding=padding), non_linearity(inplace=True)]
        for i in range(repeats):
            layers += [nn.Conv2d(out_dims, out_dims, 3, padding=padding), non_linearity(inplace=True)]

        self.normconv = nn.Sequential(*[nn.Conv2d(out_dims, out_dims, 2, padding=padding), non_linearity(inplace=True)])
        self.convs = nn.Sequential(*layers)

    def forward(self, x, y):

        x = self.conv(x)

        if 1 == y.size(2) % 2:
            y = self.normconv(y)

        y = clip_align(x, y)

        x = torch.cat([x, y], dim=1)
        return self.convs(x)


class EUnet(nn.Module):
    """
    Deep neural network with skip connections
    """

    def __init__(self, in_dims, out_dims, k=1, s=1, l=1, depth=3, base=8, init_xavier=False, padding=0,
                 non_linearity=nn.ReLU, use_dropout=False, use_bn=False):
        """
        Creates a u-net network
        :param in_dims: input image number of channels
        :param out_dims: number of feature maps
        :param k: width coefficient
        :param s: number of repeats in encoder part
        :param l: number of repeats in decoder part
        """
        super(EUnet, self).__init__()
        self.conv = nn.Conv2d(in_dims, base * k, 3, padding=padding)

        self.depth = depth
        self.down = []
        self.up = []

        for i in range(self.depth):
            dn = DownModule(base * (2 ** i) * k, base * (2 ** (i + 1)) * k, s, non_linearity=non_linearity,
                            padding=padding, use_dropout=use_dropout, use_bn=use_bn)
            up = UpModule(base * (2 ** (i + 1)) * k, base * (2 ** i) * k, l, non_linearity=non_linearity,
                          padding=padding)
            self.add_module("Down" + str(i), dn)
            self.add_module("Up" + str(i), up)
            self.down.append(dn)
            self.up.append(up)

        self.conv1x1 = nn.Conv2d(8 * k, out_dims, 1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if init_xavier:
                    torch.nn.init.xavier_uniform_(m.weight)
                else:
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        inter = [self.conv(x)]
        for i in range(self.depth):
            dn = self.down[i](inter[i])
            inter.append(dn)

        up = inter[-1]
        for i in range(1, self.depth + 1):
            m = self.up[self.depth - i]
            up = m(up, inter[-i - 1])

        return self.conv1x1(up)
