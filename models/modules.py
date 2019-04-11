import torch
from functools import partial
import models.utils as mutils

class ResnetBlock(torch.nn.Module):
    def __init__(self, n_C, norm=torch.nn.BatchNorm2d, activation=torch.nn.ReLU, use_bias=False):
        super().__init__()
        layers = []
        layers.append(torch.nn.Conv2d(in_channels=n_C, out_channels=n_C,
                                      kernel_size=3, stride=1, padding=0,
                                      bias=use_bias))
        if norm is not None:
            layers.append(norm(n_C))
        layers.append(activation())
        layers.append(torch.nn.Conv2d(in_channels=n_C, out_channels=n_C,
                                      kernel_size=3, stride=1, padding=0,
                                      bias=use_bias))
        if norm is not None:
            layers.append(norm(n_C))
        self.straight = torch.nn.Sequential(*layers)
        self.skip = torch.nn.ZeroPad2d(-2)
        self.block = lambda x: self.skip(x) + self.straight(x)

    def forward(self, x):
        return self.block(x)


class CycleResnetBlock(torch.nn.Module):
    def __init__(self, n_C, norm=torch.nn.BatchNorm2d,
                 activation=torch.nn.ReLU,
                 keep_prob=0.5,
                 use_bias=False):
        super().__init__()
        layers = []
        layers += [
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(in_channels=n_C, out_channels=n_C,
                            kernel_size=3, stride=1, padding=0,
                            bias=use_bias)
        ]
        if norm is not None:
            layers.append(norm(n_C))
        layers.append(activation())
        if keep_prob < 1:
            layers.append(torch.nn.Dropout2d(1 - keep_prob))
        layers += [
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(in_channels=n_C, out_channels=n_C,
                            kernel_size=3, stride=1, padding=0,
                            bias=use_bias)
        ]
        self.straight = torch.nn.Sequential(*layers)
        self.block = LambdaModule(lambda x: x + self.straight(x))

    def forward(self, x):
        return self.block(x)


MyLeakyReLU = partial(torch.nn.LeakyReLU, negative_slope=0.2)


class UnetBlock(torch.nn.Module):
    def __init__(self, in_channels,
                 left_filters, right_filters, lower=None,
                 left_norm=torch.nn.BatchNorm2d, right_norm=torch.nn.BatchNorm2d,
                 left_activation=torch.nn.ReLU, right_activation=MyLeakyReLU,
                 left_keep_prob=1, right_keep_prob=0.5
                 ):
        super().__init__()
        self.lower = lower
        self.right_filters = right_filters
        self.in_channels = in_channels
        self.block = None
        left_use_bias = mutils.requires_bias(left_norm)
        right_use_bias = mutils.requires_bias(right_norm)
        if self.lower is None:
            left = mutils.make_CDk(in_channels=in_channels, out_channels=left_filters,
                                   kernel_size=4, stride=2, padding=1,
                                   norm=left_norm, activation=left_activation,
                                   keep_prob=left_keep_prob, use_bias=left_use_bias)
            right = mutils.make_CDkT(in_channels=left_filters, out_channels=right_filters,
                                     kernel_size=4, stride=2, padding=1, output_padding=0,
                                     norm=right_norm, activation=right_activation,
                                     keep_prob=right_keep_prob, use_bias=right_use_bias)
            block = [left, right]
            self.block = torch.nn.Sequential(*block)
        else:
            assert left_filters == self.lower.in_channels, \
                'Left filters must match lower in channels'
            left = mutils.make_CDk(in_channels=in_channels, out_channels=left_filters,
                                   kernel_size=4, stride=2, padding=1,
                                   norm=left_norm, activation=left_activation,
                                   keep_prob=left_keep_prob, use_bias=left_use_bias)
            right = mutils.make_CDkT(in_channels=left_filters + self.lower.right_filters,
                                     out_channels=right_filters, kernel_size=4, stride=2,
                                     padding=1, output_padding=0,
                                     norm=right_norm, activation=right_activation,
                                     keep_prob=right_keep_prob, use_bias=right_use_bias)
            middle = LambdaModule(lambda x: torch.cat([x, self.lower(x)], 1))
            block = [left, middle, right]
            self.block = torch.nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class LambdaModule(torch.nn.Module):
    def __init__(self, lambd):
        super().__init__()
        import types
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)
