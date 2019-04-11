import re
import torch
import models.modules as mmodules
import models.utils as mutils


class CycleResnetGenerator(torch.nn.Module):
    def __init__(self, in_channels=3, out_channels=None, ngf=64, n_down=2,
                 n_blocks=9, norm=torch.nn.InstanceNorm2d, keep_prob=0.5,
                 activation=torch.nn.ReLU, out_activation=torch.nn.Tanh):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        use_bias = mutils.requires_bias(norm)
        layers = []
        layers += [
            torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(in_channels=in_channels,
                            out_channels=ngf, kernel_size=7,
                            stride=1, padding=0, bias=use_bias),
            norm(ngf),
            activation()
        ]
        for n in range(n_down):
            layers += [
                torch.nn.Conv2d(in_channels=ngf*(2**n),
                                out_channels=ngf*(2**(n+1)),
                                kernel_size=3, stride=2,
                                padding=1, bias=use_bias),
                norm(ngf*(2**(n+1))),
                activation()
            ]
        for n in range(n_blocks):
            layers.append(
                mmodules.CycleResnetBlock(
                    n_C=ngf*(2**n_down), norm=norm,
                    activation=activation, keep_prob=keep_prob,
                    use_bias=use_bias)
            )
        n_up = n_down
        for n in reversed(range(n_up)):
            layers += [
                torch.nn.ConvTranspose2d(in_channels=ngf*(2**(n+1)),
                                         out_channels=ngf*(2**n),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                norm(ngf*(2**n)),
                activation()
            ]
        layers += [
            torch.nn.ReflectionPad2d(3),
            torch.nn.Conv2d(in_channels=ngf,
                            out_channels=out_channels, kernel_size=7,
                            stride=1, padding=0, bias=use_bias),
            out_activation()
        ]
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResnetGenerator(torch.nn.Module):
    """ResnetGenerator
    Note: Only use on even-sized tensors!

    """

    def __init__(self, in_channels=3, out_channels=None,
                 ngf=32, n_down=2,
                 n_blocks=5, norm=torch.nn.BatchNorm2d,
                 activation=torch.nn.ReLU,
                 out_activation=torch.nn.Tanh
                 ):
        """__init__

        :param in_channels:
        :param ngf: number of generator features, also of first conv
        :param n_down: downsamples by factor of 2
        :param n_blocks: resnet blocks
        :param norm: norm layer
        :param activation: activation function
        """
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        use_bias = mutils.requires_bias(norm)
        pad = n_down*4*n_blocks  # out_shape is n + ( pad/n_down - 4*n_blocks)
        layers = []
        layers.append(torch.nn.ReflectionPad2d(pad))
        layers.append(
            mutils.make_CDk(in_channels=in_channels, out_channels=ngf,
                            kernel_size=9, stride=1, padding=4,
                            norm=norm, activation=activation,
                            use_bias=use_bias)
        )
        for i in range(n_down):
            layers.append(
                mutils.make_CDk(in_channels=ngf*(2**i), out_channels=ngf*(2**(i+1)),
                                kernel_size=3, stride=2, padding=1,
                                norm=norm, activation=activation,
                                use_bias=use_bias)
            )
        resblock_n_C = ngf*(2**n_down)
        for i in range(n_blocks):
            layers.append(
                mmodules.ResnetBlock(n_C=resblock_n_C, norm=norm,
                                     activation=activation, use_bias=use_bias)
            )
        for i in reversed(range(n_down)):
            layers.append(
                mutils.make_CDkT(in_channels=ngf*(2**(i+1)), out_channels=ngf*(2**i),
                                 kernel_size=3, stride=2, padding=1,
                                 norm=norm, activation=activation,
                                 output_padding=1, use_bias=use_bias)
            )
        layers.append(
            mutils.make_CDk(in_channels=ngf, out_channels=out_channels,
                            kernel_size=9, stride=1, padding=4,
                            norm=norm, activation=torch.nn.Tanh,
                            use_bias=use_bias)
        )
        layers.append(out_activation())
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class UnetGenerator(torch.nn.Module):

    def __init__(self, in_channels, out_channels=None,
                 ngf=64, n_blocks=8, cap=3,
                 norm=torch.nn.BatchNorm2d, keep_prob=0.5,
                 out_activation=torch.nn.Tanh
                 ):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        unet_block = None
        bottleneck_filters = ngf*2**(cap)
        for n in reversed(range(1, n_blocks)):
            filters = min(bottleneck_filters, ngf*(2**(n)))
            n_C = min(bottleneck_filters, ngf*(2**(n-1)))
            right_keep_prob = keep_prob if n >= n_blocks - cap \
                else 1
            unet_block = mmodules.UnetBlock(in_channels=n_C, left_filters=filters,
                                            right_filters=n_C, lower=unet_block,
                                            left_norm=norm, right_norm=norm,
                                            left_keep_prob=1, right_keep_prob=right_keep_prob
                                            )
        unet_block = mmodules.UnetBlock(in_channels=in_channels, left_filters=ngf,
                                        right_filters=ngf, lower=unet_block,
                                        right_norm=norm, left_norm=None,
                                        right_keep_prob=1, left_keep_prob=1
                                        )
        net = [
            unet_block,
            torch.nn.Conv2d(in_channels=ngf, out_channels=out_channels,
                            kernel_size=1, stride=1, padding=0, bias=True),
            out_activation()
        ]
        self.net = torch.nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


class PatchDiscriminator(torch.nn.Module):
    """PatchDiscriminator
    Patch discriminator
    Sliding window of size 4*2^{n_conv} with stride of 2^{n_conv}
    NOTE: 70x70 is in fact 64x64 with stride 16
          16x16 is 16x16 though with stride 4
          1x1 is what it is
          286x286 is in fact 256x256 with stride 64
    We keep the naming of the original author P.Isola
    """

    _ConfigsDict = {
        '64x64':   {'init_n_C': 64, 'n_conv': 4,
                    'maxpow_n_C': 3, 'no_norm_ids': [0],
                    'kernel_size': 4,
                    'stride':     2, 'padding': 1, 'activation': mmodules.MyLeakyReLU
                    },
        '16x16':   {'init_n_C': 64, 'n_conv': 2,
                    'maxpow_n_C': 1, 'no_norm_ids': [],
                    'kernel_size': 4,
                    'stride':     2, 'padding': 1, 'activation': mmodules.MyLeakyReLU
                    },
        '1x1':     {'init_n_C': 64, 'n_conv': 2,
                    'maxpow_n_C': 1, 'no_norm_ids': [],
                    'kernel_size': 1,
                    'stride':     1, 'padding': 0, 'activation': mmodules.MyLeakyReLU
                    },
        '256x256': {'init_n_C': 64, 'n_conv': 6,
                    'maxpow_n_C': 3, 'no_norm_ids': [],
                    'kernel_size': 4,
                    'stride':     2, 'padding': 1, 'activation': mmodules.MyLeakyReLU
                    },
    }

    def __init__(self, in_channels=3, init_n_C=64,
                 n_conv=4, maxpow_n_C=3, no_norm_ids=[0],
                 norm=torch.nn.BatchNorm2d, kernel_size=4, stride=2,
                 padding=1, activation=mmodules.MyLeakyReLU
                 ):
        super().__init__()
        layers = []
        in_n_C = in_channels
        out_n_C = init_n_C
        max_out_n_C = init_n_C*2**maxpow_n_C
        for n in range(n_conv):
            norm_layer = None if n in no_norm_ids else norm
            use_bias = mutils.requires_bias(norm_layer)
            layer = mutils.make_CDk(in_channels=in_n_C, out_channels=out_n_C,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, activation=mmodules.MyLeakyReLU,
                                    use_bias=use_bias, norm=norm_layer,
                                    keep_prob=1
                                    )
            layers.append(layer)
            in_n_C = out_n_C
            out_n_C = min(2*out_n_C, max_out_n_C)
        layers.append(
            torch.nn.Conv2d(in_channels=out_n_C, out_channels=1,
                            kernel_size=kernel_size, stride=1,
                            padding=0, bias=True
                            )
        )
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ------------------------- Functions for manipulating nets -------------------


def make_generator(gen_type, image_shape,
                   norm_name, out_activation=torch.nn.Tanh,
                   out_channels=None):
    norm = mutils.get_norm_layer(norm_name)
    in_channels = image_shape[0]
    if re.search('^resnet_[0-9]+$', gen_type) is not None:
        n_blocks = int(gen_type.split('_')[1])
        net = ResnetGenerator(in_channels=in_channels,
                              norm=norm, n_blocks=n_blocks,
                              out_activation=out_activation,
                              out_channels=out_channels)
        return net
    elif re.search('^unet_[0-9]+$', gen_type) is not None:
        unet_dim = int(gen_type.split('_')[1])
        assert image_shape[1] == image_shape[2] and \
            unet_dim <= image_shape[1]
        dim2blocks = {128: 7, 256: 8, 512: 9}
        n_blocks = dim2blocks[unet_dim]
        net = UnetGenerator(in_channels=in_channels,
                            cap=3, n_blocks=n_blocks, norm=norm, out_channels=out_channels, out_activation=out_activation)
        return net
    elif re.search('^cycleresnet_[0-9]+$', gen_type) is not None:
        n_blocks = int(gen_type.split('_')[1])
        net = CycleResnetGenerator(in_channels=in_channels,
                                   norm=norm, n_blocks=n_blocks, out_channels=out_channels, out_activation=out_activation)
        return net
    else:
        raise NotImplementedError(
            '{} generator not implemented'.format(gen_type))


def make_discriminator(in_channels, disc_type, norm_name):
    norm = mutils.get_norm_layer(norm_name)
    params_dict = PatchDiscriminator._ConfigsDict[disc_type]
    params_dict['norm'] = norm
    params_dict['in_channels'] = in_channels
    return PatchDiscriminator(**params_dict)
