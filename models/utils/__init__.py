from inspect import signature
from functools import partial
import torch
from .torchviz_dot import *
from .image_pool import *


def make_CDk(in_channels, out_channels, kernel_size, padding,
             stride, activation=torch.nn.ReLU, norm=torch.nn.BatchNorm2d,
             keep_prob=1, use_bias=False):
    layers = [
        torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                        kernel_size=kernel_size, stride=stride, padding=padding,
                        bias=use_bias)
    ]
    if norm is not None:
        layers.append(norm(out_channels))
    if keep_prob < 1:
        layers.append(torch.nn.Dropout2d(1-keep_prob))
    layers.append(activation())
    return torch.nn.Sequential(*layers)


def make_CDkT(in_channels, out_channels, kernel_size, padding,
              stride, output_padding, activation=torch.nn.ReLU,
              norm=torch.nn.BatchNorm2d, keep_prob=1,
              use_bias=False):
    layers = [
        torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding,
                                 output_padding=output_padding, bias=use_bias)
    ]
    if norm is not None:
        layers.append(norm(out_channels))
    if keep_prob < 1:
        layers.append(torch.nn.Dropout2d(1-keep_prob))
    layers.append(activation())
    return torch.nn.Sequential(*layers)


def requires_bias(norm):
    if norm is None:
        return True
    base_class = torch.nn.modules.batchnorm._BatchNorm
    norm_class = norm
    if isinstance(norm, partial):
        norm_class = norm.func
    assert issubclass(norm_class, base_class), \
        '{} is not a batchnorm baseclass'.format(norm)
    sig = signature(norm)
    assert 'affine' in sig.parameters, 'Constructor has no affine parameter'
    return not sig.parameters['affine'].default


def get_norm_layer(layer_name):
    if layer_name == 'BatchNorm':
        return torch.nn.BatchNorm2d
    elif layer_name == 'InstanceNorm':
        return partial(torch.nn.InstanceNorm2d, affine=True,
                       track_running_stats=False)
    else:
        raise NotImplementedError(
            '{} not an implemented norm layer'.format(layer_name))


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - opt.decay_epoch) \
                / float(opt.n_epochs - opt.decay_epoch)
            return lr_l
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(
                    m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, gain)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('Initialized network {} with {}'.format(net._get_name(), init_type))
    net.apply(init_func)


def move_to_gpus(net, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(device=torch.device('cuda:{:d}'.format(gpu_ids[0])))
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)
    return net

def init_net(net, init_type='normal', init_gain=0.02):
    init_weights(net, init_type, gain=init_gain)
    return net


def get_named_layers(net):
    conv2d_idx = 0
    convT2d_idx = 0
    linear_idx = 0
    batchnorm2d_idx = 0
    instnorm2d_idx = 0
    named_layers = {}
    for mod in net.modules():
        if isinstance(mod, torch.nn.Conv2d):
            layer_name = 'Conv2d{}_{}-{}'.format(
                conv2d_idx, mod.in_channels, mod.out_channels
            )
            named_layers[layer_name] = mod
            conv2d_idx += 1
        elif isinstance(mod, torch.nn.ConvTranspose2d):
            layer_name = 'ConvT2d{}_{}-{}'.format(
                conv2d_idx, mod.in_channels, mod.out_channels
            )
            named_layers[layer_name] = mod
            convT2d_idx += 1
        elif isinstance(mod, torch.nn.BatchNorm2d):
            layer_name = 'BatchNorm2D{}_{}'.format(
                batchnorm2d_idx, mod.num_features)
            named_layers[layer_name] = mod
            batchnorm2d_idx += 1
        elif isinstance(mod, torch.nn.Linear):
            layer_name = 'Linear{}_{}-{}'.format(
                linear_idx, mod.in_features, mod.out_features
            )
            named_layers[layer_name] = mod
            linear_idx += 1
        elif isinstance(mod, torch.nn.InstanceNorm2d):
            layer_name = 'InstNorm2D{}_{}'.format(
                instnorm2d_idx, mod.num_features)
            named_layers[layer_name] = mod
            instnorm2d_idx += 1
    return named_layers
