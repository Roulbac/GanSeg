import torch
import numpy as np
from datasets.utils import get_var_mask


class ToTensor(object):
    """ToTensor
    Transforms to tensors a tuple of two np arrays
    """

    def __init__(self, is_hwc=True, y_long=False):
        self.is_hwc = is_hwc
        self.y_long = y_long

    def to_tensor(self, x):
        if self.is_hwc:
            if len(x.shape) == 4:
                x = np.transpose(x, (0, 3, 1, 2))
            elif len(x.shape) == 3:
                x = np.transpose(x, (2, 0, 1))
            else:
                raise ValueError()
        return torch.Tensor(x)

    def __call__(self, sample):
        if isinstance(sample, tuple):
            assert len(sample) == 2
            x, y = sample
            x, y = self.to_tensor(x), self.to_tensor(y)
            if self.y_long:
                y = y.long()
            return x, y
        else:
            return self.to_tensor(sample)


class FilterBackground(object):
    def __init__(self, ker_size=4, threshold=0.1):
        self.ker_size = ker_size
        self.threshold = threshold

    def __call__(self, x):
        # stds = self.np_avg_pool2d(x, self.ker_size)
        # mask = (stds > self.threshold)
        if isinstance(x, np.ndarray):
            minim, mean, maxim = np.min(x), np.mean(x), np.max(x)
        elif isinstance(x, torch.Tensor):
            minim, mean, maxim = x.min().item(), x.median().item(), x.max().item()
        if abs((mean-minim)/((maxim-minim + 1e-9))) > self.threshold:
            return True
        else:
            return False


class MaskBackground(object):
    def __init__(self, ker_size=4, threshold=1e-7):
        self.ker_size = ker_size
        self.threshold = threshold

    def __call__(self, sample):
        x_t, y_t = sample
        mask_x = get_var_mask(x_t, self.ker_size, self.threshold)
        mask_y = get_var_mask(y_t, self.ker_size, self.threshold)
        mask = mask_x * mask_y
        inv_mask = (1 - mask).abs()
        x_t = x_t*mask + x_t.min()*inv_mask
        y_t = y_t*mask + y_t.min()*inv_mask
        return x_t, y_t


class NormToTanh(object):
    def __call__(self, x):
        minim, _ = torch.min(x, dim=0)
        maxim, _ = torch.max(x, dim=0)
        offset = -(maxim - minim)/2
        scale = 2/(maxim + minim)
        return (x + offset)*scale


class Rescale(object):
    def __init__(self, output_size=(512, 512), interp='bilinear'):
        self.h = int(output_size[0])
        self.w = int(output_size[1])
        self.interp = interp

    def __call__(self, sample):
        x_t, y_t = sample
        if x_t.shape[-2:] == (self.h, self.w):
            return sample
        x_t_resized = torch.nn.functional.interpolate(x_t, size=(
            self.h, self.w), mode=self.interp, align_corners=False)
        y_t_resized = torch.nn.functional.interpolate(y_t, size=(
            self.h, self.w), mode=self.interp, align_corners=False)
        return x_t_resized, y_t_resized
