def test_labvol_dataset_nolabel():
    import argparse
    from datasets.labvolslice_dataset import LabVolSliceDataset
    parser = argparse.ArgumentParser()
    opts = parser.parse_args([])
    opts.rootdir = './test_data/test_slicing'
    opts.slice_type = 'SAG'
    opts.mode = 'test'
    opts.no_labels = True
    ds = LabVolSliceDataset(opts)
    print(len(ds))
    print(ds[0].shape)


def test_vol_dataset():
    import SimpleITK as sitk
    import numpy as np
    from datasets.volslice_dataset import VolSliceDataset
    from datasets.utils import slices_to_vol
    rootdir = './test_data/test_slicing'
    slice_type = 'SAG'
    ds = VolSliceDataset(rootdir=rootdir, slice_type=slice_type, use_vols=True)
    print(len(ds))
    print(ds[0].shape)
    vol = sitk.ReadImage('./test_data/test_slicing/Img_01.nii.gz')
    direction, spacing, origin = vol.GetDirection(), vol.GetSpacing(), vol.GetOrigin()
    tfmd_vol = slices_to_vol(ds[0], direction, spacing, origin, 'SAG')
    print('Norm dist: ', np.linalg.norm(sitk.GetArrayFromImage(vol) - sitk.GetArrayFromImage(tfmd_vol)))

def test_segloss():
    import argparse
    import torch
    import models.losses as mlosses
    parser = argparse.ArgumentParser()
    opts = parser.parse_args([])
    opts.n_labels = 2
    opts.image_shape = '1, 256, 256'
    opts.disc_type = '64x64'
    opts.norm = 'InstanceNorm'
    opts.gpu_ids = ''
    opts.init_type = 'xavier'
    opts.lr = 0.0002
    opts.beta1 = 0.5
    opts.reshape_tensors = False
    for loss in ['ce', 'ogvanilla', 'vanilla', 'wgangp']:
        opts.loss = loss
        segloss = mlosses.SegLoss(opts)
        x = torch.Tensor(1, 1, 256, 256).uniform_()
        y = torch.Tensor(1, 1, 256, 256).long().random_(0, 2)
        y_pred = torch.Tensor(1, 2, 256, 256).random_(0, 2).requires_grad_()
        pooled = torch.Tensor(1, 3, 256, 256)
        pooled[:, 0, ...].uniform_()
        pooled[:, 1:, ...].random_(0, 2)
        print(loss)
        print(segloss.backward(y, y_pred, x, pooled))

def test_labvolslice_dataset():
    import argparse
    from datasets.labvolslice_dataset import LabVolSliceDataset
    parser = argparse.ArgumentParser()
    opts = parser.parse_args([])
    opts.rootdir = './data/DS0'
    opts.slice_type = 'SAG'
    opts.mode = 'train'
    opts.n_validation = 2
    opts.n_labels = 1
    ds = LabVolSliceDataset(opts)
    print(len(ds))
    print(ds[0][0].shape, ds[0][1].shape)
    t_set, v_set = ds.get_train_val(opts)
    print(len(t_set), len(v_set))

def test_tuplevolslice_dataset():
    import argparse
    from datasets.tuplevolslice_dataset import TupleVolSliceDataset
    parser = argparse.ArgumentParser()
    opts = parser.parse_args([])
    opts.rootdir = './data/PairedMRCTDataset'
    opts.is_unpaired = False
    opts.direction = 'XY'
    opts.slice_type = 'SAG'
    ds = TupleVolSliceDataset(opts)
    print(len(ds), ds[0][0].shape, ds[0][1].shape)


def test_volslice_dataset():
    from datasets.volslice_dataset import VolSliceDataset
    datasetX = VolSliceDataset('./data/PairedMRCTDataset/X', slice_type='SAG')
    datasetY = VolSliceDataset('./data/PairedMRCTDataset/Y', slice_type='SAG')
    print(len(datasetX))
    print(datasetX[0].shape)
    print(len(datasetY))
    print(datasetY[0].shape)


def test_resize_vol_slices():
    import SimpleITK as sitk
    from datasets.utils import resize_vol_slices, code2ors
    or_str = 'AIR'
    print(or_str)
    ax_dir = tuple(code2ors(or_str)[1].flatten())
    print(ax_dir)
    img = sitk.Image(100, 100, 100, sitk.sitkFloat32)
    img.SetDirection(ax_dir)
    print(img.GetSize())

    def voxtomm(si, sp):
        return (si[0]*sp[0], si[1]*sp[1], si[2]*sp[2])
    resized = resize_vol_slices(img, 'SAG', 20)
    size_mm = voxtomm(resized.GetSize(), resized.GetSpacing())
    print('SAG', resized.GetSize(), size_mm)
    resized = resize_vol_slices(img, 'COR', 20)
    size_mm = voxtomm(resized.GetSize(), resized.GetSpacing())
    print('COR', resized.GetSize(), size_mm)
    resized = resize_vol_slices(img, 'AX', 20)
    size_mm = voxtomm(resized.GetSize(), resized.GetSpacing())
    print('AX', resized.GetSize(), size_mm)

def test_wgangp():
    import torch
    from models import create_model, get_model_parsing_modifier
    from options.options_parser import OptionsParser
    import sys
    sys.argv = ['test_fun.py', '--model', 'cgan', '--dataset', 'base', 'train']
    parser = OptionsParser()
    model_name = parser.get_model_name()
    dataset_name = parser.get_dataset_name()
    print('Model name: {}'.format(model_name))
    print('Dataset name: {}'.format(dataset_name))

    model_parser_modifier = get_model_parsing_modifier(model_name)
    model_parser_modifier(parser, parser.is_train())
    opts, _ = parser.parse_options()
    opts.image_shape = '3, 256, 256'
    opts.model = 'cgan'
    opts.mode = 'train'
    opts_str = parser.make_opts_string(opts, verbose=False)
    print(opts_str)
    opts.gan_mode = 'wgangp_10'
    model = create_model(opts)
    x = torch.Tensor(4, 6, 256, 256).uniform_()
    y = torch.Tensor(4, 6, 256, 256).uniform_()
    model.set_input((x, y))
    gan_lossmodule = model.losses['GAN']
    net_D = model.nets['D']

    def zero_grad(net):
        for p in net.parameters():
            if p.requires_grad:
                p.grad = torch.Tensor(p.shape).zero_()
    zero_grad(net_D)
    params = list(net_D.parameters())
    means = [param.grad.mean().cpu().item() for param in params]
    mean = sum(means)/len(means)
    print('Before backprop: {}'.format(mean))
    gan_loss = gan_lossmodule.grad_penalty(x, y)
    gan_loss.backward()
    print('Grad loss: {}'.format(gan_loss.cpu().item()))
    params = list(net_D.parameters())
    means = [param.grad.mean().cpu().item() for param in params]
    mean = sum(means)/len(means)
    print('After backprop: {}'.format(mean))


if __name__ == '__main__':
    import sys
    globals()[sys.argv[1]]()
