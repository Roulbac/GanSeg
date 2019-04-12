import os
import SimpleITK as sitk
from torch.utils.data import Dataset
from datasets.transforms import ToTensor
from datasets.volslice_dataset import VolSliceDataset
from datasets.utils import slices_to_vol


class LabVolSliceDataset(Dataset):

    @staticmethod
    def modify_commandline_options(options_parser, is_train=True):
        parser = options_parser.parser
        parser.add_argument('--rootdir', type=str, required=True,
                            help='Data root directory')
        parser.add_argument('--image_shape', type=str, required=True,
                            help='Image shape')
        parser.add_argument('--slice_type', type=str, default='SAG',
                            choices=['SAG', 'AX', 'COR'],
                            help='How to slice the volumes')
        parser.add_argument('--n_labels', type=int, required=True,
                            help='Number of target labels')
        if is_train:
            parser.add_argument('--n_validation', type=int, default=1,
                                help='Number validation samples')
        else:
            # Test options
            parser.add_argument('--no_labels', action='store_true',
                                help='Test volumes without labels')
        return options_parser

    def __init__(self, opts, tfm=ToTensor(y_long=True)):
        self.rootdir = os.path.abspath(opts.rootdir)
        self.slice_type = opts.slice_type
        self.tfm = tfm
        self.is_train = (opts.mode == 'train')
        self.no_labels = opts.no_labels if hasattr(
            opts, 'no_labels') else False
        self.start_idx = 0
        assert self.slice_type in ['SAG', 'COR', 'AX']
        if self.is_train:
            self.init_train()
        else:
            self.init_test()

    def init_test(self):
        rootdir_img = os.path.join(self.rootdir, 'Images')
        self.dataset_img = VolSliceDataset(
            rootdir_img, self.slice_type, use_vols=True)
        if not self.no_labels:
            rootdir_lab = os.path.join(self.rootdir, 'Labels')
            self.dataset_lab = VolSliceDataset(rootdir_lab, self.slice_type, use_vols=True)
            assert set(self.dataset_lab.vol_paths) == set(
                self.dataset_img.vol_paths)
            self.dataset_lab.vol_paths = self.dataset_img.vol_paths
        self.n_vols = self.dataset_img.n_vols

    def init_train(self):
        rootdirX = os.path.join(self.rootdir, 'Images')
        rootdirY = os.path.join(self.rootdir, 'Labels')
        if os.path.exists(rootdirX + '.json') and os.path.exists(rootdirX + '.hdf5') and \
                os.path.exists(rootdirY + '.json') and os.path.exists(rootdirY + '.hdf5'):
            dsX = VolSliceDataset(
                rootdirX, slice_type=self.slice_type)
            dsY = VolSliceDataset(
                rootdirY, slice_type=self.slice_type)
            assert dsX.vols_dict == dsY.vols_dict
            assert dsX.n_slices == dsY.n_slices
            assert dsX.slshape == dsY.slshape
            self.n_slices = dsX.n_slices
            self.slshape = dsX.slshape
            self.dataset_img, self.dataset_lab = dsX, dsY
            self.vols_dict = dsX.vols_dict
        else:
            vols_dictX, _, slshapeX = VolSliceDataset.make_vols_dict(
                rootdirX, self.slice_type)
            vols_dictY, _, slshapeY = VolSliceDataset.make_vols_dict(
                rootdirY, self.slice_type)
            assert slshapeY == slshapeX, 'Dataset do not have same slice shapes'
            vols_dict = {}
            slshape = slshapeX
            n_slices = 0
            for dirname, fnamesdict in vols_dictX.items():
                vols_dict[dirname] = {}
                for fname, fdata in fnamesdict.items():
                    n_slicesX, slice_dimsX, slice_idsX, _, _ = vols_dictX[dirname][fname]
                    n_slicesY, slice_dimsY, slice_idsY, _, _ = vols_dictY[dirname][fname]
                    if slice_dimsX != slice_dimsY:
                        print('Skipping slices for volume {} not oriented alike'.format(
                            fname)
                        )
                        continue
                    slice_ids_intersec = list(
                        set(slice_idsX) & set(slice_idsY))
                    start, end = n_slices, n_slices + len(slice_ids_intersec)
                    vols_dict[dirname][fname] = (
                        len(slice_ids_intersec), slice_dimsX,
                        slice_ids_intersec, start, end)
                    n_slices += len(slice_ids_intersec)
            self.n_slices = n_slices
            self.slshape = slshape
            VolSliceDataset.init_files(
                rootdirX, vols_dict, n_slices,
                slshape)
            VolSliceDataset.init_files(
                rootdirY, vols_dict, n_slices,
                slshape)
            self.dataset_img = VolSliceDataset(
                rootdirX, slice_type=self.slice_type)
            self.dataset_lab = VolSliceDataset(
                rootdirY, slice_type=self.slice_type)
            self.vols_dict = vols_dict

    def get_train_val(self, opts):
        n_vols = opts.n_validation
        ids_list = []
        for key, val in self.vols_dict.items():
            map_src = zip(val.keys(), val.values())
            ids_list += list(
                map(lambda x: (os.path.join(
                    key, x[0]), (x[1][3], x[1][4])), map_src)
            )
        total_vols = len(ids_list)
        assert n_vols < total_vols
        # Get last n volumes
        ids_list.sort(key=lambda x: x[1][0])
        val_ids = ids_list[-n_vols:]
        val_arrays = list(map(lambda x: self[x[1][0]:x[1][1]], val_ids))
        self.n_slices = val_ids[0][1][0]
        return self, val_arrays

    def __len__(self):
        if hasattr(self, 'n_slices'):
            return self.n_slices
        else:
            return self.n_vols

    def __getitem__(self, idx):
        if isinstance(idx, int):
            assert idx < len(self)
        elif isinstance(idx, slice):
            assert idx.stop <= len(self)
        if self.no_labels:
            sample = self.dataset_img[idx]
        else:
            sample = (self.dataset_img[idx],
                      self.dataset_lab[idx])
        if self.tfm is not None:
            return self.tfm(sample)
        else:
            return sample

    def save_slices_as_vol(self, idx, results_dir, functor):
        src_path = os.path.join(self.dataset_img.rootdir,
                                self.dataset_img.vol_paths[idx])
        vol = sitk.ReadImage(src_path)
        direction, spacing, origin = vol.GetDirection(), vol.GetSpacing(), vol.GetOrigin()
        del vol
        if self.no_labels:
            slices, label = self[idx], None
        else:
            slices, label = self[idx]
        processed_vol = slices_to_vol(slices, direction,
                                      spacing, origin,
                                      self.slice_type, functor)
        relpath = os.path.relpath(src_path, os.path.dirname(self.rootdir))
        out_path = os.path.join(results_dir, relpath)
        if not os.path.exists(os.path.dirname(out_path)):
            os.makedirs(os.path.dirname(out_path))
        sitk.WriteImage(processed_vol, out_path)
