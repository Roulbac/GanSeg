import os
import h5py
import json
import numpy as np
import SimpleITK as sitk
from torch.utils.data import Dataset
from datasets.utils import get_slice_dims, vol_to_slices, get_batch_slice


class VolSliceDataset(Dataset):

    def __init__(self, rootdir, slice_type, tfm=None, tfm_filter=None, fpath=None, use_vols=False):
        self.rootdir = rootdir
        self.slice_type = slice_type
        self.tfm = tfm
        self.tfm_filter = tfm_filter
        self.use_vols = use_vols
        if self.use_vols:
            self.init_vols(rootdir, slice_type)
        else:
            self.init_slices(rootdir, slice_type, fpath, tfm_filter)

    def init_vols(self, rootdir, slice_type):
        self.vols_dict, _, _ = VolSliceDataset.make_vols_dict(
            rootdir, slice_type)
        self.vol_paths = []
        for key, val in self.vols_dict.items():
            self.vol_paths += list(map(lambda x: os.path.join(key, x), val.keys()))
        self.n_vols = len(self.vol_paths)

    def init_slices(self, rootdir, slice_type, fpath, tfm_filter=None):
        if fpath is None:
            fpath = self.rootdir
        self.h5_p = '{}.hdf5'.format(fpath)
        self.js_p = '{}.json'.format(fpath)
        if not (os.path.exists(self.h5_p) and os.path.exists(self.js_p)):
            ds_data = VolSliceDataset.make_vols_dict(rootdir,
                                                     slice_type,
                                                     tfm_filter)
            self.vols_dict, self.n_slices, self.slshape = ds_data
            VolSliceDataset.init_files(
                self.rootdir, self.vols_dict, self.n_slices, self.slshape)
        else:
            with open(self.js_p, 'r') as f:
                self.vols_dict = json.load(f)
            print('Read {} volsdict file'.format(self.js_p))
            h5_f = h5py.File(self.h5_p, mode='r')
            h5_ds = h5_f['slices']
            self.n_slices, self.slshape = h5_ds.shape[0], h5_ds.shape[1:]
            h5_f.close()
            print('Read {} dataset file'.format(self.h5_p))

    def __getitem__(self, idx):
        """__getitem__
        Slices of shape NHWC

        :param idx:
        """
        if isinstance(idx, int):
            assert idx < len(self)
        if self.use_vols:
            path = os.path.join(self.rootdir, self.vol_paths[idx])
            sls = vol_to_slices(path, self.slice_type).copy()
            if self.tfm is not None:
                sls = self.tfm(sls)
            return sls
        else:
            if isinstance(idx, slice):
                assert idx.stop <= len(self)
            h5_f = h5py.File(self.h5_p, mode='r')
            sls = h5_f['slices'][idx, ...].copy()
            h5_f.close()
            if self.tfm is not None:
                sls = self.tfm(sls)
            return sls

    @staticmethod
    def init_files(rootdir, vols_dict, n_slices, slshape, fpath=None):
        if fpath is None:
            fpath = rootdir
        h5_p = '{}.hdf5'.format(fpath)
        VolSliceDataset.make_h5_file(
            rootdir, h5_p, vols_dict, n_slices, slshape
        )
        with open('{}.json'.format(fpath), 'w') as jfile:
            json.dump(vols_dict, jfile)
        print('Created {}{} volsdict file'.format(fpath, '.json'))

    @staticmethod
    def count_slices(path, slice_type, tfm_filter=None):
        reader = sitk.ImageFileReader()
        reader.SetFileName(path)
        reader.ReadImageInformation()
        direction = reader.GetDirection()
        size = reader.GetSize()

        def transpose(d):
            return d[slice(0, 9, 3)] + d[slice(1, 9, 3)] + d[slice(2, 9, 3)]
        sv = sitk.Version()
        if (sv.MajorVersion() <= 1 and sv.MinorVersion() <= 2):  # Bug fixed after 1.2.0
            direction = transpose(direction)
        slice_dims = get_slice_dims(direction, slice_type)
        slice_dim, slice_order = slice_dims
        if tfm_filter is not None:
            arr = sitk.GetArrayFromImage(reader.Execute())
            slicer = {0: lambda x: np.rollaxis(x, 2, 0),
                      1: lambda x: np.rollaxis(x, 1, 0),
                      2: lambda x: x}
            sliced = slicer[slice_dim](arr)
            slice_ids = list(filter(lambda x: tfm_filter(sliced[x]),
                                    range(len(sliced))))
        else:
            slice_ids = list(range(size[slice_dim]))
        n_slices = len(slice_ids)
        first_dim, second_dim = slice_order[0], slice_order[2]
        slice_shape = (size[first_dim], size[second_dim])
        return n_slices, slice_dims, slice_ids, slice_shape

    def __len__(self):
        if self.use_vols:
            return self.n_vols
        else:
            return self.n_slices

    @staticmethod
    def make_vols_dict(rootdir, slice_type, tfm_filter=None):
        total_slices = 0
        slices_shape = None
        vols_dict = {}
        for currdir, _, filelist in os.walk(rootdir):
            dirname = os.path.relpath(currdir, rootdir)
            for fname in filelist:
                fpath = os.path.join(os.path.abspath(currdir), fname)
                try:
                    fdata = VolSliceDataset.count_slices(
                        fpath, slice_type, tfm_filter)
                except RuntimeError:
                    continue
                n_slices, slice_dims, slice_ids, slice_shape = fdata
                if slices_shape is None:
                    slices_shape = slice_shape
                if slices_shape != slice_shape:
                    print('Slice shape mismatch for {}, expected {} found {}'.format(
                        fpath, slices_shape, slice_shape)
                    )
                    continue
                if dirname not in vols_dict:
                    vols_dict[dirname] = {}
                start = total_slices
                end = total_slices + n_slices
                vols_dict[dirname][fname] = (n_slices, slice_dims, slice_ids, start, end)
                total_slices += n_slices
        return vols_dict, total_slices, (slices_shape + (1,))  # HWC

    @staticmethod
    def read_vol(volpath):
        reader = sitk.ImageFileReader()
        reader.SetFileName(volpath)
        return reader.Execute()

    @staticmethod
    def make_h5_file(rootdir, h5_p, vols_dict, n_slices, slshape):
        h5_fp = h5_p
        h5_f = h5py.File(h5_fp, mode='w')
        ds_shape = (n_slices,) + slshape  # NHWC
        h5_ds = h5_f.create_dataset('slices', shape=ds_shape, dtype=np.float32)
        idx = 0
        for subdir, voldict in vols_dict.items():
            for volname, slice_data in voldict.items():
                n_slices, slice_dims, slice_ids, start, end = slice_data
                volpath = os.path.join(os.path.abspath(rootdir),
                                       subdir, volname)
                vol = VolSliceDataset.read_vol(volpath)
                np_slices = get_batch_slice(vol,
                                            slice_dims,
                                            slice_ids)
                h5_ds[start:end, ...] = np_slices
                idx += n_slices
        h5_f.close()
        print('Created {} dataset file'.format(h5_fp))
