import os
import SimpleITK as sitk

for root, _, fnames in os.walk('.'):
    for f in fnames:
        if f.endswith('mhd'):
            fpath = os.path.abspath(os.path.join(root, f))
            img = sitk.ReadImage(fpath)
            out_name = os.path.splitext(fpath)[0] + '.nii.gz'
            sitk.WriteImage(img, out_name)


