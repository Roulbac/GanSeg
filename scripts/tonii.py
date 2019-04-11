import sys
import os
import SimpleITK as sitk

idx = 0
for f in os.scandir(sys.argv[1]):
    if f.is_file():
        try:
            img = sitk.ReadImage(f.path)
        except RuntimeError:
            continue
        sitk.WriteImage(img, os.path.join(sys.argv[1], '{}.nii.gz'.format(idx)))
        idx += 1
