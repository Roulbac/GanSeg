import os
import argparse
import SimpleITK as sitk


def get_3D_img(fpath):
    img = sitk.ReadImage(fpath)
    if len(img.GetSize()) == 3:
        return img
    arr = sitk.GetArrayFromImage(img)
    img_dir = img.GetDirection()
    new_img_dir = img_dir[0:3] + img_dir[4:7] + img_dir[8:11]
    new_img = sitk.GetImageFromArray(arr[0])
    new_img.SetDirection(new_img_dir)
    new_img.SetOrigin(img.GetOrigin()[:-1])
    new_img.SetSpacing(img.GetSpacing()[:-1])
    return new_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        required=True, help='Input Directory')
    parser.add_argument('--output', type=str,
                        required=True, help='Output Directory')
    args = parser.parse_args()
    parent_dirpath = os.path.dirname(os.path.abspath(args.input))
    for root, _, files in os.walk(args.input):
        relpath = os.path.relpath(root, parent_dirpath)
        for f in files:
            try:
                img = get_3D_img(os.path.join(root, f))
            except RuntimeError:
                continue
            output_dir = os.path.join(args.output, relpath)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_path = os.path.join(output_dir, f)
            try:
                sitk.WriteImage(img, output_path)
            except RuntimeError:
                print('Was not able to write image to {}'.format(output_path))
            print('Wrote image to {}'.format(output_path))
