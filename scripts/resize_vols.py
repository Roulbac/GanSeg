import os
import argparse
import SimpleITK as sitk
import numpy as np


def code2ors(code):
    """code2ors
        Turns a slice code into a cosine orientation matrix
        Uses ITK convention for xyz and ITK-SNAP convention for naming orientations
    :param code: slice orientation name
    """
    chars = len(code)
    assert chars <= 3, 'Can\'t get code of more than 3 chars'
    # RAI positive, LPS Neg, ITK-SNAP NAMING in ITK coords, letter indicates where the direction departs from
    dirs = [['L', 'P', 'S'],
            ['R', 'A', 'I']]
    ors = np.zeros((3, 3))
    for i, char in enumerate(code):
        vec = np.zeros(3)
        if char in dirs[0]:
            vec[dirs[0].index(char)] = -1
        elif char in dirs[1]:
            vec[dirs[1].index(char)] = 1
        else:
            raise ValueError('Invalid axis code char')
        ors[:, i] = vec
    if chars == 2:
        ors[:, 2] = np.cross(ors[:, 0], ors[:, 1])
        third_list = ors[:, 2].tolist()
        val = int([i for i in range(3) if third_list[i] != 0][0])
        if third_list[val] == 1:
            last_char = dirs[1][val]
        elif third_list[val] == -1:
            last_char = dirs[0][val]
        else:
            raise ValueError('Third direction can only be +- 1')
        code += last_char
    assert abs(np.linalg.det(ors) -
               1) < 0.01, 'Code not defining a right-handed orientation'
    return code, ors


def get_npornt(direction):
    if not all(map(lambda x: x in [1, 0, -1], direction)):
        e1, e2, e3 = np.array(direction[:3]), np.array(
            direction[3:6]), np.array(direction[6:])

        def get_max_projector(x):
            abs_x = np.abs(x)
            max_abs = abs_x.max()
            bool_arr = (abs_x == max_abs)
            return np.sign(x[bool_arr]) * (bool_arr)

        e1, e2, e3 = get_max_projector(
            e1), get_max_projector(e2), get_max_projector(e3)
        direction = tuple(e1) + tuple(e2) + tuple(e3)
    ornt = np.array(direction).reshape((3, 3))
    return ornt

def get_slice_dims(direction, slice_type):
    """get_slice_dims
        Get an image object and slice name (AX, COR or SAG), returns slice_dims containing
        information about how to slice the image
    :param direction: sitk image direction
    :param slice_type: slice name str
    """
    # Orientation values can only be 0, 1 and -1')):
    ornt = get_npornt(direction)
    slice_codes = {'SAG': 'AS', 'COR': 'RS', 'AX': 'RA'}
    _, slice_or = code2ors(slice_codes[slice_type])
    reorered_or = np.dot(ornt.T, slice_or)
    first_dim = np.where(reorered_or[:, 0] != 0)[0].item()
    first_dir = reorered_or[first_dim, 0]
    second_dim = np.where(reorered_or[:, 1] != 0)[0].item()
    second_dir = reorered_or[second_dim, 1]
    slice_dim = np.where(reorered_or[:, 2] != 0)[0].item()
    slice_dim = int(slice_dim)
    first_dim, first_dir, second_dim, second_dir = \
        int(first_dim), int(first_dir), int(second_dim), int(second_dir)
    slice_order = (first_dim, first_dir, second_dim, second_dir)
    slice_dims = slice_dim, slice_order
    return slice_dims

def get_new_origin(vol, direction, slice_dim, targ_physize):
    origin = list(vol.GetOrigin())
    size = vol.GetSize()
    spacing = vol.GetSpacing()
    dims = [dim for dim in range(3) if dim != slice_dim]
    for dim in dims:
        pt = [0, 0, 0]
        targ_voxsize = targ_physize // spacing[dim]
        pt[dim] = int(size[dim]//2 - targ_voxsize//2)
        pt = vol.TransformIndexToPhysicalPoint(tuple(pt))
        origin[dim] = pt[dim]
    return tuple(origin)


def resize_vol_slices(vol, slice_type, target_size, interp='bspline'):
    """resize_vol_slices
    Resizes the volume's slices specifically to target_size

    :param vol:
    :param slice_type:
    :param target_size:
    """
    direction = vol.GetDirection()
    slice_dims = get_slice_dims(direction, slice_type)
    slice_dim = slice_dims[0]
    interps = dict(bspline=sitk.sitkBSpline,
                   nearest=sitk.sitkNearestNeighbor,
                   linear=sitk.sitkLinear)
    new_size_dict = {0: lambda x, y: (x, y, y),
                     1: lambda x, y: (y, x, y),
                     2: lambda x, y: (y, y, x)}
    old_size = vol.GetSize()
    new_size = new_size_dict[slice_dim](old_size[slice_dim], target_size)
    old_spacing = vol.GetSpacing()
    old_physsize = tuple(map(
        lambda x: x[0]*x[1],
        zip(old_size, old_spacing))
    )
    targ_physize = max((old_physsize[:slice_dim] + old_physsize[slice_dim+1:]))
    direction = vol.GetDirection()
    new_spacing = tuple()
    for i in range(3):
        if i is not slice_dim:
            new_spacing += (targ_physize/target_size,)
        else:
            new_spacing += (old_spacing[i],)
    identity_tfm = sitk.Transform(3, sitk.sitkIdentity)
    minfil = sitk.MinimumMaximumImageFilter()
    minfil.Execute(vol)
    new_origin = get_new_origin(vol, direction, slice_dim, targ_physize)
    resampled = sitk.Resample(vol, new_size, identity_tfm,
                              interps[interp], new_origin,
                              new_spacing, direction,
                              minfil.GetMinimum())
    return resampled


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        required=True, help='Input Directory')
    parser.add_argument('--output', type=str,
                        required=True, help='Output Directory')
    parser.add_argument('--slice_type', type=str,
                        default='SAG', help='Slice dir to resize')
    parser.add_argument('--target_size', type=int,
                        default=256, help='Target size')
    parser.add_argument('--interp', type=str, default='bspline',
                        help='Resample interpolator')
    args = parser.parse_args()
    parent_dirpath = os.path.dirname(os.path.abspath(args.input))
    for root, _, files in os.walk(args.input):
        relpath = os.path.relpath(root, parent_dirpath)
        for f in files:
            try:
                img = sitk.ReadImage(os.path.join(root, f))
            except RuntimeError:
                continue
            try:
                resized = resize_vol_slices(img,
                                            slice_type=args.slice_type,
                                            target_size=args.target_size,
                                            interp=args.interp)
            except RuntimeError:
                print('Was not able to preprocess {}'.format(f))
                continue
            output_dir = os.path.join(args.output, relpath)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_path = os.path.join(output_dir, f)
            try:
                sitk.WriteImage(resized, output_path)
            except RuntimeError:
                print('Was not able to write image to {}'.format(output_path))
            print('Wrote image to {}'.format(output_path))
