import os
import numpy as np
import SimpleITK as sitk
import torch.nn as nn


def get_var_mask(tensor, ker_size=4, threshold=0.01):
    # Float tensor of shape CxHxW
    tensor = tensor.unsqueeze(0)
    # Float tensor of shape NxCxHxW
    avgs = nn.functional.avg_pool2d(tensor, ker_size)
    avgs = nn.functional.interpolate(
        avgs, size=tensor.shape[-2:], mode='nearest')
    diffs_sq = (tensor - avgs)**2
    variances = nn.functional.avg_pool2d(diffs_sq, ker_size)
    variances = nn.functional.interpolate(
        variances, size=tensor.shape[-2:], mode='nearest')
    mask = (variances > threshold).float().squeeze(0)
    return mask


def code2ors(code):
    """code2ors
        Turns a slice code into a cosine orientation matrix
        Uses ITK convention for xyz and ITK-SNAP convention for naming orientations
    :param code: slice orientation name
    """
    chars = len(code)
    assert chars <= 3, 'Can\'t get code of more than 3 chars'
    # RAI positive, LPS Neg, ITK-SNAP NAMING in ITK coords, letter indicates where the direction departs from
    # In ITK coords, x and y get a negative sign wrt NIFTII coords
    # RAI in ITK is LPI in niftii, LPS is RAS in niftii
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


def get_slice_dims(direction, slice_type):
    """get_slice_dims
        Get an image object and slice name (AX, COR or SAG), returns slice_dims containing
        information about how to slice the image
    :param direction: sitk image direction
    :param slice_type: slice name str
    """
    # Orientation values can only be 0, 1 and -1')):
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
    slice_codes = {'SAG': 'SA', 'COR': 'SR', 'AX': 'AR'}
    _, slice_or = code2ors(slice_codes[slice_type])
    # from itk from initial ornt to that of the slice code
    # now each row is the uvw vector from slice_code_ornt to current ornt
    # slice code ornts are chosen st z is the slice dim, x and y are at top left corner of image
    # vox ornt -> RAI -> slice_code ornt
    reordered_or = np.dot(slice_or.T, ornt)  # Change of coords from direction -> ASL, RSA or RAI
    first_dim = np.where(reordered_or[0, :] != 0)[0].item()
    first_dir = reordered_or[0, first_dim]
    second_dim = np.where(reordered_or[1, :] != 0)[0].item()
    second_dir = reordered_or[1, second_dim]
    slice_dim = np.where(reordered_or[2, :] != 0)[0].item()
    slice_dim = int(slice_dim)
    first_dim, first_dir, second_dim, second_dir = \
        int(first_dim), int(first_dir), int(second_dim), int(second_dir)
    slice_order = (first_dim, first_dir, second_dim, second_dir)
    slice_dims = slice_dim, slice_order
    return slice_dims


def slice_view(arr_view, slice_dims, idx):
    """slice_view
        Slices an sitk array view (order z, y, x) along slice_dims at slice idx
    :param arr_view: sitk array view
    :param slice_dims: slice_dims tuple from get_slice_dims
    :param idx: slice id
    """
    slice_dim, slice_order = slice_dims
    first_dim, first_dir, second_dim, second_dir = slice_order
    itk2np = {0: 2, 1: 1, 2: 0}
    reordered = np.transpose(
        arr_view,
        tuple(map(lambda x: itk2np[x], [slice_dim, first_dim, second_dim]))
    )
    return reordered[idx, ::first_dir, ::second_dir]


def batch_slice_view(arr_view, slice_dims, slicer):
    """batch_slice_view
    Slices a view in batches where idx is a slice object

    :param arr_view:
    :param slice_dims:
    :param idx:
    """
    assert isinstance(slicer, (slice, list))
    batch = slice_view(arr_view, slice_dims, slicer)
    return batch[:, :, :, np.newaxis]  # NHWC


def get_batch_slice(vol, slice_dims, slicer):
    arr = sitk.GetArrayFromImage(vol)
    return batch_slice_view(arr, slice_dims, slicer)


def vol_to_slices(path, slice_type):
    vol = sitk.ReadImage(path)
    slice_dims = get_slice_dims(vol.GetDirection(), slice_type)
    slices = get_batch_slice(vol, slice_dims, slice(None))
    return slices

def slices_to_itknparr(slices, direction, slice_type, functor):
    slice_dims = get_slice_dims(direction, slice_type)
    slice_dim, slice_order = slice_dims
    first_dim, first_dir, second_dim, second_dir = slice_order
    slices = functor(slices)[:, ::first_dir, ::second_dir]
    dims = [slice_dim, first_dim, second_dim]
    z, y, x = dims.index(2), dims.index(1), dims.index(0)
    return np.transpose(slices, axes=(z, y, x))

def slices_to_vol(slices, direction, spacing, origin, slice_type, functor=lambda x: x.squeeze(3)):
    # slices NCHW tensor, processed should be NHW numpy
    processed_slices = slices_to_itknparr(slices, direction, slice_type, functor)
    processed_vol = sitk.GetImageFromArray(processed_slices)
    processed_vol.SetDirection(direction)
    processed_vol.SetOrigin(origin)
    processed_vol.SetSpacing(spacing)
    return processed_vol

def resample_mris(patient_list, ct_dir, mri_dir, tfm_dir, invert_tfm=True):
    head, tail = os.path.split(mri_dir)
    if tail == '':
        head, tail = os.path.split(head)
    resampled_mri_dir = os.path.join(head,
                                     tail + '_resampled')
    if not os.path.exists(resampled_mri_dir):
        os.makedirs(resampled_mri_dir)
    for patient in patient_list:
        patient_id = '{:02}'.format(int(patient))
        mri_path = os.path.join(
            mri_dir, '{}.nii.gz'.format(patient_id))
        ct_path = os.path.join(
            ct_dir, '{}.nii.gz'.format(patient_id))
        tfm_path = os.path.join(
            tfm_dir, 'CT2MR{}'.format(patient_id), 'Transform.tfm')
        resampled_path = resample_volume_wpath(
            mri_path, ct_path, tfm_path, invert_tfm)
        os.rename(
            resampled_path,
            os.path.join(resampled_mri_dir, '{}.nii.gz'.format(patient_id)))
    return resampled_mri_dir


def resample_volume(source, target, tfm, invert_tfm=True, interpolator=sitk.sitkBSpline):
    size = target.GetSize()
    spacing = target.GetSpacing()
    origin = target.GetOrigin()
    direction = target.GetDirection()
    used_tfm = tfm
    if invert_tfm:
        used_tfm = tfm.GetInverse()
    resampled_source = sitk.Resample(source, size, used_tfm,
                                     interpolator, origin,
                                     spacing, direction)
    return resampled_source


def resize_vol_slices(vol, slice_type, target_size):
    """resize_vol_slices
    Resizes the volume's slices specifically to target_size

    :param vol:
    :param slice_type:
    :param target_size:
    """
    direction = vol.GetDirection()
    slice_dims = get_slice_dims(direction, slice_type)
    slice_dim = slice_dims[0]
    new_size_dict = {0: lambda x, y: (x, y, y),
                     1: lambda x, y: (y, x, y),
                     2: lambda x, y: (y, y, x)}
    old_size = vol.GetSize()
    new_size = new_size_dict[slice_dim](old_size[slice_dim], target_size)
    origin = vol.GetOrigin()
    old_spacing = vol.GetSpacing()
    new_spacing = tuple()
    direction = vol.GetDirection()
    for i in range(3):
        if i is not slice_dim:
            scaled_spacing = old_spacing[i]*old_size[i]/target_size
            new_spacing += (scaled_spacing,)
        else:
            new_spacing += (old_spacing[i],)
    identity_tfm = sitk.Transform(3, sitk.sitkIdentity)
    resampled = sitk.Resample(vol, new_size, identity_tfm,
                              sitk.sitkBSpline, origin,
                              new_spacing, direction)
    return resampled


def series2nii(dirpath, output_dir):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    description_tag = '0008|103e'
    reader = sitk.ImageSeriesReader()
    for root, _, _ in os.walk(dirpath):
        uids = reader.GetGDCMSeriesIDs(root)
        if len(uids) == 0:
            continue
        for idx, uid in enumerate(uids):
            names = reader.GetGDCMSeriesFileNames(root, uid)
            reader.SetFileNames(names)
            reader.MetaDataDictionaryArrayUpdateOn()
            reader.LoadPrivateTagsOn()
            try:
                img = reader.Execute()
            except RuntimeError:
                print('{} names could not be read'.format(names))
                print('Continuing to next UID')
                continue
            description = reader.GetMetaData(
                0, description_tag).replace(' ', '_')
            writer = sitk.ImageFileWriter()
            output_path = os.path.join(
                os.path.abspath(output_dir),
                root.split(
                    '/{}/'.format(
                        os.path.basename(os.path.abspath(dirpath))
                    )
                )[1]
            )
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            writer.SetFileName(
                os.path.join(output_path, '{}_{}.nii.gz'.format(
                    idx, description))
            )
            writer.Execute(img)


def resample_volume_wpath(source_path, target_path, tfm_path, invert_tfm=True):
    tfm = sitk.ReadTransform(tfm_path)
    source = sitk.ReadImage(source_path)
    target = sitk.ReadImage(target_path)
    resampled_source = resample_volume(
        source, target, tfm, invert_tfm=invert_tfm)
    source_dir = os.path.dirname(source_path)
    source_name = os.path.basename(source_path).split('.')[0]
    resampled_source_path = os.path.join(
        source_dir, '{}_resampled.nii.gz'.format(source_name))

    sitk.WriteImage(resampled_source, resampled_source_path)
    return resampled_source_path
