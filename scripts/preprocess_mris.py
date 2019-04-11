import os
import argparse
import SimpleITK as sitk

def norm_negone_one(img):
    shifscaler = sitk.ShiftScaleImageFilter()
    minmaxfilter = sitk.MinimumMaximumImageFilter()
    minmaxfilter.Execute(img)
    immin = minmaxfilter.GetMinimum()
    immax = minmaxfilter.GetMaximum()
    shifscaler.SetShift(-(immax + immin)/2)
    shifscaler.SetScale(1/((immax - immin)/2))
    return shifscaler.Execute(img)

def preprocess_mri(img, negone_one=True):
    caster = sitk.CastImageFilter()
    caster.SetOutputPixelType(sitk.sitkFloat32)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    fotsu = sitk.OtsuThresholdImageFilter()
    fotsu.SetInsideValue(1)
    fotsu.SetOutsideValue(0)
    fotsu.SetNumberOfHistogramBins(200)
    normalizer = sitk.NormalizeImageFilter()
    mask = fotsu.Execute(img)
    casted = caster.Execute(img)
    corrected = corrector.Execute(casted, mask)
    if negone_one:
        return norm_negone_one(normalizer.Execute(corrected))
    else:
        return normalizer.Execute(corrected)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str,
                        required=True, help='Input Directory')
    parser.add_argument('--output', type=str,
                        required=True, help='Output Directory')
    parser.add_argument('--no_negoneone', action='store_true',
                        help='Don\'t normalize to [-1, 1]')
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
                preprocessd = preprocess_mri(img, not args.no_negoneone)
            except RuntimeError:
                print('Was not able to preprocess {}'.format(f))
                continue
            output_dir = os.path.join(args.output, relpath)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            output_path = os.path.join(output_dir, f)
            try:
                sitk.WriteImage(preprocessd, output_path)
            except RuntimeError:
                print('Was not able to write image to {}'.format(output_path))
            print('Wrote image to {}'.format(output_path))
