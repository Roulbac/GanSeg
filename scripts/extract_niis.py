import os
import SimpleITK as sitk
import argparse


def series2nii(dirpath, output_dir):
    """series2nii

    :param dirpath:
    :param output_dir:
    """
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    description_tag = '0008|103e'
    reader = sitk.ImageSeriesReader()
    parent_dir = os.path.dirname(os.path.abspath(dirpath))
    for root, _, _ in os.walk(os.path.abspath(dirpath)):
        uids = reader.GetGDCMSeriesIDs(root)
        if len(uids) == 0:
            continue
        for idx, uid in enumerate(uids):
            names = reader.GetGDCMSeriesFileNames(root, uid)
            reader.SetFileNames(names)
            reader.MetaDataDictionaryArrayUpdateOn()
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
                os.path.relpath(root, parent_dir)
            )
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            writer.SetFileName(
                os.path.join(output_path, '{}_{}.nii.gz'.format(
                    idx, description))
            )
            print('Saving {}'.format(writer.GetFileName()))
            writer.Execute(img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirpath', type=str, required=True,
                        help='Directory containing files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output Directory')
    args = parser.parse_args()

    series2nii(args.dirpath, args.output_dir)
