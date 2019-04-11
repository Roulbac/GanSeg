import os
import argparse
import zipfile

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dirpath', type=str, required=True,
                        help='Directory containing files')
    args = parser.parse_args()
    for elem in os.scandir(args.dirpath):
        if elem.is_file() and elem.name.endswith('.zip'):
            with zipfile.ZipFile(elem.path) as zip_ref:
                zip_ref.extractall(elem.path.split('.zip')[0])
                print('Unzipped {} to {}'.format(elem.name, elem.path.split('.zip')[0]))
