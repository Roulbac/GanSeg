import os
import sys

idx = 0
for root, _, fnames in os.walk(sys.argv[1]):
    if fnames != []:
        for fname in fnames:
            if 'Z' in fname:
                fpath = os.path.abspath(os.path.join(root, fname))
                os.rename(fpath, os.path.join(sys.argv[1], 'Z{}'.format(idx)))
                idx += 1
