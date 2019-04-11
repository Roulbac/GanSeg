import shutil
import os
import argparse
import gdcm


def get_series_dict(dirpath):
    directory = gdcm.Directory()
    loaded = directory.Load(dirpath)
    if not loaded: return {}
    scanner = gdcm.Scanner()
    seruid_tag = gdcm.Tag(0x0020, 0x000e)
    scanner.AddTag(seruid_tag)
    scanned = scanner.Scan(directory.GetFilenames())
    if not scanned: return {}
    uids = scanner.GetValues()
    series_dict = {}
    for uid in uids:
        series_dict[uid] = scanner.GetAllFilenamesFromTagToValue(
            seruid_tag, uid)
    return series_dict

# Go through each serie, get largest filename group with consistent size
# Assuming rows = cols at all times


def get_largest_subfnames(fnames):
    fname_grps = []
    s = gdcm.Scanner()
    tag = gdcm.Tag(0x0020, 0x0037)
    s.AddTag(tag)
    s.Scan(fnames)
    vals = s.GetValues()
    for val in vals:
        fname_grp = s.GetAllFilenamesFromTagToValue(tag, val)
        fname_grps.append(fname_grp)
    return max(fname_grps, key=lambda x: len(x))


def remove_grouplengths(f, ds):
    it = ds.GetDES().begin()
    while not it.equal(ds.GetDES().end()):
        de = it.next()
        t = de.GetTag()
        if t.IsGroupLength():
            ds.GetDES().erase(it)
        else:
            vr = gdcm.DataSetHelper.ComputeVR(f, ds, t)
            if(vr.Compatible(gdcm.VR(gdcm.VR.SQ))):
                sq = de.GetValueAsSQ()
                if sq is not None \
                   and sq.GetNumberOfItems() > 0:
                    n = sq.GetNumberOfItems()
                    for i in range(1, n+1):
                        item = sq.GetItem(i)
                        nested = item.GetNestedDataSet()
                        remove_grouplengths(f, nested)
                    de.SetValue(sq.GetPointer())
                    de.SetVLToUndefined()
                    ds.Replace(de)


def remove_tags(f, ds):
    it = ds.GetDES().begin()
    tags_to_rm = []
    while not it.equal(ds.GetDES().end()):
        de = it.next()
        t = de.GetTag()

#         def is_overlay(x):
#             return 0x6000 <= x <= 0x601e and \
#                 x % 2 == 0
#
        def is_buggy(x):
            return 0x00281055 < x < 0x7fe00010
        if (is_buggy(t.GetElementTag())):
            tags_to_rm.append(t.GetElementTag())
    for t in tags_to_rm:
        ds.Remove(gdcm.Tag(t))


def remove_retired(f):
    an = gdcm.Anonymizer()
    an.SetFile(f)
    an.RemoveGroupLength()
    an.RemoveRetired()
    an.RemovePrivateTags()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True,
                        help='Input directory for repair')
    parser.add_argument('--output', type=str, required=True,
                        help='Directory where to output repaired dcms')
    args = parser.parse_args()
    parent_dir = os.path.dirname(os.path.abspath(args.input))
    for root, _, files in os.walk(args.input):
        if len(files) == 0:
            continue
        output_dir = os.path.join(
            os.path.abspath(args.output),
            os.path.relpath(root, parent_dir)
        )
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        series_dict = get_series_dict(root)
        if (len(series_dict.keys()) == 0):
            continue
        print('Writing {} series from {} to {}'.format(
            len(list(filter(lambda x: len(x) > 1, series_dict.values()))),
            root,
            output_dir
        )
        )
        for sidx, suid in enumerate(series_dict.keys()):
            if (len(series_dict[suid]) <= 1):
                continue
            series_dict[suid] = get_largest_subfnames(
                series_dict[suid]
            )
            if (len(series_dict[suid]) <= 1):
                continue
            for fidx, fname in enumerate(series_dict[suid]):
                reader = gdcm.Reader()
                writer = gdcm.Writer()
                reader.SetFileName(fname)
                if not reader.Read():
                    print('Could not read {}'.format(fname))
                    continue
                f = reader.GetFile()
                ds = f.GetDataSet()
                remove_retired(f)
                remove_tags(f, ds)
                remove_grouplengths(f, ds)
                writer.SetFile(f)
                out_fname = os.path.join(
                    output_dir, '{}_{}.dcm'.format(sidx, fidx))
                writer.SetFileName(out_fname)
                if not writer.Write():
                    print('Could not write {}'.format(out_fname))
