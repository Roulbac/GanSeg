from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader, _DataLoaderIter
import importlib


class _DataLoaderIterWrapper(_DataLoaderIter):
    def __next__(self):
        try:
            super().__next__()
        except StopIteration:
            print('STOPPED ITERATION')  # TODO: Am i going to use this?
            raise StopIteration


class DataLoaderWrapper(DataLoader):
    def __iter__(self):
        return _DataLoaderIterWrapper(self)


def find_dataset(dataset_name):
    dataset_libname = 'datasets.{}_dataset'.format(dataset_name)
    datasetlib = importlib.import_module(dataset_libname)
    dataset = None
    for attr_str in dir(datasetlib):
        if '{}dataset'.format(dataset_name.lower()) == attr_str.lower():
            dataset = getattr(datasetlib, attr_str)
            break
    if dataset is None:
        print('Dataset {} not found'.format(dataset_libname))
        exit(0)
    return dataset


def get_dataset_parsing_modifier(dataset):
    dataset_class = find_dataset(dataset)
    return dataset_class.modify_commandline_options


def create_dataset(opts):
    dataset = find_dataset(opts.dataset)
    full_set = dataset(opts)
    print('Created dataset for {}'.format(opts.dataset))
    if opts.mode == 'test':
        return full_set
    train_set, val_set = full_set.get_train_val(opts)
    return train_set, val_set


def get_loader(data, batch_size, num_workers=0, shuffle=True):
    train_loader = DataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
    return train_loader
