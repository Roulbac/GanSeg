import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):

    def __init__(self, *args, **kwargs):
        self.data = torch.Tensor(10, 3, 256, 256).uniform_()

    def __len__(self):
        return 10

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def modify_commandline_options(opts_parser):
        return opts_parser

