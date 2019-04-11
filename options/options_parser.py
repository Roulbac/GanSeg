import argparse
import sys


class OptionsParser():
    """Options
    Class for defining a model's training and testing options

    """

    def __init__(self):
        parser = argparse.ArgumentParser(
            description='Options parser for model training and testing'
        )
        # Display visualiser options
        # Model options
        parser.add_argument('--name', type=str, default='nameless',
                            help='Experiment name, sets directory where to store results and checkpoints')
        parser.add_argument('--model', type=str, required=True,
                            help='Type of model to create')
        parser.add_argument('--gpu_ids', type=str, default='',
                            help='GPU IDs on which to load the model')
        parser.add_argument('--num_workers', type=int, default=0,
                            help='Num workers for dataloader')
        parser.add_argument('--verbose', action='store_true',
                            help='Load model in verbose mode, printing debug options')
        # Loading & saving opts
        parser.add_argument('--load_epochiter', type=str, default='',
                            help='Epoch from which to load checkpoint')
        parser.add_argument('--dataset', type=str, default='',
                            help='Dataset to load')
        self.parser = parser

    def is_train(self):
        return self.parser.get_default('mode') == 'train'

    def get_model_name(self):
        args = sys.argv
        return args[args.index('--model') + 1]

    def get_dataset_name(self):
        args = sys.argv
        return args[args.index('--dataset') + 1]

    def parse_options(self, opts_list=None):
        """parse_options
        Parse command-line options

        :param opts_list: sys.argv-like list of options. If None, use sys.argv
        """
        opts = None
        remaining_args = None
        if opts_list is None:
            opts, remaining_args = self.parser.parse_known_args()
        else:
            opts, remaining_args = self.parser.parse_known_args(opts_list)
        return opts, remaining_args

    def make_opts_string(self, opts, verbose=True):
        """make_opts_string
        Print options object
        """
        output = ''
        output += 'Experiment options:\n'
        opts_dict = vars(opts)
        for key, val in (opts_dict.items()):
            default = self.parser.get_default(key)
            output += '\t{}: {} [Default: {}]\n'.format(key, val, default)
        output += 'End of model options\n'
        if verbose:
            print(output)
        return output
