import os
from shutil import copyfile
from abc import ABC, abstractmethod
import torch
import models.utils as mutils
import utils


class BaseModel(ABC):

    def __init__(self, opts, visualizer=None):
        self.name = opts.name
        self.opts = opts
        self.is_train = (opts.mode == 'train')
        self.gpu_ids = [int(i) for i in opts.gpu_ids.split(',') if i != '']
        self.device = torch.device('cuda:{:d}'.format(
            self.gpu_ids[0]) if self.gpu_ids else torch.device('cpu'))
        self.dirname = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'training',
            self.name
        )
        utils.mkdir(self.dirname)
        self.epoch = 0
        self.iter = 0
        self.metric = None  # For scheduler
        self.plotcache = {}
        self.imcache = {}
        self.traincache = {}
        self.inputshapes = {}
        self.nets = {}
        self.losses = {}
        self.optimizers = {}
        self.schedulers = {}
        self.visualizer = visualizer

    @abstractmethod
    def set_input(self, data_input):
        pass

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def optimize_parameters(self):
        pass

    @abstractmethod
    def do_validation(self, val_loader):
        pass

    @abstractmethod
    def make_nets(self, opts):
        pass

    @abstractmethod
    def make_losses(self):
        pass

    @abstractmethod
    def make_optimizers(self):
        pass

    def init_viz(self, opts_str=''):
        if self.visualizer is None:
            return
        self.base_init_viz(opts_str)

    def update_viz(self):
        if self.visualizer is None:
            return
        self.base_update_viz()

    def make_test_state_dict(self, checkpoint_dict):
        """make_test_state_dict
        Removes entries from checkpoint dict that are unnecessary
        when in test mode

        :param checkpoint_dict: dictionary as saved on disk
        """
        entries_to_delete = ['cache', 'optimizers', 'schedulers']
        keys = list(checkpoint_dict.keys())
        for key in keys:
            if any(map(lambda x: x in key, entries_to_delete)):
                del checkpoint_dict[key]

    def base_init_viz(self, opts_str):
        assert self.visualizer is not None
        assert self.is_train
        viz = self.visualizer
        inputshapes = self.inputshapes
        env_main = '{}_main'.format(self.name)
        env_plots = '{}_plots'.format(self.name)
        viz.handle
        viz.handle.delete_env(env_main)
        viz.handle.delete_env(env_plots)
        viz.make_text_window(opts_str,
                             win='Experiment options',
                             env=env_main)
        with open('{}/opts.txt'.format(self.dirname), 'w+') as f:
            f.write(opts_str)
        #   Make network graphs
        svg_strings = mutils.torchviz_dot.get_model_svgs(self, inputshapes)
        #   Make SVG windows. Use .pipe() for SVG and dot.format = 'svg' to svg
        for name, val in svg_strings.items():
            viz.plot_svg(val, win=name, env=env_main)
        #   Initialize loss plots
        iters = list(range(1, self.iter+1, self.opts.log_freq))
        if all(map(lambda x: x != [], self.plotcache.values())):
            viz.plot_cache(self, iters, all_history=True)
        #   Initialize gradient plots
        for name, net in self.nets.items():
            named_layers = mutils.get_named_layers(net)
            viz.plot_grad_flow(
                named_layers,
                net_name=name,
                win='{}_grads'.format(name),
                env=env_plots
            )
        #  Initialize weights histograms
        viz.plot_layers_weights(self, env=env_plots)

    def base_update_viz(self):
        # Update losses, gradient barchart and weights histograms
        assert self.visualizer is not None
        assert self.is_train
        viz = self.visualizer
        env_plots = '{}_plots'.format(self.name)
        iters = [self.iter]
        viz.plot_cache(self, iters, all_history=False)
        # Update gradient barchart
        for name, net in self.nets.items():
            named_layers = mutils.get_named_layers(net)
            viz.plot_grad_flow(
                named_layers,
                net_name=name,
                win='{}_grads'.format(name),
                env=env_plots
            )
        # Update layer weights
        viz.plot_layers_weights(self, env=env_plots)

    def make_train_state_dict(self):
        """make_train_state_dict
        Makes model-specific training state_dict

        :param epoch:
        """
        state_dict = {'epoch': self.epoch, 'iter': self.iter}
        for key, val in self.nets.items():
            state_dict['nets_{}'.format(key)] = val.state_dict()
        for key, val in self.optimizers.items():
            state_dict['optimizers_{}'.format(key)] = val.state_dict()
        for key, val in self.schedulers.items():
            state_dict['schedulers_{}'.format(key)] = val.state_dict()
        state_dict['plotcache'] = self.plotcache
        state_dict['imcache'] = self.imcache
        return state_dict

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return parser

    def add_loss(self, loss, key, *args):
        self.losses[key] = loss
        self.traincache[key] = 0
        self.plotcache[key] = []
        for arg in args:
            self.traincache[arg] = 0
            self.plotcache[arg] = []

    def set_schedulers(self):
        """setup
        Set up model schedulers
        Print info
        """
        for name, optimizer in self.optimizers.items():
            self.schedulers[name] = mutils.get_scheduler(
                optimizer, self.opts)  # Last epoch of scheduler

    def set_train(self):
        for net in self.nets.values():
            net.train()

    def set_eval(self):
        for net in self.nets.values():
            net.eval()

    def save_traincache(self):
        def is_numerical(x):
            try:
                float(x)
            except ValueError:
                return False
            return True
        for name, val in self.traincache.items():
            if is_numerical(val):
                self.plotcache[name].append(val)

    def print_info(self):
        print('\n\n\t\t\tMODEL INFORMATION                   \n\n')
        print('------------------------Model: {}--------------------------- \n\n'.format(self.name))
        print('-----------------------Networks-----------------------------')
        for net_name, net in self.nets.items():
            # net.print_info() #TODO
            # print_net_info(net)
            print(net)
            print('\n')

    def _append_to_state_dict(self, state_dict, **kwargs):
        """_append_to_state_dict
        Appends attribute to state dict
        Attribute key is of form <type>_name

        :param state_dict:
        :param **kwargs:
        """
        def is_pytorch(x):
            return isinstance(x, torch.optim.Optimizer) or \
                isinstance(x, torch.nn.Module)

        def save_as(x):
            return x.state_dict() if is_pytorch(x) else x
        appended_dict = {}
        for key, val in kwargs.items():
            appended_dict[key] = save_as(val)
        state_dict.update(appended_dict)

    def _init_from_checkpoint_dict(self, state_dict):
        """_init_from_checkpoint_dict
        Takes a state dict and initializes the object's attributes
        from that state dict
        The keys in the state dict represent the attributes.
        If they are prefixed with a _, this means the attribute is in a dictionary.
        The prefix corresponds to the dictionary's attribute name under self.

        :param state_dict: state dict to initialize from
        """
        for key, val in state_dict.items():
            split_key = key.split('_')
            if len(split_key) == 1:
                setattr(self, key, val)
            else:
                prefix = split_key[0]
                attr_name = '_'.join(split_key[1:])
                attr_dict = getattr(self, prefix)

                def is_pytorch(x):
                    if isinstance(
                        x,
                        (torch.optim.Optimizer,
                         torch.nn.Module,
                         torch.optim.lr_scheduler._LRScheduler)
                    ):
                        return True
                    return False

                def load_attr(attr_dict, attr_name, val):
                    attr = attr_dict[attr_name]
                    if is_pytorch(attr):
                        attr.load_state_dict(val)
                    else:
                        attr_dict[attr_name] = val
                load_attr(attr_dict, attr_name, val)

    def save_checkpoint(self, latest=True):
        """save_checkpoint
        Saves networks, optimizers and loss state dicts

        :param epociter: Epoch and iteration at which to checkpoint
        :param latest: Copies checkpoint to be latest
        """
        checkpoint_dir = os.path.join(
            self.dirname,
            'checkpoints'
        )
        utils.mkdirs(checkpoint_dir)
        checkpoint_path = os.path.abspath(
            os.path.join(checkpoint_dir, '{}_{}.tar'.format(
                self.epoch, self.iter))
        )
        checkpoint_dict = self.make_train_state_dict()
        torch.save(checkpoint_dict, checkpoint_path)
        print('Saved checkpoint {}'.format(checkpoint_path))
        if latest:
            file_path_latest = os.path.join(checkpoint_dir,
                                            'latest.tar')
            copyfile(checkpoint_path, file_path_latest)

    def restore_checkpoint(self, epoch_iter='latest'):
        """restore_checkpoint
        Restores saved checkpoint

        :param epochiter: Epoch at which the checkpoint was saved
        """
        checkpoint_dir = os.path.join(
            self.dirname,
            'checkpoints'
        )
        checkpoint_path = os.path.join(
            checkpoint_dir, '{}.tar'.format(epoch_iter))
        checkpoint_dict = torch.load(checkpoint_path, map_location=self.device)
        if not self.is_train:
            self.make_test_state_dict(checkpoint_dict)
        self._init_from_checkpoint_dict(checkpoint_dict)
        print('Loaded checkpoint from {}'.format(checkpoint_path))

    def set_requires_grad(self, nets, requires_grad=False):
        """set_requires_grad
        Sets whether or not we computer the gradient for nets

        :param nets: Net or list of nets
        :param requires_grad: bool
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def update_learning_rate(self):
        """update_learning_rate

        To be called at the end of an epoch
        """
        lrs = {}
        for name, scheduler in self.schedulers.items():
            scheduler.step(self.metric)
            lrs[name] = scheduler.get_lr()
        print('Learning rates:\n\t{}'.format(lrs))
