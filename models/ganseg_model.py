import os
from functools import partial
import torch
import numpy as np
import models.networks as mnets
import models.losses as mlosses
from models.base_model import BaseModel
import models.utils as mutils

class GanSeg(BaseModel):

    @staticmethod
    def modify_commandline_options(options_parser, is_train=True):
        # Add model construction opts
        parser = options_parser.parser
        parser.add_argument('--norm', type=str, default='InstanceNorm',
                            choices=['InstanceNorm', 'BatchNorm'],
                            help='Normalzation layer')
        parser.add_argument('--no_dropout', action='store_true',
                            help='Use dropout for the model')
        parser.add_argument('--gen_type', type=str, default='cycleresnet_9',
                            help='Generator architecture'
                            'or resnet')
        if is_train:
            # Add training construction opts
            parser.add_argument('--pool_size', type=int, default=50,
                                help='Buffer size for previously generated images')
            parser.add_argument('--loss', type=str, default='nll',
                                help='GAN mode for generator [nll, vanilla, lsgan or wgangp]')
            parser.add_argument('--reshape_tensors', action='store_true',
                                help='Reshape tensors from NCHW to (NHW)C')
            parser.add_argument('--disc_type', type=str, default='64x64',
                                help='Discriminator type')
            parser.add_argument('--lambda_L1', type=float, default=0.0,
                                help='L1 loss scalar')
            parser.add_argument('--beta1', type=float, default=0.5,
                                help='Adam beta1 param')
        else:
            parser.add_argument('--eval', action='store_true',
                                help='Set norm and dropout layers to eval at test time')

    def __init__(self, opts, visualizer=None):
        super().__init__(opts, visualizer)
        self.image_shape = tuple(map(int, opts.image_shape.split(',')))
        self.inputshapes = {'G': (1,
                                  self.image_shape[0],
                                  self.image_shape[1],
                                  self.image_shape[2])
                            }
        self.make_nets()
        if self.is_train:
            self.make_losses()
            self.make_optimizers()
            self.image_pooler = mutils.image_pool.ImagePool(opts.pool_size)
            self.set_schedulers()

    def make_nets(self):
        """make_nets
        Make generator.
        If training mode, make discriminator

        """
        net_G = mnets.make_generator(gen_type=self.opts.gen_type,
                                     image_shape=self.image_shape,
                                     norm_name=self.opts.norm,
                                     out_channels=self.opts.n_labels,
                                     out_activation=partial(
                                         torch.nn.LogSoftmax, dim=1)
                                     )
        net_G = mutils.move_to_gpus(net_G, self.gpu_ids)
        self.nets['G'] = net_G
        if self.is_train:
            init_type = None
            if hasattr(self.opts, 'init_type'):
                init_type = self.opts.init_type
            mutils.init_net(net_G, init_type=init_type)
            self.traincache['dice'] = 0
            self.plotcache['dice'] = []

    def make_optimizers(self):
        """make_optimizers
        Make optimizers

        """
        net_G = self.nets['G']
        optimizer_G = torch.optim.Adam(
            net_G.parameters(), lr=self.opts.lr, betas=(self.opts.beta1, 0.999))
        self.optimizers['G'] = optimizer_G

    def batch_segment_volume(self, x, as_numpy=True):
        """batch_segment_volume
        Outputs NCHW

        :param x: slices of shape NCHW, tensor
        """
        batches = []
        n = x.shape[0]
        idxs = list(range(0, n, 5)) + [n]
        with torch.no_grad():
            for i in range(len(idxs) - 1):
                batch_idxs = list(range(idxs[i], idxs[i+1]))
                batch = self.nets['G'](x[batch_idxs].to(self.device)).cpu()
                batches.append(batch.argmax(1, keepdim=True))
        if as_numpy:
            return np.concatenate(
                map(lambda x: x.numpy(), batches), axis=0).astype(np.long)
        else:
            return torch.cat(batches, dim=0).long()  # NCHW seg

    def test(self, dataset):
        results_dir = os.path.join(self.opts.results_dir, self.name)

        def fun(x):
            return self.batch_segment_volume(x, as_numpy=True).squeeze(1).astype(np.short)
        for i in range(len(dataset)):
            dataset.save_slices_as_vol(i, results_dir, fun)

    def make_losses(self):
        """make_losses
        Make GANLoss which gives D and G loss tensors for LSGAN, WGANGP, Vanilla or OGVanilla
        Make L1 loss

        Creates traincache values for these losses

        """
        segloss_module = mlosses.SegLoss(self.opts)
        self.add_loss(segloss_module, 'loss')

    def set_input(self, data_input):
        """set_input
        Give as input a pair of x,y

        :param data_input:
        """
        x, y = None, None
        if isinstance(data_input, (tuple, list)):
            x, y = data_input
        else:
            assert not self.is_train
            x = data_input
        self.x = x.to(self.device)
        self.y = y.to(self.device)

    def forward(self):
        """forward
        Sample the generator and place it in self.y_pred
        """
        net_G = self.nets['G']
        self.y_pred = net_G(self.x)

    def backward_G(self):
        """backward_G
        Generator backward pass using GAN loss and L1 loss
        """
        segloss_module = self.losses['loss']
        pooled_xy = None
        if self.opts.loss in mlosses.GANLoss._modes:
            pooled_xy = self.image_pooler.query(
                torch.cat([
                    self.x,
                    self.y_pred
                ], dim=1)
            )
        loss = segloss_module(self.y, self.y_pred, self.x, pooled_xy)
        if self.opts.lambda_L1 > 0:
            loss += self.opts.lambda_L1 * \
                torch.nn.functional.l1_loss(self.y_pred, self.y.float())
        loss.backward()
        self.traincache['loss'] = loss.item()

    def optimize_parameters(self):
        self.forward()
        optimizer_G = self.optimizers['G']
        optimizer_G.zero_grad()
        self.backward_G()
        optimizer_G.step()

    def do_validation(self, v_set):
        """do_validation

        :param val_loader:
        """
        vols, labels = [], []
        for v in v_set:
            vols.append(v[0])
            labels.append(v[1])
        assert len(vols) == len(labels)
        preds = []
        for vol in vols:
            preds.append(self.batch_segment_volume(vol, as_numpy=False))
        dices = []
        for label, pred in zip(labels, preds):
            denom = torch.sum(label).item() + torch.sum(pred).item()
            numer = 2*torch.sum(label & pred).item()
            dices.append(numer/denom)
        self.traincache['dice'] = sum(dices)/len(dices)

    def plot_samples(self, batch, win, caption, title, cache=True):
        with torch.no_grad():
            x, y = batch
            with torch.no_grad():
                y_pred = self.nets['G'](x).argmax(dim=1).unsqueeze(1)
            y_pred = y_pred.expand(-1, x.shape[1], -1, -1).float()
            y = y.expand(-1, x.shape[1], -1, -1).float()
            imgs = torch.cat([x, y_pred, y], dim=0).cpu()
            if cache:
                self.imcache[win] = imgs
            self.visualizer.plot_images(imgs, ncol=3, title=title,
                                        caption=caption,
                                        win=win, env='{}_imgs'.format(self.name))

    def init_viz(self, opts_str=''):
        if self.visualizer is None:
            return
        super().init_viz(opts_str)
        for name, imgs in self.imcache.items():
            self.visualizer.plot_images(imgs, ncol=3, title=name,
                                        caption='X Y X',
                                        win=name, env='{}_imgs'.format(self.name))

    def update_viz(self):
        if self.visualizer is None:
            return
        super().update_viz()
        batch = (self.x, self.y)
        self.plot_samples(batch, win='trainsamples',
                          caption='X Y X', title='Training samples')
