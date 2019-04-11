import torch
import models.networks as mnets
import models.utils as mutils


class MSELossWithSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.sigmoid = torch.nn.Sigmoid()
        self.loss = lambda x, y: self.mse(self.sigmoid(x), y)

    def forward(self, source, target):
        return self.loss(source, target)


class GANLoss(object):

    _modes = ['vanilla', 'ogvanilla', 'lsgan', 'wgangp']

    def __init__(self, net_D, mode='vanilla', reshape_tensors=False):
        """__init__
        Initialize GANLoss with one of many modes
        Optionally, if predictions are of shape NxCxHxW,
        they can be kept that way or reshaped to (N*H*W)xC

        :param net_D: Discriminator
        :param mode: Gan loss mode
        :param reshape_tensors: If True, preds reshaped to (N*H*W)xC
        """
        super().__init__()
        mode_split = mode.split('_')
        self.mode = mode_split[0]
        self.reshape_tensors = reshape_tensors
        assert self.mode in GANLoss._modes
        self.lambd = 10
        if self.mode == 'wgangp' and len(mode_split) == 2:
            self.lambd = float(mode_split[1])
        self.net_D = net_D
        self.one_tensor = torch.Tensor([1.0])
        self.zero_tensor = torch.Tensor([0.0])
        self._bcewlogits = torch.nn.BCEWithLogitsLoss()
        self._mseloss = MSELossWithSigmoid()
        self._wganloss = lambda real, fake: fake.mean() - real.mean()

    def get_preds_and_labels(self, X, label):
        if self.reshape_tensors:
            pred, label = self.get_reshaped_tensors(X, label)
        else:
            pred, label = self.get_non_reshaped_tensors(X, label)
        label = label.to(device=pred.device)
        return pred, label

    def get_preds(self, X):
        if self.reshape_tensors:
            return self.reshape_pred(self.net_D(X))
        else:
            return self.net_D(X)

    def reshape_pred(self, pred):
        """reshape_pred
        Reshapes pred of shape NxCxHxW to (N*H*W)xC

        :param pred: pred of shape NxCxHxW
        """
        assert len(pred.shape) == 4, 'Prediction not of shape NxCxHxW'
        n_N, n_C, n_H, n_W = pred.shape
        reshaped_pred = pred.permute(0, 3, 2, 1)
        reshaped_pred = reshaped_pred.reshape(n_N*n_H*n_W, n_C)
        return reshaped_pred

    def get_reshaped_tensors(self, X, label):
        """get_reshaped_tensors
        Reshapes tensors for use in loss function
        Outputs pred of shape (NxHxW)xC and label of shape (NxHxW)
        Each mini-batch element is expanded as HxW elements with the same label
        as the original mini-batch element

        :param pred: tensor of shape NxCxHxW
        :param label: labels for the batch, either integer or tensor of size N
        """
        pred = self.net_D(X)
        n_N, n_C, n_H, n_W = pred.shape
        pred = reshape_pred(pred)
        reshaped_label = None
        if label == 1:
            reshaped_label = self.one_tensor.expand_as(reshaped_pred)
        elif label == 0:
            reshaped_label = self.zero_tensor.expand_as(reshaped_pred)
        else:
            assert label.numel() == n_N and len(label.shape) == 1, 'Label not of shape (N,)'
            reshaped_label = label.reshape(label.shape + (1, 1)).expand(
                (-1, n_H, n_W)
            )
            reshaped_label = reshaped_label.reshape(n_N*n_H*n_W)
        return reshaped_pred, reshaped_label

    def get_non_reshaped_tensors(self, X, label):
        """get_non_reshaped_tensors
        Returns predictions and labels of shape NxCxHxW

        :param X:
        :param label:
        """
        pred = self.net_D(X)
        label_tensor = None
        if label == 1:
            label_tensor = self.one_tensor.expand_as(pred)
        elif label == 0:
            label_tensor = self.zero_tensor.expand_as(pred)
        else:
            raise NotImplementedError()
        return pred, label_tensor

    def get_D_loss_vanilla(self, real_X, fake_X):
        real_pred, real_target = self.get_preds_and_labels(real_X, 1)
        fake_pred, fake_target = self.get_preds_and_labels(fake_X, 0)
        expanded_label = torch.cat([real_target, fake_target], dim=0)
        expanded_pred = torch.cat([real_pred, fake_pred], dim=0)
        return self._bcewlogits(expanded_pred, expanded_label)

    def get_G_loss_vanilla(self, fake_X):
        fake_pred, fake_target = self.get_preds_and_labels(fake_X, 1)
        return self._bcewlogits(fake_pred, fake_target)

    def get_D_loss_ogvanilla(self, real_X, fake_X):
        return self.get_D_loss_vanilla(real_X, fake_X)

    def get_G_loss_ogvanilla(self, fake_X):
        fake_pred, fake_target = self.get_preds_and_labels(fake_X, 0)
        return (-1) * self._bcewlogits(fake_pred, fake_target)

    def get_D_loss_lsgan(self, real_X, fake_X):
        real_pred, real_target = self.get_preds_and_labels(real_X, 1)
        fake_pred, fake_target = self.get_preds_and_labels(fake_X, 0)
        expanded_label = torch.cat([real_target, fake_target], dim=0)
        expanded_pred = torch.cat([real_pred, fake_pred], dim=0)
        return self._mseloss(expanded_pred, expanded_label)

    def get_G_loss_lsgan(self, fake_X):
        fake_pred, fake_target = self.get_preds_and_labels(fake_X, 1)
        return self._mseloss(fake_pred, fake_target)

    def grad_penalty(self, real_X, fake_X, lambd=10):
        N = real_X.shape[0]
        rand_shape = (N,) + (1,) * (len(real_X.shape) - 1)
        if real_X.device != fake_X.device:
            fake_X = fake_X.to(real_X.device)
        alphas = torch.rand(rand_shape).expand_as(real_X).to(real_X.device)
        x_hat = alphas*real_X + (1 - alphas)*fake_X
        x_hat.requires_grad_()
        x_hat_pred = self.get_preds(x_hat)
        grad_outputs = self.one_tensor.expand_as(x_hat_pred).to(real_X.device)
        grads = torch.autograd.grad(outputs=x_hat_pred, inputs=x_hat,
                                    grad_outputs=grad_outputs, retain_graph=True,
                                    create_graph=True, only_inputs=True)[0]
        grad_norms = grads.view((grads.shape[0], -1)).norm(p=2, dim=1)
        penalty_term = lambd*(((grad_norms - 1)**2).mean())
        return penalty_term

    def get_D_loss_wgangp(self, real_X, fake_X):
        real_pred, real_target = self.get_preds_and_labels(real_X, 1)
        fake_pred, fake_target = self.get_preds_and_labels(fake_X, 0)
        loss = self._wganloss(real_pred, fake_pred)
        lambd = self.lambd  # self.mode is wgangp-<lambda>
        loss += self.grad_penalty(real_X, fake_X, lambd=lambd)
        return loss

    def get_G_loss_wgangp(self, fake_X):
        fake_pred = self.net_D(fake_X)
        return -fake_pred.mean()

    def get_D_loss(self, real_X, fake_X):
        assert real_X.shape == fake_X.shape
        loss_fun = getattr(self, 'get_D_loss_{}'.format(self.mode))
        return loss_fun(real_X, fake_X)

    def get_G_loss(self, fake_X):
        loss_fun = getattr(self, 'get_G_loss_{}'.format(self.mode))
        return loss_fun(fake_X)


class SegLoss(object):

    LOSSES = ['nll'] + GANLoss._modes

    def __init__(self, opts):
        """__init__
        SegLoss is a class for computing segmentation loss
        Loss can be one of normal crossentropy or an adversarial type of loss
        Adversarial loss makes use of a neural network for parametrization

        :param opts: opts must have the following fields:
                -loss (type of loss to use)
                -disc_type
                -norm
                -image_shape
                -init_type
                -gpu_ids
                -lr
                -beta1
                -reshape_tensors
        """
        assert opts.loss.split('_')[0] in SegLoss.LOSSES, 'Invalid loss'
        self.opts = opts
        if opts.loss.split('_')[0] in GANLoss._modes:
            n_C_D = int(opts.image_shape.split(',')[0]) + opts.n_labels
            net_D = mnets.make_discriminator(in_channels=n_C_D,
                                             disc_type=opts.disc_type,
                                             norm_name=opts.norm
                                             )
            net_D = mutils.move_to_gpus(net_D, list(map(int, opts.gpu_ids.split(','))))
            mutils.init_net(net_D, init_type=opts.init_type)
            self.optimizer_D = torch.optim.Adam(
                net_D.parameters(), lr=opts.lr, betas=(opts.beta1, 0.999))
            self.loss = GANLoss(net_D, opts.loss, opts.reshape_tensors)
        elif opts.loss == 'nll':
            self.loss = torch.nn.NLLLoss()

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

    def __call__(self, y, y_pred, x=None, pooled=None):
        """

        :param y: NHW with entries being of type int64
        :param y_pred: NCcHW with Cc being the number of classes
        :param x: NCiHW with Ci being the number of channels of input image
        :param pooled: pooled tensor of shaped N(Ci+Cc)HW from the pooler
        """
        if self.opts.loss.split('_')[0] in GANLoss._modes:
            assert x is not None, 'Need input tensor for adversarial loss'
            n, _, h, w = y.shape
            c = self.opts.n_labels
            y_one_hot = torch.Tensor(n, c, h, w).to(y.device).zero_()
            y_one_hot.scatter_(1, y, 1)
            net_D = self.loss.net_D
            self.set_requires_grad(net_D, requires_grad=True)
            real_XY = torch.cat([x, y_one_hot], dim=1)
            fake_XY = torch.cat([x, y_pred], dim=1)
            if pooled is not None:
                gan_loss_D = 0.5 * self.loss.get_D_loss(
                    real_XY,
                    pooled
                )
            else:
                gan_loss_D = 0.5 * self.loss.get_D_loss(
                    real_XY,
                    fake_XY.detach()
                )
            self.optimizer_D.zero_grad()
            gan_loss_D.backward()
            self.optimizer_D.step()
            self.set_requires_grad(net_D, requires_grad=False)
            gan_loss_G = self.loss.get_G_loss(fake_XY)
            return gan_loss_G + gan_loss_D.detach()
        elif self.opts.loss == 'nll':
            if len(y.shape) == 4:
                loss = self.loss(y_pred, y.squeeze(1))
            else:
                loss = self.loss(y_pred, y)
            return loss
