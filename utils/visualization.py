import os
import socket
import time
import sys
from collections import OrderedDict
from visdom import Visdom
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.utils import get_named_layers
from math import sqrt, ceil


class Visualizer(object):
    DEFAULT_PORT = 8096
    DEFAULT_HOSTNAME = "localhost"

    def __init__(self,
                 port=DEFAULT_PORT,
                 hostname=DEFAULT_HOSTNAME,
                 is_remote=False
                 ):
        """__init__
        Initializes Visualizer handle on server

        :param port:
        :param hostname:
        """
        self.port = port
        self.hostname = hostname
        self.is_remote = is_remote
        if not self.socket_is_used() and not self.is_remote:
            print('No running visdom server on {}:{}, starting a new screen'.format(
                hostname, port))
            self.start_connection_in_screen()
        viz = Visdom(port=port, server=hostname)
        self.handle = viz
        self.envs = self.handle.get_env_list()
        self.wins = []

    def socket_is_used(self):
        is_used = False
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((self.hostname, self.port))
        except socket.error:
            is_used = True
        finally:
            s.close()
        return is_used

    def start_connection_in_screen(self):
        command = 'screen -dmS viz visdom -p {} --hostname {}'.format(
            self.port, self.hostname
        )
        os.system(command)
        is_used = False
        while not is_used:
            is_used = self.socket_is_used()
            time.sleep(1)

    def plot_layers_weights(self, model, env=None, bins=30):
        nets_named_layers = {}
        figs = []
        for name, net in model.nets.items():
            named_layers = get_named_layers(net)
            nets_named_layers[name] = named_layers
        for net_name, named_layers in nets_named_layers.items():
            nrowcol = ceil(sqrt(len(named_layers)))
            index = 1
            figsize = (2*nrowcol, 2*nrowcol)
            fig = plt.figure(figsize=figsize)
            fig.suptitle('{} weights'.format(net_name))
            for layer_name, layer in named_layers.items():
                weights = layer.weight.cpu().detach().numpy().flatten()
                ax = fig.add_subplot(nrowcol, nrowcol, index)
                ax.hist(weights, bins=bins)
                ax.set_title(layer_name, fontsize='small')
                ax.tick_params(axis='x', labelsize='x-small')
                index += 1
            fig.tight_layout(rect=[0, 0.03, 1, 0.95])
            figs.append(fig)
        winid = 0
        for i in range(len(figs)):
            fig = figs[i]
            self.plot_matplot(fig, env=env, win='weights_{}'.format(winid))
            plt.close(fig)
            winid += 1

    def plot_cache(self, model, iters, all_history=False):
        """plot_cache
        Plots losses for iters
        Can either plot all losses from all iters
        or append last value of plotcache to plot

        :param model:
        :param iters:
        :param all_history:
        """
        # -----------------UTIL FUNCTIONS---------------------
        def split_dict(dic, sep='_'):
            prefix_dict = {}
            dicts = []
            for key in dic.keys():
                split = key.split(sep)
                if len(split) == 1:
                    dicts.append({key: dic[key]})
                else:
                    if split[0] not in prefix_dict.keys():
                        prefix_dict[split[0]] = [key]
                    else:
                        prefix_dict[split[0]] += [key]
            for val in prefix_dict.values():
                temp = {}
                for key in val:
                    temp[key] = dic[key]
                dicts.append(temp)
            return dicts

        def plot_dict_items(dic):
            winname = None
            Y = []
            legend = []
            for name, loss in dic.items():
                if winname is None:
                    winname = name.split('_')[0]
                if all_history:
                    Y.append(loss)
                else:
                    Y.append([loss[-1]])
                legend.append(name)
            Y = np.array(Y).T
            X = iters
            if len(X) != Y.shape[0]:
                X = list(
                    range(int(model.iter/Y.shape[0]),
                          model.iter+1,
                          int(model.iter/Y.shape[0]))
                )
            assert len(legend) == Y.shape[1]
            if all_history:
                self.plot_line(Y, X=X,
                               win=winname,
                               env=env_plots, legend=legend)
            else:
                self.plot_line(Y, X=X,
                               win=winname, update='append',
                               env=env_plots, legend=legend)
        # ------------------END OF UTIL FUNCTIONS --------------------
        env_plots = '{}_plots'.format(model.name)
        plot_dicts = split_dict(model.plotcache)
        for dic in plot_dicts:
            plot_dict_items(dic)

    def make_base_opts(self, title, **kwargs):
        """make_base_opts

        :param title:
        :param **kwargs:
        """
        opts = dict(title=title)
        opts.update(kwargs)
        return opts

    def plot_svg(self, svg, win=None, env=None, opts=None):
        """plot_svg

        :param text:
        """
        viz = self.handle
        win_name = viz.svg(svg, win=win, env=env, opts=opts)
        if win_name not in self.wins:
            self.wins.append(win_name)
        if env is not None and env not in self.envs:
            self.envs.append(env)

    def plot_grad_flow(self, named_layers, net_name='', win=None, env=None):
        ave_grads = []
        max_grads = []
        layers = []
        for layer_name, layer in named_layers.items():
            for n, p in layer.named_parameters():
                if(p.requires_grad and ("bias" not in n)):
                    layers.append(layer_name)
                    abs_grad = None
                    if p.grad is None:
                        abs_grad = torch.zeros(p.shape)
                    else:
                        abs_grad = p.grad.abs()
                    ave_grads.append(abs_grad.mean().item())
                    max_grads.append(abs_grad.max().item())
                    break
        fig, ax = plt.subplots()
        ax.bar(np.arange(len(max_grads)), max_grads,
               alpha=0.1, lw=1, color="c")
        ax.bar(np.arange(len(max_grads)), ave_grads,
               alpha=0.1, lw=1, color="b")
        ax.hlines(0, 0, len(ave_grads), lw=2, color="k")
        ax.set_xticks(range(0, len(ave_grads), 1))
        ax.set_xticklabels(layers, rotation='vertical', fontsize='small')
        ax.set_yscale('log')
        ax.set_xlabel("Layers")
        ax.set_ylabel("Gradient magnitude")
        ax.set_title('{} Gradient flow'.format(net_name))
        ax.grid(True)
        ax.legend([plt.Line2D([0], [0], color="c", lw=4),
                   plt.Line2D([0], [0], color="b", lw=4)], ['max-gradient', 'mean-gradient'])
        fig.tight_layout()
        self.plot_matplot(fig, win, env)
        plt.close(fig)

    def make_text_window(self, text, win=None, env=None):
        """make_text_window

        :param text:
        """
        viz = self.handle
        win_name = viz.text(text.replace('\n', '<br>'), win=win, env=env)
        if win_name not in self.wins:
            self.wins.append(win_name)
        if env is not None and env not in self.envs:
            self.envs.append(env)

    def plot_image(self, img_tensor, win=None, caption='', env=None):
        """plot_image
        Plots an image tensor

        :param env_name:
        """
        assert len(
            img_tensor.shape) == 3, 'Input should be a tensor of shape CxHxW'

        def normalize(x):
            m = x.min()
            M = x.max()
            ret = (x - m) / (M - m)
            ret[ret != ret] = 0
            return ret

        viz = self.handle
        normalized_img_t = normalize(img_tensor)
        win_name = viz.image(normalized_img_t,
                             win=win,
                             env=env,
                             opts=dict(caption=caption)
                             )
        if win_name not in self.wins:
            self.wins.append(win_name)
        if env is not None and env not in self.envs:
            self.envs.append(env)

    def plot_images(self, imgs, ncol, padding=2, title='', caption='', win=None, env=None):
        """plot_images

        :param imgs_tensor:
        :param nrow:
        :param padding:
        :param caption:
        :param win:
        :param env:
        """

        def normalize(x):
            x_view = x.view(x.shape[0], -1)
            m = x_view.min(dim=1)[0]
            M = x_view.max(dim=1)[0]
            m = m.reshape(-1, 1, 1, 1).expand_as(x)
            M = M.reshape(-1, 1, 1, 1).expand_as(x)
            ret = (x - m) / (M - m)
            ret[ret != ret] = 0  # Replace NaNs with 0
            return ret

        viz = self.handle
        normalized_imgs = normalize(imgs)
        b = normalized_imgs.shape[0]
        nrow = b//ncol
        win_name = viz.images(normalized_imgs, win=win, env=env,
                              padding=padding, nrow=nrow,
                              opts=dict(caption=caption, title=title)
                              )
        if win_name not in self.wins:
            self.wins.append(win_name)
        if env is not None and env not in self.envs:
            self.envs.append(env)

    def plot_line(self, Y, X=None, update=None,
                  win=None, env=None,
                  legend=None, fillarea=False
                  ):
        """plot_line

        :param Y:
        :param X:
        :param opts:
        :param update:
        :param win:
        :param env:
        """
        viz = self.handle
        opts = dict(legend=legend,
                    fillarea=fillarea)
        win_name = viz.line(Y, X=X, opts=opts,
                            update=update, win=win, env=env)
        if win_name not in self.wins:
            self.wins.append(win_name)
        if env is not None and env not in self.envs:
            self.envs.append(env)

    def plot_histogram(self, Y, X=None, numbins=30,
                       win=None, env=None, layoutopts={}):
        """plot_histogram

        :param Y:
        :param X:
        :param opts:
        :param update:
        :param win:
        :param env:
        """
        viz = self.handle
        opts = dict(numbins=numbins)
        opts.update(layoutopts)
        win_name = viz.histogram(Y, X=X, opts=opts,
                                 win=win, env=env)
        if win_name not in self.wins:
            self.wins.append(win_name)
        if env is not None and env not in self.envs:
            self.envs.append(env)

    def plot_matplot(self, fig, win=None, env=None):
        """plot_matplot

        :param fig:
        """
        viz = self.handle
        win_name = viz.matplot(fig, win=win, env=env)
        if win_name not in self.wins:
            self.wins.append(win_name)
        if env is not None and env not in self.envs:
            self.envs.append(env)


def update_progressbar(curr_iter, total_iters, str_dict={}):
    barLength = 20  # Modify this to change the length of the progress bar
    prefix_str = ''
    for key, val in str_dict.items():
        prefix_str += '{}: {:.4f}, '.format(key, val)
    prefix_str = prefix_str[:-2]
    progress = curr_iter / total_iters
    assert 0 <= progress <= 1
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1
    block = int(round(barLength*progress))
    eta = -1
    if not hasattr(update_progressbar, 'hist'):
        update_progressbar.hist = dict(progress=progress,
                                       time=time.time(),
                                       eta=eta)
        print('\n')
    else:
        curr_time = time.time()
        d_progress = progress - update_progressbar.hist['progress']
        d_time = curr_time - update_progressbar.hist['time']
        eta = d_time/d_progress * (1 - progress)
        update_progressbar.hist['progress'] = progress
        update_progressbar.hist['time'] = curr_time
    time_str = ''
    if eta >= 0:
        time_str = secondsToText(eta)
    text = "\033[F\033[F{0}\n\tProgress: [{1}] {2}/{3} {4} left\n".format(
        prefix_str, "#"*block + "-"*(barLength-block),
        curr_iter, total_iters, time_str)
    sys.stdout.write(text)
    sys.stdout.flush()
    if progress == 1:
        del update_progressbar.hist

# Credit to https://github.com/sksq96/pytorch-summary


def summary(model, input_size, batch_size=-1, device="cuda"):
    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, torch.nn.Sequential)
            and not isinstance(module, torch.nn.ModuleList)
            and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format(
        "Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) *
                           batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print(
        "Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")


def secondsToText(secs):
    days = secs//86400
    hours = (secs - days*86400)//3600
    minutes = (secs - days*86400 - hours*3600)//60
    seconds = secs - days*86400 - hours*3600 - minutes*60
    result = ("{0} d ".format(days) if days else "") + \
        ("{0} h ".format(hours) if hours else "") + \
        ("{0} m ".format(minutes) if minutes else "") + \
        ("{0:.0f} s".format(seconds) if seconds >= 0 else "")
    return result
