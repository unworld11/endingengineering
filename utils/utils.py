from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import os
import torch
import numpy as np
from PIL import Image

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


class ModelEma(object):
    """Simple EMA tracker for model weights and buffers."""
    def __init__(self, model, decay):
        self.decay = float(decay)
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        for param in self.ema.parameters():
            param.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            model_params = dict(model.named_parameters())
            ema_params = dict(self.ema.named_parameters())
            for name, ema_param in ema_params.items():
                model_param = model_params[name]
                ema_param.mul_(self.decay).add_(model_param.detach(), alpha=1.0 - self.decay)

            model_buffers = dict(model.named_buffers())
            ema_buffers = dict(self.ema.named_buffers())
            for name, ema_buffer in ema_buffers.items():
                model_buffer = model_buffers[name]
                if torch.is_floating_point(ema_buffer):
                    ema_buffer.mul_(self.decay).add_(model_buffer.detach(), alpha=1.0 - self.decay)
                else:
                    ema_buffer.copy_(model_buffer.detach())

def save_models(model, path, suffix=''):
    """Save model to given path
    Args:
        model: model to be saved
        path: path that the model would be saved
    """
    if not os.path.exists(path):
        os.makedirs(path)
    file_path = os.path.join(path, "model_{}.pt".format(suffix))
    torch.save(model, file_path)

def save_states(states, path, suffix=''):
    """Save training states to given path
    Args:
        states: states of the optimizer, scheduler, and epoch to be saved
        path: path that states would be saved
    """
    if not os.path.exists(path):
        os.makedirs(path)
    file_path = os.path.join(path, "states_{}.pt".format(suffix))
    torch.save(states, file_path)

#lighting data augmentation
imagenet_pca = {
    'eigval': np.asarray([0.2175, 0.0188, 0.0045]),
    'eigvec': np.asarray([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}

class Lighting(object):
    """ 
    color jitter 
    source: https://github.com/liuzechun/MetaPruning
    """
    def __init__(self, alphastd,
                 eigval=imagenet_pca['eigval'],
                 eigvec=imagenet_pca['eigvec']):
        self.alphastd = alphastd
        assert eigval.shape == (3,)
        assert eigvec.shape == (3, 3)
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0.:
            return img
        rnd = np.random.randn(3) * self.alphastd
        rnd = rnd.astype('float32')
        v = rnd
        old_dtype = np.asarray(img).dtype
        v = v * self.eigval
        v = v.reshape((3, 1))
        inc = np.dot(self.eigvec, v).reshape((3,))
        img = np.add(img, inc)
        if old_dtype == np.uint8:
            img = np.clip(img, 0, 255)
        img = Image.fromarray(img.astype(old_dtype), 'RGB')
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'
