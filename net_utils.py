import random as rd
import torch
import torch.nn as nn

from torch.nn import init
from torch.optim import lr_scheduler




CONV = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}
CONVT = {1: nn.ConvTranspose1d, 2: nn.ConvTranspose2d, 3: nn.ConvTranspose3d}

MAXPOOL = {1: nn.MaxPool1d, 2: nn.MaxPool2d, 3: nn.MaxPool3d}

BNORM = {1: nn.BatchNorm1d, 2: nn.BatchNorm2d, 3: nn.BatchNorm3d}
INORM = {1: nn.InstanceNorm1d, 2: nn.InstanceNorm2d, 3: nn.InstanceNorm3d}



def get_scheduler(optimizer):
    return lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.9)

def init_weights(net, init_type="normal", init_gain=0.02):
    """
    :net: (network)   -- network to be initialized
    :init_type: (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
    :init_gain: (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    print(f"Initialize network with {init_type}")
    net.apply(init_func)  # apply the initialization function <init_func>


class RandomElementPool():
    """ Taken from https://github.com/deepakhr1999/cyclegans """
    def __init__(self, rate=0.5, pool_size=50):
        self.rate = rate
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_elts = 0
            self.pool = []

    def query(self, elements):
        #Receive a batch of elements?
        """ """
        if self.pool_size == 0: # If no buffer always return last elements
            return elements
        return_elts = []
        for elt in elements:
            elt = torch.unsqueeze(elt.data, 0) # Needed?
            if self.num_elts < self.pool_size:
                # If buffer still has room, stuff it
                self.num_elts += 1
                self.pool.append(elt)
                return_elts.append(elt)
            else:
                p = rd.uniform(0, 1)
                # Depending on given rate, pop an element from the pool to
                # replace by the current one and return poped element...
                if p > self.rate:
                    idx = rd.randint(0, self.pool_size - 1)
                    tmp = self.pool[idx].clone()
                    self.pool[idx] = elt
                    return_elts.append(tmp)
                # ... Or return current element
                else:
                    return_elts.append(elt)
        return torch.cat(return_elts, 0)
