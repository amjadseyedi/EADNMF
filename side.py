import numpy as np
import torch
from torch.autograd import Variable


def np_to_tensor(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)


def np_to_var(img_np, dtype=torch.cuda.FloatTensor):
    '''Converts image in numpy.array to torch.Variable.

    From C x W x H [0..1] to  1 x C x W x H [0..1]
    '''
    return Variable(np_to_tensor(img_np)[None, :])


def var_to_np(img_var):
    '''Converts an image in torch.Variable format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.data.cpu().numpy()[0]