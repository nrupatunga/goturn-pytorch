"""
File: caffenet.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: caffenet architecture for training goturn
"""
import numpy as np
import torch
import torch.nn as nn


class CaffeNetArch(nn.Module):

    """Docstring for AlexNet. """

    def __init__(self, num_classes=1000):
        """This defines the caffe version of alexnet"""
        super(CaffeNetArch, self).__init__()

        self.features = nn.Sequential(nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2),
                                      nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75),
                                      # conv 2
                                      nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2),
                                      nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75),
                                      # conv 3
                                      nn.Conv2d(256, 384, kernel_size=3, padding=1),
                                      nn.ReLU(inplace=True),
                                      # conv 4
                                      nn.Conv2d(384, 384, kernel_size=3, padding=1, groups=2),
                                      nn.ReLU(inplace=True),
                                      # conv 5
                                      nn.Conv2d(384, 256, kernel_size=3, padding=1, groups=2),
                                      nn.ReLU(inplace=True),
                                      nn.MaxPool2d(kernel_size=3, stride=2))


def transfer_weights(model, pretrained_model_path, dbg=False):
    weights_bias = np.load(pretrained_model_path, allow_pickle=True,
                           encoding='latin1').item()
    layer_num = 0
    with torch.no_grad():
        for layer in model.modules():
            if type(layer) == torch.nn.modules.conv.Conv2d:
                layer_num = layer_num + 1
                key = 'conv{}'.format(layer_num)
                w, b = weights_bias[key][0], weights_bias[key][1]
                layer.weight.copy_(torch.from_numpy(w).float())
                layer.bias.copy_(torch.from_numpy(b).float())

    if dbg:
        layer_num = 0
        for layer in model.modules():
            if type(layer) == torch.nn.modules.conv.Conv2d:
                layer_num = layer_num + 1
                key = 'conv{}'.format(layer_num)
                w, b = weights_bias[key][0], weights_bias[key][1]
                assert (layer.weight.detach().numpy() == w).all()
                assert (layer.bias.detach().numpy() == b).all()


def CaffeNet(pretrained_model_path=None):
    """Alexenet pretrained model
    @pretrained_model_path: pretrained model path for initialization
    """

    model = CaffeNetArch().features
    if pretrained_model_path:
        transfer_weights(model, pretrained_model_path)

    return model
