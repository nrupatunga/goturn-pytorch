"""
File: alexnet.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: This is caffe version of alexnet
Source: https://github.com/jiecaoyu/pytorch_imagenet/blob/master/networks/model_list/alexnet.py
"""
import torch
import torch.nn as nn


class AlexNetNN(nn.Module):

    """Docstring for AlexNet. """

    def __init__(self, num_classes=1000):
        """This defines the caffe version of alexnet"""
        super(AlexNetNN, self).__init__()

        self.features = nn.Sequential(nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
                                      nn.ReLU(inplace=True),
                                      nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75),
                                      nn.MaxPool2d(kernel_size=3, stride=2),
                                      # conv 2
                                      nn.Conv2d(96, 256, kernel_size=5, padding=2, groups=2),
                                      nn.ReLU(inplace=True),
                                      nn.LocalResponseNorm(5, alpha=0.0001, beta=0.75),
                                      nn.MaxPool2d(kernel_size=3, stride=2),
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

        self.classifier = nn.Sequential(nn.Linear(256 * 6 * 6, 4096),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(),
                                        nn.Linear(4096, 4096),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(),
                                        nn.Linear(4096, num_classes))


def AlexNet(pretrained_model_path=None):
    """Alexenet pretrained model
    @pretrained_model_path: pretrained model path for initialization
    """

    model = AlexNetNN()
    if pretrained_model_path:
        pretrained_model = torch.load(pretrained_model_path)
        model.load_state_dict(pretrained_model['state_dict'])

    return model
