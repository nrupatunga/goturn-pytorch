"""
File: network.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: network architecture
"""

import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from loguru import logger
from torchsummary import summary

try:
    from goturn.network.caffenet import CaffeNet
except ImportError:
    logger.error('Please run $source settings.sh from root directory')
    sys.exit(1)


class GoturnNetwork(nn.Module):

    """AlexNet based network for training goturn tracker"""

    def __init__(self, pretrained_model=None,
                 init_fc=None, num_output=4):
        """ """
        super(GoturnNetwork, self).__init__()

        self._net_1 = CaffeNet(pretrained_model_path=pretrained_model)
        self._net_2 = CaffeNet(pretrained_model_path=pretrained_model)
        dropout_ratio = 0.5
        self._classifier = nn.Sequential(nn.Linear(256 * 6 * 6 * 2, 4096),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(dropout_ratio),
                                         nn.Linear(4096, 4096),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(dropout_ratio),
                                         nn.Linear(4096, 4096),
                                         nn.ReLU(inplace=True),
                                         nn.Dropout(dropout_ratio),
                                         nn.Linear(4096, num_output))

        self._num_output = num_output
        if init_fc:
            logger.info('Using caffe fc weights')
            self._init_fc = init_fc
            self._caffe_fc_init()
        else:
            logger.info('Not using caffe fc weights')
            self.__init_weights()

    def forward(self, x1, x2):
        """Foward pass
        @x: input
        """
        x1 = self._net_1(x1)
        x1 = x1.view(x1.size(0), 256 * 6 * 6)

        x2 = self._net_2(x2)
        x2 = x2.view(x2.size(0), 256 * 6 * 6)

        x = torch.cat((x1, x2), 1)
        x = self._classifier(x)

        return x

    def __init_weights(self):
        """Initialize the extra layers """
        for m in self._classifier.modules():
            if isinstance(m, nn.Linear):
                if self._num_output == m.out_features:
                    init.normal_(m.weight.data, mean=0.0, std=0.01)
                    init.zeros_(m.bias.data)
                else:
                    init.normal_(m.weight.data, mean=0.0, std=0.005)
                    init.ones_(m.bias.data)

    def _caffe_fc_init(self):
        """Init from caffe normal_
        """
        wb = np.load(self._init_fc, allow_pickle=True).item()

        layer_num = 0
        with torch.no_grad():
            for layer in self._classifier.modules():
                if isinstance(layer, nn.Linear):
                    layer_num = layer_num + 1
                    key_w = 'fc{}_w'.format(layer_num)
                    key_b = 'fc{}_b'.format(layer_num)
                    w, b = wb[key_w], wb[key_b]
                    w = np.reshape(w, (w.shape[1], w.shape[0]))
                    b = np.squeeze(np.reshape(b, (b.shape[1],
                                                  b.shape[0])))
                    layer.weight.copy_(torch.from_numpy(w).float())
                    layer.bias.copy_(torch.from_numpy(b).float())


if __name__ == "__main__":
    net = GoturnNetwork().cuda()
    summary(net, input_size=[(3, 227, 227), (3, 227, 227)])
