"""
File: train.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: Training scripts for goturn
"""

import argparse
import random
import sys
from collections import OrderedDict

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torch.backends.cudnn as cudnn
from loguru import logger
from pytorch_lightning import Trainer
from pytorch_lightning.core.lightning import LightningModule
from torch.utils.data import DataLoader
from torch_lr_finder import LRFinder
from multiprocessing import Manager

try:
    from goturn.dataloaders.goturndataloader import GoturnDataloader
    from goturn.helper.vis_utils import Visualizer
    from goturn.network.network import GoturnNetwork
    from goturn.helper.BoundingBox import BoundingBox
    from goturn.helper.draw_util import draw
    from goturn.optimizer.caffeSGD import CaffeSGD
except ImportError:
    logger.error('Please run $source settings.sh from root directory')
    sys.exit(1)


class GoturnTrain(LightningModule):

    """Docstring for GoturnTrain. """

    def __init__(self, hparams, dbg=False):
        '''
        Pytorch lightning module for training goturn tracker.

        @hparams: all the argparse arguments for training
        @dbg: boolean for switching on visualizer
        '''
        logger.info('=' * 15)
        logger.info('GOTURN TRACKER')
        logger.info('=' * 15)

        super(GoturnTrain, self).__init__()

        # self.__set_seed(hparams.seed)
        self.hparams = hparams
        logger.info('Setting up the network...')

        # network with pretrained model
        self._model = GoturnNetwork(self.hparams.pretrained_model)
        self._dbg = dbg
        if dbg:
            self._viz = Visualizer(port=8097)

    def __freeze(self):
        """Freeze the model features layer
        """
        features_layer = self._model._net
        for param in features_layer.parameters():
            param.requires_grad = False

    def _set_conv_layer(self, conv_layers, param_dict):
        conv_layer_num = 0
        for layer in conv_layers.modules():
            if type(layer) == torch.nn.modules.conv.Conv2d:
                # conv_layer_num += 1
                if conv_layer_num == 5:
                    param_dict.append({'params': layer.weight,
                                       'lr': 1,
                                       'weight_decay': self.hparams.wd})
                    param_dict.append({'params': layer.bias,
                                       'lr': 1,
                                       'weight_decay': 0})
                else:
                    param_dict.append({'params': layer.weight,
                                       'lr': 0,
                                       'weight_decay': self.hparams.wd})
                    param_dict.append({'params': layer.bias,
                                       'lr': 0,
                                       'weight_decay': 0})

        return param_dict

    def __set_lr(self):
        '''set learning rate for classifier layer'''
        param_dict = []
        if 1:
            conv_layers = self._model._net_1
            param_dict = self._set_conv_layer(conv_layers, param_dict)
            conv_layers = self._model._net_2
            param_dict = self._set_conv_layer(conv_layers, param_dict)

            regression_layer = self._model._classifier
            for layer in regression_layer.modules():
                if type(layer) == torch.nn.modules.linear.Linear:
                    param_dict.append({'params': layer.weight,
                                       'lr': 10 * self.hparams.lr,
                                       'weight_decay': self.hparams.wd})
                    param_dict.append({'params': layer.bias,
                                       'lr': 20 * self.hparams.lr,
                                       'weight_decay': 0})

        if 0:
            regression_layer = self._model._classifier
            for layer in regression_layer.modules():
                if type(layer) == torch.nn.modules.linear.Linear:
                    param_dict.append({'params': layer.weight,
                                       'lr': 10 * self.hparams.lr,
                                       'weight_decay': self.hparams.wd})
                    param_dict.append({'params': layer.bias,
                                       'lr': 20 * self.hparams.lr,
                                       'weight_decay': 0})
        if 0:
            param_dict = [{'params': regression_layer[0].weight, 'lr': 10 * self.hparams.lr, 'weight_decay': self.hparams.wd},
                          {'params': regression_layer[3].weight, 'lr': 10 * self.hparams.lr, 'weight_decay': self.hparams.wd},
                          {'params': regression_layer[6].weight, 'lr': 10 * self.hparams.lr, 'weight_decay': self.hparams.wd},
                          {'params': regression_layer[9].weight, 'lr': 10 * self.hparams.lr, 'weight_decay': self.hparams.wd},
                          {'params': regression_layer[0].bias, 'lr': 20 * self.hparams.lr, 'weight_decay': 0},
                          {'params': regression_layer[3].bias, 'lr': 20 * self.hparams.lr, 'weight_decay': 0},
                          {'params': regression_layer[6].bias, 'lr': 20 * self.hparams.lr, 'weight_decay': 0},
                          {'params': regression_layer[9].bias, 'lr': 20 * self.hparams.lr, 'weight_decay': 0}]

        return param_dict

    def find_lr(self):
        """finding suitable learning rate """
        model = self._model
        params = self.__set_lr()

        criterion = torch.nn.L1Loss(size_average=False)
        if 0:
            optimizer = torch.optim.SGD(params,
                                        lr=1e-8,
                                        momentum=self.hparams.momentum,
                                        weight_decay=self.hparams.wd)
        else:
            optimizer = CaffeSGD(params,
                                 lr=1e-8,
                                 momentum=self.hparams.momentum,
                                 weight_decay=self.hparams.wd)

        lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
        trainloader = self.train_dataloader()
        lr_finder.range_test(trainloader, start_lr=1e-7, end_lr=1,
                             num_iter=500)
        lr_finder.plot()

    def __set_seed(self, SEED):
        ''' set all the seeds for reproducibility '''
        logger.info('Settings seed = {}'.format(SEED))
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)
        cudnn.deterministic = True

    @staticmethod
    def add_model_specific_args(parent_parser):
        ''' These are specific parameters for the sample generator '''
        ap = argparse.ArgumentParser(parents=[parent_parser])

        ap.add_argument('--min_scale', type=float, default=-0.4,
                        help='min scale')
        ap.add_argument('--max_scale', type=float, default=0.4,
                        help='max scale')
        ap.add_argument('--lamda_shift', type=float, default=5)
        ap.add_argument('--lamda_scale', type=int, default=15)
        return ap

    def configure_optimizers(self):
        """Configure optimizers"""
        logger.info('Configuring optimizer: SGD with lr = {}, momentum = {}'.format(self.hparams.lr, self.hparams.momentum))
        params = self.__set_lr()

        if 0:
            optimizer = torch.optim.SGD(params,
                                        lr=self.hparams.lr,
                                        momentum=self.hparams.momentum,
                                        weight_decay=self.hparams.wd)
        else:
            optimizer = CaffeSGD(params,
                                 lr=self.hparams.lr,
                                 momentum=self.hparams.momentum,
                                 weight_decay=self.hparams.wd)

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.hparams.lr_step, gamma=self.hparams.gamma)

        return [optimizer], [scheduler]

    @pl.data_loader
    def train_dataloader(self):
        """train dataloader"""
        logger.info('===' * 20)
        logger.info('Loading dataset for training, please wait...')
        logger.info('===' * 20)

        imagenet_path = self.hparams.imagenet_path
        alov_path = self.hparams.alov_path
        mean_file = None
        manager = Manager()
        objGoturn = GoturnDataloader(imagenet_path, alov_path,
                                     mean_file=mean_file,
                                     images_p=manager.list(),
                                     targets_p=manager.list(),
                                     bboxes_p=manager.list(),
                                     val_ratio=0.005,
                                     isTrain=True, dbg=False)
        train_loader = DataLoader(objGoturn,
                                  batch_size=self.hparams.batch_size, shuffle=True,
                                  num_workers=6,
                                  collate_fn=objGoturn.collate)

        return train_loader

    @pl.data_loader
    def val_dataloader(self):
        """validation dataloader"""
        logger.info('===' * 20)
        logger.info('Loading dataset for Validation, please wait...')
        logger.info('===' * 20)

        imagenet_path = self.hparams.imagenet_path
        alov_path = self.hparams.alov_path
        mean_file = None

        manager = Manager()
        objGoturn = GoturnDataloader(imagenet_path, alov_path,
                                     mean_file=mean_file,
                                     images_p=manager.list(),
                                     targets_p=manager.list(),
                                     bboxes_p=manager.list(),
                                     val_ratio=0.005,
                                     isTrain=False, dbg=False)
        val_loader = DataLoader(objGoturn,
                                batch_size=self.hparams.batch_size, shuffle=True,
                                num_workers=6,
                                collate_fn=objGoturn.collate)
        return val_loader

    def forward(self, prev, curr):
        """forward function
        """
        pred_bb = self._model(prev.float(), curr.float())
        return pred_bb

    def vis_images(self, prev, curr, gt_bb, pred_bb, prefix='train'):

        def unnormalize(image, mean):
            image = np.transpose(image, (1, 2, 0)) + mean
            image = image.astype(np.float32)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            return image

        for i in range(0, prev.shape[0]):
            # _mean = np.load(self.hparams.mean_file)
            _mean = np.array([104, 117, 123])
            prev_img = prev[i].cpu().detach().numpy()
            curr_img = curr[i].cpu().detach().numpy()

            prev_img = unnormalize(prev_img, _mean)
            curr_img = unnormalize(curr_img, _mean)

            gt_bb_i = BoundingBox(*gt_bb[i].cpu().detach().numpy().tolist())
            gt_bb_i.unscale(curr_img)
            curr_img = draw.bbox(curr_img, gt_bb_i, color=(255, 0, 255))

            pred_bb_i = BoundingBox(*pred_bb[i].cpu().detach().numpy().tolist())
            pred_bb_i.unscale(curr_img)
            curr_img = draw.bbox(curr_img, pred_bb_i)

            out = np.concatenate((prev_img[np.newaxis, ...], curr_img[np.newaxis, ...]), axis=0)
            out = np.transpose(out, [0, 3, 1, 2])

            self._viz.plot_images_np(out, title='sample_{}'.format(i),
                                     env='goturn_{}'.format(prefix))

    def training_step(self, batch, batch_idx):
        """Training step
        @batch: current batch data
        @batch_idx: current batch index
        """
        curr, prev, gt_bb = batch
        pred_bb = self.forward(prev, curr)
        loss = torch.nn.L1Loss(size_average=False)(pred_bb.float(), gt_bb.float())

        if self.trainer.use_dp:
            loss = loss.unsqueeze(0)

        if self._dbg:
            if batch_idx % 1000 == 0:
                d = {'loss': loss.item()}
                iters = (self.trainer.num_training_batches - 1) * self.current_epoch + batch_idx
                self._viz.plot_curves(d, iters, title='Train', ylabel='train_loss')
            if batch_idx % 1000 == 0:
                self.vis_images(prev, curr, gt_bb, pred_bb)

        tqdm_dict = {'batch_loss': loss}
        output = OrderedDict({'loss': loss,
                              'progress_bar': tqdm_dict,
                              'log': tqdm_dict})
        return output

    def validation_step(self, batch, batch_idx):
        """validation step
        @batch: current batch data
        @batch_idx: current batch index
        """
        curr, prev, gt_bb = batch
        pred_bb = self.forward(prev, curr)
        loss = torch.nn.L1Loss(size_average=False)(pred_bb, gt_bb.float())

        if self.trainer.use_dp:
            loss = loss.unsqueeze(0)

        if self._dbg:
            if batch_idx % 100 == 0:
                d = {'loss': loss.item()}
                iters = (self.trainer.num_val_batches - 1) * self.current_epoch + batch_idx
                self._viz.plot_curves(d, iters, title='Validation', ylabel='val_loss')

            if batch_idx % 1000 == 0:
                self.vis_images(prev, curr, gt_bb, pred_bb, prefix='val')

        tqdm_dict = {'val_loss': loss}
        output = OrderedDict({'val_loss': loss,
                              'progress_bar': tqdm_dict,
                              'log': tqdm_dict})
        return output

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'val_loss': avg_loss}


def get_args():
    """ These are common arguments such as
    1. Path to dataset (imagenet and alov)
    2. Architecture, learning rate, batch size
    3. Optimizers: learning rate, momentum, weight decay, learning step,
    gamma
    4. Seed for reproducibility
    5. save path for the model
    """

    ap = argparse.ArgumentParser(add_help=False,
                                 description='Arguments for training Goturn Tracker')
    ap.add_argument('--gpus', type=int, default=1,
                    help='number of gpus, 0: means no gpu, -1 to use all \
                    gpus, 1 = use one gpu, 2 = use two gpus')

    # Data settings
    ap.add_argument('--imagenet_path', type=str,
                    required=True, help='path to imagenet folder, this \
                    folder shoud have images and gt folder')
    ap.add_argument('--alov_path', type=str,
                    required=True, help='path to ALOV folder, this \
                    folder should have images and gt folder')

    # architecture and hyperparameters
    ap.add_argument('--arch', default='alexnet',
                    choices={'alexnet'}, help='model architecture, \
                    default: alexnet, currently only alexnet is \
                    supported')
    ap.add_argument('--pretrained_model',
                    default='../goturn/models/pretrained/alexnet.pth.tar',
                    help='Path to pretrained model')
    ap.add_argument('--epochs', default=90,
                    type=int, help='number of total epochs to run')
    ap.add_argument('--batch_size', default=3,
                    type=int, help='number of images per batch')

    # Optimizer settings
    ap.add_argument('--lr', default=8e-7, type=float,
                    help='initial learning rate', dest='lr')
    ap.add_argument('--momentum', default=0.9, type=float, help='momentum')
    ap.add_argument('--wd', default=5e-4, type=float, help='weight decay (default: 5e-4)',
                    dest='wd')
    ap.add_argument('--lr_step', default=1, type=int,
                    help='Number of epoch after which we change the learning rate',
                    dest='lr_step')
    ap.add_argument('--gamma', default=0.1, type=float,
                    help='multiplicative factor for learning rate',
                    dest='gamma')

    # reproducibility
    ap.add_argument('--seed', type=int, default=800, help='seed value')

    # save path
    ap.add_argument('--save_path', default=".", type=str, help='path to save output')

    # goturn specific arguments
    ap = GoturnTrain.add_model_specific_args(ap)
    return ap.parse_args()


def read_images_dbg(idx):
    idx = idx + 1
    _mean = np.array([104, 117, 123])
    images = []
    target = []
    bbox = []
    parent_path = '/media/nthere/datasets/goturn_samples/0{}'.format(idx)
    gt_path = '{}/gt.txt'.format(parent_path)
    with open(gt_path) as f:
        for i, line in enumerate(f):
            prev_path = '{}/Image{}_curr.png'.format(parent_path, i)
            curr_path = '{}/Image{}_target.png'.format(parent_path, i)
            prev = cv2.imread(prev_path) - _mean
            prev = np.transpose(prev, axes=(2, 0, 1))
            curr = cv2.imread(curr_path) - _mean
            curr = np.transpose(curr, axes=(2, 0, 1))
            gt = line.strip().split(',')[0:4]
            gt = [float(p) for p in gt]
            images.append(prev)
            target.append(curr)
            bbox.append(gt)

    images = torch.from_numpy(np.stack(images)).to('cuda:0')
    targets = torch.from_numpy(np.stack(target)).to('cuda:0')
    bboxes = torch.from_numpy(np.stack(bbox)).to('cuda:0')
    return images, targets, bboxes


def main(hparams):
    hparams = get_args()
    model = GoturnTrain(hparams, dbg=True)
    # resume_from_ckpt = './caffenet-dbg-2/lightning_logs/version_0/checkpoints/epoch=2.ckpt'
    resume_from_ckpt = None
    trainer = Trainer(default_save_path=hparams.save_path,
                      gpus=[0, ], min_nb_epochs=hparams.epochs,
                      accumulate_grad_batches=1,
                      train_percent_check=1,
                      resume_from_checkpoint=resume_from_ckpt,
                      val_percent_check=1, profiler=True)
    # model.find_lr()
    trainer.fit(model)


if __name__ == "__main__":
    main(get_args())
