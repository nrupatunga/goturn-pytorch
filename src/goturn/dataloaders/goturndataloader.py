"""
File: goturndataloader.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: goturn dataloader
"""

import math
import sys
from multiprocessing import Manager
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from goturn.dataloaders.alov import AlovDataset
    from goturn.dataloaders.imagenet import ImageNetDataset
    from goturn.dataloaders.sampler import sample_generator
    from goturn.helper.image_io import resize
except ImportError:
    logger.error('Please run $source settings.sh from root directory')
    sys.exit(1)


class GoturnDataloader(Dataset):
    """Docstring for goturnDataloader. """

    def __init__(self, imagenet_path, alov_path, mean_file=None, isTrain=True,
                 val_ratio=0.005, width=227, height=227,
                 images_p=None, targets_p=None, bboxes_p=None,
                 dbg=False):
        """Dataloader initialization for goturn tracker """

        # ALOV
        img_dir = Path(imagenet_path).joinpath('images')
        ann_dir = Path(imagenet_path).joinpath('gt')
        self._imagenetD = ImageNetDataset(str(img_dir), str(ann_dir),
                                          isTrain=isTrain,
                                          val_ratio=val_ratio)

        # Imagenet
        img_dir = Path(alov_path).joinpath('images')
        ann_dir = Path(alov_path).joinpath('gt')
        self._alovD = AlovDataset(str(img_dir), str(ann_dir),
                                  isTrain=isTrain, val_ratio=val_ratio)

        # sample generator
        self._sample_gen = sample_generator(5, 15, -0.4, 0.4, dbg=dbg)
        self._kGenExPerImage = 10

        self._images = []
        self._targets = []
        self._bboxes = []

        # from prev batch
        self._images_p = images_p
        self._targets_p = targets_p
        self._bboxes_p = bboxes_p

        self._width = width
        self._height = height
        if mean_file:
            self._mean = np.load(mean_file)
        else:
            self._mean = np.array([104, 117, 123])

        self._minDataLen = min(len(self._imagenetD), len(self._alovD))
        self._maxDataLen = max(len(self._imagenetD), len(self._alovD))
        self._batchSize = 50

    def __len__(self):
        ''' length of the total dataset, is max of one of the dataset '''
        return int((self._maxDataLen + self._minDataLen) * 1.5)

    def __getitem__(self, idx):
        """Get the current idx data
        @idx: Current index for the data
        """
        imagenet_pair = self._imagenetD[idx % len(self._imagenetD)]
        alov_pair = self._alovD[idx % len(self._alovD)]

        return imagenet_pair, alov_pair

    def collate(self, batch):
        ''' Custom data collation for alov and imagenet
        @batch: batch of data
        '''
        self._images = []
        self._targets = []
        self._bboxes = []

        count = 0
        for i, batch_i in enumerate(batch):
            for i, (img_prev, bbox_prev, img_cur, bbox_cur) in enumerate(batch_i):
                if count == 5:
                    break
                self._sample_gen.reset(bbox_cur, bbox_prev, img_cur,
                                       img_prev)
                self.__make_training_samples()
                count = count + 1

        num_prev_batch = len(self._images_p)
        num_curr_batch = self._batchSize - num_prev_batch

        last_idx = (self._kGenExPerImage + 1) * math.ceil(num_curr_batch / (self._kGenExPerImage + 1))

        self._images_c = self._images[0:num_curr_batch]
        self._targets_c = self._targets[0:num_curr_batch]
        self._bboxes_c = self._bboxes[0:num_curr_batch]

        batch_images, batch_targets, batch_boxes = self._images, self._targets, self._bboxes
        if num_prev_batch != 0:
            self._images = self._images_p[:]
            self._targets = self._targets_p[:]
            self._bboxes = self._bboxes_p[:]
            self._images_p[:] = []
            self._targets_p[:] = []
            self._bboxes_p[:] = []
        else:
            self._images = []
            self._targets = []
            self._bboxes = []

        self._images_p.extend(batch_images[num_curr_batch:last_idx])
        self._targets_p.extend(batch_targets[num_curr_batch:last_idx])
        self._bboxes_p.extend(batch_boxes[num_curr_batch:last_idx])

        self._images.extend(self._images_c)
        self._targets.extend(self._targets_c)
        self._bboxes.extend(self._bboxes_c)

        _images = []
        _targets = []
        _bboxes = []

        for i, (im, tar, bbox) in enumerate(zip(self._images,
                                                self._targets,
                                                self._bboxes)):

            im = resize(im, (self._width, self._height)) - self._mean
            _images.append(np.transpose(im, axes=(2, 0, 1)))

            tar = resize(tar, (self._width, self._height)) - self._mean
            _targets.append(np.transpose(tar, axes=(2, 0, 1)))

            _bboxes.append(np.array([bbox.x1, bbox.y1, bbox.x2,
                                     bbox.y2]))

        images = torch.from_numpy(np.stack(_images))
        targets = torch.from_numpy(np.stack(_targets))
        bboxes = torch.from_numpy(np.stack(_bboxes))

        return images, targets, bboxes

    def __make_training_samples(self):
        """
        1. First decide the current search region, which is
        kContextFactor(=2) * current bounding box.
        2. Crop the valid search region and copy to the new padded image
        3. Recenter the actual bounding box of the object to the new
        padded image
        4. Scale the bounding box for regression
        """

        sample_gen = self._sample_gen
        images = self._images
        targets = self._targets
        bboxes = self._bboxes

        image, target, bbox_gt_scaled = sample_gen.make_true_sample()

        images.append(image)
        targets.append(target)
        bboxes.append(bbox_gt_scaled)

        # Generate more number of examples
        images, targets, bbox_gt_scaled = sample_gen.make_training_samples(self._kGenExPerImage, images, targets, bboxes)


if __name__ == "__main__":
    imagenet_path = '/media/nthere/datasets/ISLVRC2014_Det/'
    alov_path = '/media/nthere/datasets/ALOV/'
    manager = Manager()
    objGoturn = GoturnDataloader(imagenet_path, alov_path,
                                 images_p=manager.list(),
                                 targets_p=manager.list(),
                                 bboxes_p=manager.list(),
                                 isTrain=True, dbg=True)

    dataloader = DataLoader(objGoturn, batch_size=3, shuffle=True,
                            num_workers=6, collate_fn=objGoturn.collate)
    for i, *data in tqdm(enumerate(dataloader), desc='Loading imagenet/alov', total=len(objGoturn) / 3, unit='files'):
        pass
