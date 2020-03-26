"""
File: imagenet.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: loading imagenet dataset
"""

import itertools
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from loguru import logger

try:
    from goturn.helper import image_io
    from goturn.helper.annotation import annotation
    from goturn.helper.vis_utils import Visualizer
    from goturn.dataloaders.sampler import sample_generator
except ImportError:
    logger.error('Please run $source settings.sh from root directory')
    sys.exit(1)


class ImageNetDataset:

    def __init__(self, imgs_dir, ann_dir, isTrain=True, val_ratio=0.2, dbg=False):
        '''
        loading images and annotation from imagenet
        @imgs_dir: images path
        @ann_dir: annotations path
        @isTrain: True: Training, False: validation
        @val_ratio: validation data ratio
        @dbg: For visualization
        '''

        if not Path(imgs_dir).is_dir():
            logger.error('{} is not a valid directory'.format(imgs_dir))

        self._imgs_dir = Path(imgs_dir)
        self._ann_dir = Path(ann_dir)
        self._kMaxRatio = 0.66
        self._list_of_annotations = self.__loadImageNetDet(isTrain=isTrain, val_ratio=val_ratio)
        self._data_fetched = []  # for debug purposes
        assert len(self._list_of_annotations) > 0, 'Number of valid annotations is {}'.format(len(self._list_of_annotations))

        self._dbg = dbg
        if dbg:
            self._env = 'ImageNet'
            self._viz = Visualizer(env=self._env)

    def __len__(self):
        ''' number of valid annotations that is fetched in
        load_annotation_file'''

        return len(self._list_of_annotations)

    def __getitem__(self, idx):
        """Get the current idx data
        @idx: Current index for the data
        """

        # set the dbg to true, to see if all the images are fetched
        # atleast once during the entire epoch
        image, bbox = self.__load_annotation(idx)

        # For imagenet, prev frame and current frame are the same
        return image, bbox, image, bbox

    def __loadImageNetDet(self, isTrain=True, val_ratio=0.2):
        ''' Loads all the image annotations files '''

        subdirs = [x.parts[-1] for x in self._imgs_dir.iterdir() if x.is_dir()]
        train_ratio = 1 - val_ratio
        num_end = int(train_ratio * len(subdirs))

        if isTrain:
            subdirs = subdirs[0:num_end]
        else:
            subdirs = subdirs[num_end:]

        self._subdirs = subdirs
        num_annotations = 0
        list_of_annotations_out = []

        for i, subdir in enumerate(subdirs):
            ann_files = self._ann_dir.joinpath(subdir).glob('*.xml')
            logger.info('Loading {}/{} - annotation file from folder = {}'.format(i + 1, len(subdirs), subdir))
            for ann in ann_files:
                list_of_annotations, num_ann_curr = self.__load_annotation_file(ann)
                num_annotations = num_annotations + num_ann_curr
                if len(list_of_annotations) == 0:
                    continue
                list_of_annotations_out.append(list_of_annotations)

        all_annotations = list(itertools.chain.from_iterable(list_of_annotations_out))
        # random.shuffle(all_annotations)

        logger.info('+' * 60)
        logger.info("Found {} annotations from {} images"
                    " ({:.2f} annotations/image)".format(num_annotations,
                                                         len(list_of_annotations_out),
                                                         (num_annotations / len(list_of_annotations_out))))

        return all_annotations

    def __load_annotation_file(self, annotation_file):
        """ Loads the bounding box annotations in xml file
        @annotation_file: annotation file (.xml), which contains
        bounding box information
        """

        list_of_annotations = []
        num_annotations = 0
        root = ET.parse(annotation_file).getroot()
        folder = root.find('folder').text
        filename = root.find('filename').text
        size = root.find('size')
        disp_width = int(size.find('width').text)
        disp_height = int(size.find('height').text)

        for obj in root.findall('object'):
            bbox = obj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            xmax = int(bbox.find('xmax').text)
            ymin = int(bbox.find('ymin').text)
            ymax = int(bbox.find('ymax').text)

            width = xmax - xmin
            height = ymax - ymin

            kMaxRatio = self._kMaxRatio
            if width > (kMaxRatio * disp_width) or height > (kMaxRatio * disp_height):
                continue

            if ((xmin < 0) or (ymin < 0) or (xmax <= xmin) or (ymax <= ymin)):
                continue

            objAnnotation = annotation()
            objAnnotation.setbbox(xmin, xmax, ymin, ymax)
            objAnnotation.setWidthHeight(disp_width, disp_height)
            objAnnotation.setImagePath(Path(folder).joinpath(filename))
            list_of_annotations.append(objAnnotation)
            num_annotations = num_annotations + 1

        return list_of_annotations, num_annotations

    def __load_annotation(self, idx):
        """
        this loads the image and its corresponding bounding box gt
        @idx: current image number to be fetched
        """

        random_ann = self._list_of_annotations[idx]

        img_path = self._imgs_dir.joinpath(random_ann.image_path.with_suffix('.JPEG'))

        image = image_io.load(img_path)
        image = np.asarray(image, dtype=np.uint8)

        img_height = image.shape[0]
        img_width = image.shape[1]

        sc_factor_1 = 1.0
        if img_height != random_ann.disp_height or img_width != random_ann.disp_width:
            logger.info('Image Number = {}, Image file = {}'.format(idx, img_path))
            logger.info('Image Size = {} x {}'.format(img_width, img_height))
            logger.info('Display Size = {} x {}'.format(random_ann.disp_width, random_ann.disp_height))

            sc_factor_1 = (img_height * 1.) / random_ann.disp_height
            sc_factor_2 = (img_width * 1.) / random_ann.disp_width

            logger.info('Factor: {} {}'.format(sc_factor_1, sc_factor_2))

        bbox = random_ann.bbox
        bbox.x1 = bbox.x1 * sc_factor_1
        bbox.x2 = bbox.x2 * sc_factor_1
        bbox.y1 = bbox.y1 * sc_factor_1
        bbox.y2 = bbox.y2 * sc_factor_1

        if self._dbg:
            self._data_fetched.append((img_path.name, bbox.x1, bbox.y1,
                                       bbox.x2, bbox.y2))

        # return image_io.image_to_tensor(image), bbox
        return image, bbox


def make_training_examples(sample_gen):
    """
    1. First decide the current search region, which is
    kContextFactor(=2) * current bounding box.
    2. Crop the valid search region and copy to the new padded image
    3. Recenter the actual bounding box of the object to the new
    padded image
    4. Scale the bounding box for regression
    """
    images = []
    targets = []
    bboxes = []

    image, target, bbox_gt_scaled = sample_gen.make_true_sample()

    images.append(image)
    targets.append(target)
    bboxes.append(bbox_gt_scaled)

    # Generate more number of examples
    images, targets, bbox_gt_scaled = sample_gen.make_training_samples(10, images, targets, bboxes)


if __name__ == "__main__":
    imagnet_path = '/media/nthere/datasets/ISLVRC2014_Det/dummy/'
    img_dir = Path(imagnet_path).joinpath('images')
    ann_dir = Path(imagnet_path).joinpath('gt')
    imagenetD = ImageNetDataset(str(img_dir), str(ann_dir),
                                isTrain=False, val_ratio=0, dbg=True)

    for i, (image, bbox, image, bbox) in enumerate(imagenetD):
        pass

    sample_gen = sample_generator(5, 15, -0.4, 0.4, dbg=True, env=imagenetD._env)
    for i, (image, bbox, image, bbox) in enumerate(imagenetD):
        sample_gen.reset(bbox, bbox, image, image)
        make_training_examples(sample_gen)
