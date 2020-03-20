"""
File: alov.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: Alov dataloader
"""

import sys
from pathlib import Path

import numpy as np
from loguru import logger
from tqdm import tqdm

try:
    from goturn.helper.BoundingBox import BoundingBox
    from goturn.helper.video import video, frame
    from goturn.helper.vis_utils import Visualizer
    from goturn.helper.draw_util import draw
    from goturn.helper import image_io
    from goturn.dataloaders.sampler import sample_generator
except ImportError:
    logger.error('Please run $source settings.sh from root directory')
    sys.exit(1)


class AlovDataset:

    """Docstring for alov. """

    def __init__(self, imgs_dir, ann_dir, isTrain=True, val_ratio=0.2, dbg=False):
        '''
        loading video frames and annotation from alov
        @imgs_dir: alov video frames directory
        @ann_dir: annotations path
        @isTrain: True: Training, False: validation
        @val_ratio: validation data ratio
        @dbg: For visualization
        '''

        if not Path(imgs_dir).is_dir():
            logger.error('{} is not a valid directory'.format(imgs_dir))

        self._imgs_dir = Path(imgs_dir)
        self._ann_dir = Path(ann_dir)

        self._cats = {}
        self._isTrain = isTrain
        self._val_ratio = val_ratio

        self.__loaderAlov()
        self._alov_imgpairs = []
        self._alov_vids = self.__get_videos(self._isTrain, self._val_ratio)
        self.__parse_all()  # get all the image pairs in a list
        self._dbg = dbg
        if dbg:
            self._env = 'Alov'
            self._viz = Visualizer(env=self._env)

    def __loaderAlov(self):
        """Load annotations from the imgs_dir"""

        subdirs = sorted([x.parts[-1] for x in self._imgs_dir.iterdir() if x.is_dir()])
        for i, sub_dir in enumerate(subdirs):
            ann_files = sorted(self._ann_dir.joinpath(sub_dir).glob('*.ann'))
            logger.info('Loading {}/{} - annotation file from folder = {}'.format(i + 1, len(subdirs), sub_dir))

            for ann_file in ann_files:
                self.__load_annotation_file(sub_dir, ann_file)

    def __len__(self):
        ''' number of valid annotations that is fetched in '''
        return len(self._alov_imgpairs)

    def __getimage_info(self, vid, ann_idx):

        ann_frame = vid._annotations[ann_idx]
        frame_num = ann_frame._frame_num
        img_path = vid._all_frames[frame_num]
        bbox = ann_frame._bbox

        return img_path, bbox

    def __parse_all(self):
        '''Parse all the videos and the respective annotations. and
        build one single list of image annotation pairs'''

        len_vids = len(self._alov_vids)
        for i in range(len_vids):
            vid = self._alov_vids[i]
            anns = vid._annotations
            if len(anns) < 2:
                continue

            for j in range(len(anns) - 1):
                img1_p, bbox1 = self.__getimage_info(vid, j)
                img2_p, bbox2 = self.__getimage_info(vid, j + 1)
                self._alov_imgpairs.append([img1_p, bbox1, img2_p,
                                            bbox2])

    def __getitem__(self, idx):
        """Get the current idx data
        @idx: Current index for the data
        """
        prev_imgpath, bbox_prev, curr_imgpath, bbox_curr = self._alov_imgpairs[idx]
        image_prev = image_io.load(prev_imgpath)
        image_prev = np.asarray(image_prev, dtype=np.uint8)
        image_curr = image_io.load(curr_imgpath)
        image_curr = np.asarray(image_curr, dtype=np.uint8)

        if self._dbg:
            viz, env = self._viz, self._env
            prev_img_bbox = draw.bbox(image_prev, bbox_prev)
            curr_img_bbox = draw.bbox(image_curr, bbox_curr)
            viz.plot_image_opencv(prev_img_bbox, 'prev', env=env)
            viz.plot_image_opencv(curr_img_bbox, 'current', env=env)

            del prev_img_bbox
            del curr_img_bbox

        return image_prev, bbox_prev, image_curr, bbox_curr

    def __load_annotation_file(self, alov_sub_dir, ann_file, ext='jpg'):
        '''
        @alov_sub_dir: subdirectory of images directory
        @ann_file: annotation file to fetch the bounding boxes from
        @ext: image extension
        '''

        vid_path = self._imgs_dir.joinpath(alov_sub_dir).joinpath(ann_file.stem)
        all_frames = vid_path.glob('*.{}'.format(ext))

        objVid = video(vid_path)
        objVid._all_frames = sorted(all_frames)

        with open(ann_file) as f:
            data = f.read().rstrip().split('\n')
            for bb in data:
                frame_num, ax, ay, bx, by, cx, cy, dx, dy = [float(i) for i in bb.split()]
                frame_num = int(frame_num)

                x1 = min(ax, min(bx, min(cx, dx))) - 1
                y1 = min(ay, min(by, min(cy, dy))) - 1
                x2 = max(ax, max(bx, max(cx, dx))) - 1
                y2 = max(ay, max(by, max(cy, dy))) - 1

                bbox = BoundingBox(x1, y1, x2, y2)
                objFrame = frame(frame_num - 1, bbox)
                objVid._annotations.append(objFrame)

        if alov_sub_dir not in self._cats.keys():
            self._cats[alov_sub_dir] = []

        self._cats[alov_sub_dir].append(objVid)

    def __get_videos(self, isTrain, val_ratio):
        """
        @isTrain: train mode or test mode
        @val_ratio: ratio of val
        """

        videos = []
        num_categories = len(self._cats)
        cats = self._cats
        keys = sorted(cats.keys())
        for i in range(num_categories):
            cat_video = cats[keys[i]]
            num_videos = len(cat_video)
            num_val = max(1, int(val_ratio * num_videos))
            num_train = num_videos - num_val

            if isTrain:
                start_num = 0
                end_num = num_train - 1
            else:
                start_num = num_train
                end_num = num_videos - 1

            for i in range(start_num, end_num + 1):
                video = cat_video[i]
                videos.append(video)

        return videos


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

    # This is the true frames from the video, not synthetically
    # generated unlike make_training_samples. ground truths come from
    # video
    image, target, bbox_gt_scaled = sample_gen.make_true_sample()

    images.append(image)
    targets.append(target)
    bboxes.append(bbox_gt_scaled)

    # Generate more number of examples
    images, targets, bbox_gt_scaled = sample_gen.make_training_samples(10, images, targets, bboxes)


if __name__ == "__main__":
    alovDir = '/Users/nrupatunga/2020/dataset/alov/'
    img_dir = Path(alovDir).joinpath('images')
    ann_dir = Path(alovDir).joinpath('gt')
    alovD = AlovDataset(str(img_dir), str(ann_dir), isTrain=False,
                        dbg=True)

    sample_gen = sample_generator(5, 15, -0.4, 0.4, dbg=True,
                                  env=alovD._env)
    for i, (prev, prev_bbox, curr, curr_bbox) in tqdm(enumerate(alovD)):
        sample_gen.reset(curr_bbox, prev_bbox, curr, prev)
        make_training_examples(sample_gen)
