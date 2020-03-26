"""
File: sample_generator.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: Generating samples from single frame
"""
import sys

import cv2
import numpy as np

from loguru import logger

try:
    from goturn.helper.BoundingBox import BoundingBox
    from goturn.helper.image_proc import cropPadImage
    from goturn.helper.vis_utils import Visualizer
    from goturn.helper import image_io
    from goturn.helper.draw_util import draw
except ImportError:
    logger.error('Please run $source settings.sh from root directory')
    sys.exit(1)


class bbParams:

    """Docstring for bbParams. """

    def __init__(self, lamda_shift, lamda_scale, min_scale, max_scale):
        """parameters for generating synthetic data"""
        self.lamda_shift = lamda_shift
        self.lamda_scale = lamda_scale
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __repr__(self):
        return str({'lamda_shift': self.lamda_shift, 'lamda_scale':
                    self.lamda_scale, 'min_scale': self.min_scale,
                    'max_scale': self.max_scale})


class sample_generator:

    """Generate samples from single frame"""

    def __init__(self, lamda_shift, lamda_scale, min_scale, max_scale,
                 dbg=False, env='sample_generator'):
        """set parameters """

        self._lamda_shift = lamda_shift
        self._lamda_scale = lamda_scale
        self._min_scale = min_scale
        self._max_scale = max_scale
        self._kSamplesPerImage = 10  # number of synthetic samples per image

        self._viz = None
        if dbg:
            self._env = env
            self._viz = Visualizer(env=self._env)

        self._dbg = dbg

    def make_true_sample(self):
        """Generate true target:search_region pair"""

        curr_prior_tight = self.bbox_prev_gt_
        target_pad = self.target_pad_
        # To find out the region in which we need to search in the
        # current frame, we use the previous frame bbox to get the
        # region in which we can make the search
        output = cropPadImage(curr_prior_tight, self.img_curr_,
                              self._dbg, self._viz)
        curr_search_region, curr_search_location, edge_spacing_x, edge_spacing_y = output

        bbox_curr_gt = self.bbox_curr_gt_
        bbox_curr_gt_recentered = BoundingBox(0, 0, 0, 0)
        bbox_curr_gt_recentered = bbox_curr_gt.recenter(curr_search_location, edge_spacing_x, edge_spacing_y, bbox_curr_gt_recentered)

        if self._dbg:
            env = self._env + '_make_true_sample'
            search_dbg = draw.bbox(self.img_curr_, curr_search_location)
            search_dbg = draw.bbox(search_dbg, bbox_curr_gt, color=(255, 255, 0))
            self._viz.plot_image_opencv(search_dbg, 'search_region', env=env)

            recentered_img = draw.bbox(curr_search_region,
                                       bbox_curr_gt_recentered,
                                       color=(255, 255, 0))
            self._viz.plot_image_opencv(recentered_img,
                                        'cropped_search_region', env=env)
            del recentered_img
            del search_dbg

        bbox_curr_gt_recentered.scale(curr_search_region)

        return curr_search_region, target_pad, bbox_curr_gt_recentered

    def make_training_samples(self, num_samples, images, targets, bbox_gt_scales):
        """
        @num_samples: number of samples
        @images: set of num_samples appended to images list
        @target: set of num_samples targets appended to targets list
        @bbox_gt_scales: bounding box to be regressed (scaled version)
        """
        for i in range(num_samples):
            image_rand_focus, target_pad, bbox_gt_scaled = self.make_training_sample_BBShift()
            images.append(image_rand_focus)
            targets.append(target_pad)
            bbox_gt_scales.append(bbox_gt_scaled)
            if self._dbg:
                self.visualize(image_rand_focus, target_pad, bbox_gt_scaled, i)

        return images, targets, bbox_gt_scales

    def visualize(self, image, target, bbox, idx):
        """
        sample generator prepares image and the respective targets (with
        bounding box). This function helps you to visualize it.

        The visualization is based on the Visdom server, please
        initialize the visdom server by running the command
        $ python -m visdom.server
        open http://localhost:8097 in your browser to visualize the
        images
        """

        if image_io._is_pil_image(image):
            image = np.asarray(image)

        if image_io._is_pil_image(target):
            target = np.asarray(target)

        target = cv2.resize(target, (227, 227))
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (227, 227))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bbox.unscale(image)
        bbox.x1, bbox.x2, bbox.y1, bbox.y2 = int(bbox.x1), int(bbox.x2), int(bbox.y1), int(bbox.y2)

        image_bb = draw.bbox(image, bbox)
        out = np.concatenate((target[np.newaxis, ...], image_bb[np.newaxis, ...]), axis=0)
        out = np.transpose(out, [0, 3, 1, 2])
        self._viz.plot_images_np(out, title='sample_{}'.format(idx),
                                 env=self._env + '_train')

    def get_default_bb_params(self):
        """default bb parameters"""
        default_params = bbParams(self._lamda_shift, self._lamda_scale,
                                  self._min_scale, self._max_scale)
        return default_params

    def make_training_sample_BBShift_(self, bbParams, dbg=False):
        """generate training samples based on bbparams"""

        bbox_curr_gt = self.bbox_curr_gt_
        bbox_curr_shift = BoundingBox(0, 0, 0, 0)
        bbox_curr_shift = bbox_curr_gt.shift(self.img_curr_, bbParams.lamda_scale, bbParams.lamda_shift, bbParams.min_scale, bbParams.max_scale, True, bbox_curr_shift)
        rand_search_region, rand_search_location, edge_spacing_x, edge_spacing_y = cropPadImage(bbox_curr_shift, self.img_curr_,
                                                                                                dbg=self._dbg, viz=self._viz)

        bbox_curr_gt = self.bbox_curr_gt_
        bbox_gt_recentered = BoundingBox(0, 0, 0, 0)
        bbox_gt_recentered = bbox_curr_gt.recenter(rand_search_location, edge_spacing_x, edge_spacing_y, bbox_gt_recentered)

        if dbg:
            env = self._env + '_make_training_sample_bbshift'
            viz = self._viz
            curr_img_bbox = draw.bbox(self.img_curr_,
                                      bbox_curr_gt)
            recentered_img = draw.bbox(rand_search_region,
                                       bbox_gt_recentered)

            viz.plot_image_opencv(curr_img_bbox, 'curr shifted bbox', env=env)
            viz.plot_image_opencv(recentered_img, 'recentered shifted bbox', env=env)

        bbox_gt_recentered.scale(rand_search_region)
        bbox_gt_scaled = bbox_gt_recentered

        return rand_search_region, self.target_pad_, bbox_gt_scaled

    def make_training_sample_BBShift(self):
        """
        bb_params consists of shift, scale, min-max scale for shifting
        the current bounding box
        """
        default_bb_params = self.get_default_bb_params()
        image_rand_focus, target_pad, bbox_gt_scaled = self.make_training_sample_BBShift_(default_bb_params, self._dbg)

        return image_rand_focus, target_pad, bbox_gt_scaled

    def reset(self, bbox_curr, bbox_prev, img_curr, img_prev):
        """This prepares the target image with enough context (search
        region)
        @bbox_curr: current frame bounding box
        @bbox_prev: previous frame bounding box
        @img_curr: current frame
        @img_prev: previous frame
        """

        target_pad, pad_image_location, _, _ = cropPadImage(bbox_prev,
                                                            img_prev, dbg=self._dbg, viz=self._viz)
        self.img_curr_ = img_curr
        self.bbox_curr_gt_ = bbox_curr
        self.bbox_prev_gt_ = bbox_prev
        self.target_pad_ = target_pad  # crop kContextFactor * bbox_curr copied

        if self._dbg:
            env = self._env + '_targetpad'
            search_dbg = draw.bbox(img_prev, bbox_prev, color=(0, 0, 255))
            search_dbg = draw.bbox(search_dbg, pad_image_location)
            self._viz.plot_image_opencv(search_dbg, 'target_region', env=env)
            self._viz.plot_image_opencv(target_pad,
                                        'cropped_target_region', env=env)
            del search_dbg
