"""
File: demo.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: testing goturn tracking

Acknowledgement: Code is adopted from the following sources
1. https://github.com/opencv/opencv/blob/master/samples/python/mosse.py
2. https://www.pyimagesearch.com/2017/02/06/faster-video-file-fps-with-cv2-videocapture-and-opencv/
"""

import argparse
import sys
from pathlib import Path

import cv2
import imutils
import numpy as np
import torch
from imutils.video import FileVideoStream
from loguru import logger

from train import GoturnTrain

try:
    from goturn.helper.vis_utils import Visualizer
    from goturn.helper.image_proc import cropPadImage
    from goturn.helper.BoundingBox import BoundingBox
    from goturn.helper.image_io import resize, image_to_tensor
except ImportError:
    logger.error('Please run $source settings.sh from root directory')
    sys.exit(1)


class RectSelector:

    def __init__(self, win, callback):
        self.win = win
        self.callback = callback
        cv2.setMouseCallback(win, self.onmouse)
        self.drag_start = None
        self.drag_rect = None

    def onmouse(self, event, x, y, flags, param):

        x, y = np.int16([x, y])
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            return

        if self.drag_start:
            if flags & cv2.EVENT_FLAG_LBUTTON:
                xo, yo = self.drag_start
                x0, y0 = np.minimum([xo, yo], [x, y])
                x1, y1 = np.maximum([xo, yo], [x, y])
                self.drag_rect = None
                if x1 - x0 > 0 and y1 - y0 > 0:
                    self.drag_rect = (x0, y0, x1, y1)
            else:
                rect = self.drag_rect
                self.drag_start = None
                self.drag_rect = None
                if rect:
                    self.callback(rect)

    def draw(self, vis):
        if not self.drag_rect:
            return False
        x0, y0, x1, y1 = self.drag_rect
        cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 2)
        return True

    @property
    def dragging(self):
        return self.drag_rect is not None


class App:

    """Create a goturn demo app"""

    def __init__(self, vid_src, model, paused=False, dbg=False):
        """Initialize video src
        @vid_src: video path
        @paused: video paused
        """
        self._fvs = FileVideoStream(vid_src).start()
        frame = self._fvs.read()
        self._frame = imutils.resize(frame, width=227, height=227)
        self._frame = frame
        self._model = model
        self._paused = paused
        self._pred_bb = None

        cv2.imshow('frame', self._frame)
        self._rect_sel = RectSelector('frame', self.onrect)

        if dbg:
            self._viz = Visualizer()

        self._dbg = dbg

    def onrect(self, rect):
        """on rectangle selected callback
        """
        prev_frame = self._frame
        prev_frame = imutils.resize(prev_frame, width=227, height=227)
        # curr_frame = self._frame
        curr_frame = self._fvs.read()
        curr_frame = imutils.resize(curr_frame, width=227, height=227)
        self._pred_bb = self.track(curr_frame, prev_frame, rect)
        print(self._pred_bb)
        if self._dbg:
            x1, y1, x2, y2 = self._pred_bb
            curr_frame_dbg = np.copy(curr_frame)
            curr_frame_dbg = cv2.rectangle(curr_frame_dbg, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            self._viz.plot_image_opencv(curr_frame_dbg, title='tracked')

    def vis_images(self, prev, curr, gt_bb, pred_bb):
        for i in range(0, prev.shape[0]):
            _mean = np.array([104, 117, 123])
            prev_img = prev[i].cpu().detach().numpy() * 255
            curr_img = curr[i].cpu().detach().numpy() * 255

            prev_img = np.transpose(prev_img, (1, 2, 0)) + _mean
            curr_img = np.transpose(curr_img, (1, 2, 0)) + _mean

            gt_img = BoundingBox(*gt_bb[i].cpu().detach().numpy().tolist())
            gt_img.unscale(curr_img)
            x1, y1, x2, y2 = int(gt_img.x1), int(gt_img.y1), int(gt_img.x2), int(gt_img.y2)
            prev_img = cv2.rectangle(cv2.UMat(prev_img), (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

            pred_img = BoundingBox(*pred_bb[i].cpu().detach().numpy().tolist())
            pred_img.unscale(prev_img.get())
            x1, y1, x2, y2 = int(pred_img.x1), int(pred_img.y1), int(pred_img.x2), int(pred_img.y2)
            curr_img = cv2.rectangle(cv2.UMat(curr_img), (int(x1), int(y1)),
                                     (int(x2), int(y2)), (255, 255, 0), 2)

            self._viz.plot_image_opencv(prev_img.get(),
                                        title='prev_img_{}'.format(i))
            self._viz.plot_image_opencv(curr_img.get(),
                                        title='curr_img_with_bb_{}'.format(i))

    def track(self, curr_frame, prev_frame, rect):
        """track current frame
        @curr_frame: current frame
        @prev_frame: prev frame
        @rect: bounding box of previous frame
        """
        __import__('pdb').set_trace()
        prev_bbox = BoundingBox(rect[0], rect[1],
                                rect[2], rect[3])
        target_pad, _, _, _ = cropPadImage(prev_bbox, prev_frame)
        cur_search_region, search_location, edge_spacing_x, edge_spacing_y = cropPadImage(prev_bbox, curr_frame)

        if self._dbg:
            self._viz.plot_image_opencv(target_pad, 'target')
            self._viz.plot_image_opencv(cur_search_region, 'current')

        target_pad_in = self.preprocess(target_pad).unsqueeze(0)
        cur_search_region_in = self.preprocess(cur_search_region).unsqueeze(0)
        pred_bb = self._model.forward(target_pad_in,
                                      cur_search_region_in)
        if self._dbg:
            prev_bbox.scale(prev_frame)
            x1, y1, x2, y2 = prev_bbox.x1, prev_bbox.y1, prev_bbox.x2, prev_bbox.y2
            prev_bbox = torch.tensor([x1, y1, x2, y2]).unsqueeze(0)
            target_dbg = target_pad_in.clone()
            cur_search_region_dbg = cur_search_region_in.clone()
            self.vis_images(target_dbg,
                            cur_search_region_dbg, prev_bbox, pred_bb)

        pred_bb = BoundingBox(*pred_bb[0].cpu().detach().numpy().tolist())
        pred_bb.unscale(cur_search_region)
        pred_bb.uncenter(curr_frame, search_location, edge_spacing_x, edge_spacing_y)
        x1, y1, x2, y2 = int(pred_bb.x1), int(pred_bb.y1), int(pred_bb.x2), int(pred_bb.y2)
        pred_bb = [x1, y1, x2, y2]
        return pred_bb

    def run(self):
        """run tracker
        """
        while self._fvs.more():
            if not self._paused:
                prev_frame = self._frame
                curr_frame = self._fvs.read()
                curr_frame = imutils.resize(curr_frame, width=227, height=227)
                self._frame = curr_frame
                if self._pred_bb:
                    self._pred_bb = self.track(curr_frame, prev_frame, self._pred_bb)
                    self._rect_sel.drag_rect = self._pred_bb
                    if self._dbg:
                        x1, y1, x2, y2 = self._pred_bb
                        curr_frame_dbg = np.copy(curr_frame)
                        curr_frame_dbg = cv2.rectangle(curr_frame_dbg, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                        self._viz.plot_image_opencv(curr_frame_dbg, title='tracked')

            vis = self._frame.copy()
            self._rect_sel.draw(vis)

            cv2.imshow('frame', vis)
            ch = cv2.waitKey(10)
            if ch == 27:
                break

            if ch == ord(' '):
                self._paused = not self._paused

            if ch == ord('c'):
                self._trackers = []

    def preprocess(self, im):
        """preprocess image before forward pass, this is the same
        preprocessing used during training, please refer to collate function
        in train.py for reference
        @image: input image
        """
        mean = np.array([104, 117, 123])
        w, h = 227, 227
        im = resize(im, (w, h)) - mean
        im = image_to_tensor(im)
        im = im / 255.

        return im


def main(args):
    """Testing goturn tracker
    """
    model_dir = Path(args.model_dir)
    # Checkpoint path
    ckpt_dir = model_dir.joinpath('checkpoints')
    ckpt_path = next(ckpt_dir.glob('*.ckpt'))

    model = GoturnTrain.load_from_metrics(weights_path=ckpt_path,
                                          tags_csv=model_dir.joinpath('meta_tags.csv'))
    model.eval()
    model.freeze()

    vid_path = args.video_path
    App(vid_path, model, dbg=True).run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--video_path',
                    required=True, help='Path to the input video files')
    ap.add_argument('--model_dir',
                    required=True, help='model directory')

    args = ap.parse_args()
    main(args)
