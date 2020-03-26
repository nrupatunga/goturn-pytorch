"""
File: test.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: testing goturn tracker
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from loguru import logger
from train import GoturnTrain

try:
    from goturn.dataloaders.goturndataloader import GoturnDataloader
    from goturn.helper.vis_utils import Visualizer
    from goturn.helper.BoundingBox import BoundingBox
except ImportError:
    logger.error('Please run $source settings.sh from root directory')
    sys.exit(1)


def vis_images(prev, curr, gt_bb, pred_bb, viz=None):
    for i in range(0, prev.shape[0]):
        _mean = np.array([104, 117, 123])
        prev_img = prev[i].cpu().detach().numpy() * 255
        curr_img = curr[i].cpu().detach().numpy() * 255

        prev_img = np.transpose(prev_img, (1, 2, 0)) + _mean
        curr_img = np.transpose(curr_img, (1, 2, 0)) + _mean

        gt_img = BoundingBox(*gt_bb[i].cpu().detach().numpy().tolist())
        gt_img.unscale(curr_img)
        x1, y1, x2, y2 = int(gt_img.x1), int(gt_img.y1), int(gt_img.x2), int(gt_img.y2)
        prev_img = cv2.UMat(prev_img)
        # prev_img = cv2.rectangle(cv2.UMat(prev_img), (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        pred_img = BoundingBox(*pred_bb[i].cpu().detach().numpy().tolist())
        pred_img.unscale(prev_img.get())
        x1, y1, x2, y2 = int(pred_img.x1), int(pred_img.y1), int(pred_img.x2), int(pred_img.y2)
        curr_img = cv2.rectangle(cv2.UMat(curr_img), (int(x1), int(y1)),
                                 (int(x2), int(y2)), (255, 255, 0), 2)

        if viz:
            viz.plot_image_opencv(prev_img.get(),
                                  title='prev_img_{}'.format(i))
            viz.plot_image_opencv(curr_img.get(),
                                  title='curr_img_with_bb_{}'.format(i))


def main(args, dbg=True):
    """Testing goturn tracker
    """
    if dbg:
        viz = Visualizer()

    model_dir = Path(args.model_dir)
    # Checkpoint path
    ckpt_dir = model_dir.joinpath('checkpoints')
    ckpt_path = next(ckpt_dir.glob('*.ckpt'))

    model = GoturnTrain.load_from_metrics(weights_path=ckpt_path,
                                          tags_csv=model_dir.joinpath('meta_tags.csv'))
    model.eval()
    model.freeze()

    imagenet_path = args.imagenet_path
    alov_path = args.alov_path

    objGoturn = GoturnDataloader(imagenet_path, alov_path,
                                 isTrain=False, dbg=False)
    val_loader = DataLoader(objGoturn,
                            batch_size=1, shuffle=False,
                            num_workers=6,
                            collate_fn=objGoturn.collate)

    for i, (curr, prev, gt_bb) in tqdm(enumerate(val_loader)):
        for i in range(curr.shape[0]):
            c, p = curr[i].unsqueeze(0), prev[i].unsqueeze(0)
            pred_bb = model.forward(p, c)
            gt = gt_bb[i].unsqueeze(0)
            if dbg:
                vis_images(p, c, gt, pred_bb, viz=viz)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('--imagenet_path',
                    required=True, help='path to imagenet')
    ap.add_argument('--alov_path',
                    required=True, help='path to alov')
    ap.add_argument('--model_dir',
                    required=True, help='model directory')

    args = ap.parse_args()
    main(args)
