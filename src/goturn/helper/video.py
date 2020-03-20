# Date: Thursday 08 June 2017 11:53:29 AM IST
# Email: nrupatunga@whodat.com
# Name: Nrupatunga
# Description: Basic video helper functions

import numpy as np

from .image_io import load


class frame:

    """Docstring for frame. """

    def __init__(self, frame_num, bbox):
        self._frame_num = frame_num
        self._bbox = bbox


class video:

    """Docstring for video. """

    def __init__(self, video_path):
        """
        @video_path: video path
        """
        self._video_path = video_path
        self._all_frames = []
        self._annotations = []

    def load_annotation(self, annotation_index):
        """load annotation"""

        ann_frame = self._annotations[annotation_index]
        frame_num = ann_frame._frame_num
        bbox = ann_frame._bbox

        image_files = self._all_frames

        assert(len(image_files) > 0)
        assert(frame_num < len(image_files))

        image = load(image_files[frame_num])
        image = np.asarray(image, dtype=np.uint8)
        return frame_num, image, bbox
