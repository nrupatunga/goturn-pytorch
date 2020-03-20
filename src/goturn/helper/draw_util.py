"""
File: draw_util.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: Drawing utils for goturn project
"""

import cv2
import numpy as np


class draw:

    """Drawing utils for images"""

    @staticmethod
    def bbox(img, bb, color=(0, 255, 0)):
        """draw a bounding box on image
        @img: OpenCV image
        @bb: bounding box, assuming all the boundary conditions are
        satisfied
        @color: color of the bounding box
        """

        img_out = np.copy(img)
        img_out = np.ascontiguousarray(img_out)
        x1, y1 = int(bb.x1), int(bb.y1)
        x2, y2 = int(bb.x2), int(bb.y2)

        img_out = cv2.rectangle(img_out, (x1, y1), (x2, y2), color, 2)

        return img_out
