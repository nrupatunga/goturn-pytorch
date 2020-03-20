"""
File: annotation.py
Author: Nrupatunga
Email: nrupatunga.s@byjus.com
Github: https://github.com/nrupatunga
Description: Bounding box annotations
"""

import sys

from loguru import logger

try:
    from goturn.helper.BoundingBox import BoundingBox
except ImportError:
    logger.error('Please run $source settings.sh from root directory')
    sys.exit(1)


class annotation:

    def __init__(self):
        """Annotation class stores bounding box, image path"""
        self.bbox = BoundingBox(0, 0, 0, 0)
        self.image_path = []
        self.disp_width = 0
        self.disp_height = 0

    def setbbox(self, x1, x2, y1, y2):
        """ set the bounding box """
        self.bbox.x1 = x1
        self.bbox.x2 = x2
        self.bbox.y1 = y1
        self.bbox.y2 = y2

    def setImagePath(self, img_path):
        """ set the image path """
        self.image_path = img_path

    def setWidthHeight(self, disp_width, disp_height):
        """ set width and height """
        self.disp_width = disp_width
        self.disp_height = disp_height

    def __repr__(self):
        return str({'bbox': self.bbox, 'image_path': self.image_path,
                    'w': self.disp_width, 'h': self.disp_height})
