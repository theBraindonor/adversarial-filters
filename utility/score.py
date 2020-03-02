#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    A quick hack of convenience to make it easy to run scripts in an IDE and from command line.
"""

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2020, John Hoff"
__license__ = "MIT License"
__version__ = "1.0.0"


import numpy as np
from skimage.color import rgb2lab


def lab_color_distance(image_a, image_b):
    pixels_a = rgb2lab(image_a).reshape(image_a.shape[0] * image_a.shape[1], image_a.shape[2])
    pixels_b = rgb2lab(image_b).reshape(image_b.shape[0] * image_b.shape[1], image_b.shape[2])
    return np.sqrt(np.linalg.norm(pixels_b - pixels_a))
