#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    This file will create a configurable gaussian filter that can be used to test the effectiveness of the
    filter in adversarial attacks
"""

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2020, John Hoff"
__license__ = "MIT License"
__version__ = "1.0.0"


import numpy as np
from skimage.exposure import equalize_hist


class HistogramEqualizationFilter(object):
    def __init__(self):
        pass

    @staticmethod
    def get_parameters():
        return [
            [256, True],
            [256, False],
            [128, True],
            [128, False],
            [64, True],
            [64, False]
        ]

    @staticmethod
    def get_parameter_labels():
        return ['bins', 'multichannel']

    @staticmethod
    def transform_image(parameters, image):
        bins, multichannel = parameters

        if multichannel:
            image = image.copy()/255.
            for channel in range(image.shape[2]):
                image[:,:,channel] = equalize_hist(image[:,:,channel], nbins=bins)
            return np.clip(image*255., 0., 255.)
        else:
            return np.clip(equalize_hist(image/255., nbins=bins)*255., 0., 255.)
