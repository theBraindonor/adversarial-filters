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


from skimage.filters import gaussian
from skopt.space import Real


class GaussianFilter(object):
    def __init__(self):
        pass

    @staticmethod
    def get_default_dimensions():
        return [1.0]

    @staticmethod
    def get_dimensions():
        return [
            Real(low=0.25, high=24, name='sigma')
        ]

    @staticmethod
    def transform_image(dimensions, image):
        sigma = dimensions[0]
        return gaussian(image, sigma=sigma, multichannel=True)
