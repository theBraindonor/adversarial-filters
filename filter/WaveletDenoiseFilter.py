#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    This file will create a configurable wavelet denoising that can be used to test the effectiveness of the
    filter in adversarial attacks
"""

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2020, John Hoff"
__license__ = "MIT License"
__version__ = "1.0.0"


import numpy as np
from skimage.restoration import denoise_wavelet
from skopt.space import Real


class WaveletDenoiseFilter(object):
    def __init__(self):
        pass

    @staticmethod
    def get_default_dimensions():
        return [0.1]

    @staticmethod
    def get_dimensions():
        return [
            Real(low=0.01, high=1.0, name='sigma')
        ]

    @staticmethod
    def transform_image(dimensions, image):
        sigma = dimensions[0]

        return np.clip(denoise_wavelet(image/255.0, sigma=sigma, multichannel=True)*255., 0., 255.)
