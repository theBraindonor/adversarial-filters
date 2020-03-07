#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    This file will create a configurable fourier uniform filter that can be used to test the effectiveness of the
    filter in adversarial attacks
"""

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2020, John Hoff"
__license__ = "MIT License"
__version__ = "1.0.0"


import numpy as np
from scipy import fftpack
from scipy.ndimage import fourier_uniform
from skopt.space import Real


class FourierUniformFilter(object):
    def __init__(self):
        pass

    @staticmethod
    def get_default_dimensions():
        return [1.0]

    @staticmethod
    def get_dimensions():
        return [
            Real(low=0.25, high=24, name='size')
        ]

    @staticmethod
    def transform_image(dimensions, image):
        size = dimensions[0]

        image = image.copy() / 255.
        for channel in range(image.shape[2]):
            image[:, :, channel] = fftpack.ifft2(fourier_uniform(fftpack.fft2(image[:, :, channel]), size)).real
        return np.clip(image * 255., 0., 255.)

