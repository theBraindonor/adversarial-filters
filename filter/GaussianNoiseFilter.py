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
from skimage.util import random_noise
from skopt.space import Real


class GaussianNoiseFilter(object):
    def __init__(self, random_state):
        self.random_state = random_state

    @staticmethod
    def get_default_dimensions():
        return [0.0, 0.01]

    @staticmethod
    def get_dimensions():
        return [
            Real(low=-0.05, high=0.05, name='mean'),
            Real(low=0.001, high=0.05, name='var')
        ]

    def transform_image(self, dimensions, image):
        mean, var = dimensions
        previous = np.random.get_state()
        result = np.clip(random_noise(np.clip(image/255., 0., 1.), mode='gaussian',
                                      seed=self.random_state, mean=mean, var=var)*255., 0., 255.)
        np.random.set_state(previous)
        return result
