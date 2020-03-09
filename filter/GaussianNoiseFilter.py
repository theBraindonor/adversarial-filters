#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    This file defines a gaussian noise filter that can be used in experiments and evaluations.
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
    """
    This filter will apply a gaussian noise transformation with the indicated mean and variance.
    """
    def __init__(self, random_state):
        self.random_state = random_state

    @staticmethod
    def get_default_dimensions():
        return [0.0, 0.05]

    @staticmethod
    def get_dimensions():
        return [
            Real(low=-0.25, high=0.25, name='mean'),
            Real(low=0.01, high=0.1, name='var')
        ]

    def transform_image(self, dimensions, image):
        mean, var = dimensions
        previous = np.random.get_state()
        result = np.clip(random_noise(np.clip(image/255., 0., 1.), mode='gaussian',
                                      seed=self.random_state, mean=mean, var=var)*255., 0., 255.)
        np.random.set_state(previous)
        return result
