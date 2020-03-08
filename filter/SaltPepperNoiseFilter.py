#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    This file will create a configurable salt and pepper noise filter that can be used to test the effectiveness of the
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


class SaltPepperNoiseFilter(object):
    """
    This filter will apply a salt and pepper noise transformation with the indicated amount and salt_vs_pepper.
    """
    def __init__(self, random_state):
        self.random_state = random_state

    @staticmethod
    def get_default_dimensions():
        return [0.0, 0.01]

    @staticmethod
    def get_dimensions():
        return [
            Real(low=0.01, high=0.25, name='amount'),
            Real(low=0.01, high=0.99, name='salt_vs_pepper')
        ]

    def transform_image(self, dimensions, image):
        amount, salt_vs_pepper = dimensions
        previous = np.random.get_state()
        result = np.clip(random_noise(np.clip(image/255., 0., 1.), mode='s&p',
                                      seed=self.random_state, amount=amount,
                                      salt_vs_pepper=salt_vs_pepper)*255., 0., 255.)
        np.random.set_state(previous)
        return result
