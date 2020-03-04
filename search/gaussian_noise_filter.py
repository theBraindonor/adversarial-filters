#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Filters to be used in the underlying testing and transforming
"""

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2020, John Hoff"
__license__ = "MIT License"
__version__ = "1.0.0"


from keras.applications.vgg16 import VGG16, preprocess_input
import numpy as np

from filter import GaussianNoiseFilter
from utility import OptimizationSearch
from utility import use_project_path


if __name__ == '__main__':
    use_project_path()

    search = OptimizationSearch(
        np.load('data/vgg16_100_correct.npy', allow_pickle=True),
        GaussianNoiseFilter(4545),
        VGG16(weights='imagenet'),
        preprocess_input
    )

    search.perform_search()
