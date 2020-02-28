#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    This file will create an NPY file with images that are imagenet validation images that are correctly labeled
    by the VGG19 classifier.  The number of images to collect and the random seed will be configurable from
    the command line.  The goal is to ensure that we have a robust and repeatable set of images to attack without
    attacking the entire 50k image validation set.
"""

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2020, John Hoff"
__license__ = "MIT License"
__version__ = "1.0.0"


from keras.applications.vgg19 import VGG19, preprocess_input, decode_predictions

from utility import DatasetCreator


if __name__ == '__main__':
    dataset_creator = DatasetCreator('vgg19', VGG19, preprocess_input, decode_predictions)
    dataset_creator.create_and_save()
