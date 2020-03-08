#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Test the impact of daltonization to compensate for color blindness on several imagent
    base classifiers.
    Protanopia and Deuteranopia are the two types of color blindness supported by this experiment.
"""

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2020, John Hoff"
__license__ = "MIT License"
__version__ = "1.0.0"

import argparse

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as vgg19_preprocess_input
from keras.applications.densenet import DenseNet201
from keras.applications.densenet import preprocess_input as densenet_preprocess_input
from keras.applications.resnet_v2 import ResNet152V2
from keras.applications.resnet_v2 import preprocess_input as resnet_preprocess_input
import numpy as np

from filter import DaltonizationFilter
from utility import OptimizationTest
from utility import use_project_path
from utility import save_filter_test_scores

if __name__ == '__main__':
    use_project_path()

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--network', default='vgg16',
                        help='The network architecture to test: vgg16, vgg19, densenet, or resnet')
    parser.add_argument('-s', '--sample', default=100,
                        help='The sample file to use for testing.')
    arguments = vars(parser.parse_args())

    NETWORK = arguments['network']
    SAMPLE = arguments['sample']

    print('')
    print('Starting adversarial evaluation of a color blindness daltonization filter.')
    print('')
    print('       Network: %s' % NETWORK)
    print('        Sample: %s' % SAMPLE)
    print('')

    filter = DaltonizationFilter()

    if NETWORK == 'vgg16':
        model = VGG16(weights='imagenet')
        search = OptimizationTest(
            np.load('data/vgg16_%s_correct.npy' % SAMPLE, allow_pickle=True),
            filter,
            model,
            vgg16_preprocess_input,
            lambda_value=1.0
        )
        results, scores = search.perform_test()
        save_filter_test_scores(filter, results, scores, 'log/daltonization_filter_vgg16_%s_search.csv' % SAMPLE)

    if NETWORK == 'vgg19':
        model = VGG19(weights='imagenet')
        search = OptimizationTest(
            np.load('data/vgg19_%s_correct.npy' % SAMPLE, allow_pickle=True),
            filter,
            model,
            vgg19_preprocess_input,
            lambda_value=1.0
        )
        results, scores = search.perform_test()
        save_filter_test_scores(filter, results, scores, 'log/daltonization_filter_vgg19_%s_search.csv' % SAMPLE)

    if NETWORK == 'densenet':
        model = DenseNet201(weights='imagenet')
        search = OptimizationTest(
            np.load('data/densenet201_%s_correct.npy' % SAMPLE, allow_pickle=True),
            filter,
            model,
            densenet_preprocess_input,
            lambda_value=1.0
        )
        results, scores = search.perform_test()
        save_filter_test_scores(filter, results, scores, 'log/daltonization_filter_densenet201_%s_search.csv' % SAMPLE)

    if NETWORK == 'resnet':
        model = ResNet152V2(weights='imagenet')
        search = OptimizationTest(
            np.load('data/resnet152v2_%s_correct.npy' % SAMPLE, allow_pickle=True),
            filter,
            model,
            resnet_preprocess_input,
            lambda_value=1.0
        )
        results, scores = search.perform_test()
        save_filter_test_scores(filter, results, scores, 'log/daltonization_filter_resnet152v2_%s_search.csv' % SAMPLE)
