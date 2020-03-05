#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    A quick hack of convenience to make it easy to run scripts in an IDE and from command line.
"""

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2020, John Hoff"
__license__ = "MIT License"
__version__ = "1.0.0"


import numpy as np
from skimage.color import rgb2lab

from utility.ExpectationOverTransformation import ExpectationOverTransformation


def lab_color_distance(image_a, image_b):
    pixels_a = rgb2lab(image_a/255).reshape(image_a.shape[0] * image_a.shape[1], image_a.shape[2])
    pixels_b = rgb2lab(image_b/255).reshape(image_b.shape[0] * image_b.shape[1], image_b.shape[2])
    return np.mean(np.sqrt(np.linalg.norm(pixels_b - pixels_a, axis=1)))


def filtered_network_eot(images, filter, dimensions, preprocessor, network_model, lambda_value=None):
    eot_scoring = ExpectationOverTransformation(lambda_value)

    for image in images:
        original_image, target = image
        network_input = filter.transform_image(dimensions, original_image)
        color_distance = lab_color_distance(original_image, network_input)
        network_input = preprocessor(np.expand_dims(network_input, axis=0))
        predictions = network_model.predict(network_input)
        target_prediction = predictions[0][target]
        model_prediction = predictions[0][np.argmax(predictions)]
        if target_prediction < model_prediction:
            eot_scoring.update(model_prediction, color_distance, 0)
        else:
            eot_scoring.update(-target_prediction, color_distance, 1)

    eot_scoring.finalize()

    return eot_scoring
