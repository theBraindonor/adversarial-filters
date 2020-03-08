#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    This file defines a color blindness correction filter that uses Daltonization and can be used
    in experiments and evaluations.
"""

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2020, John Hoff"
__license__ = "MIT License"
__version__ = "1.0.0"


import numpy as np

from filter.ColorBlindnessFilter import apply_matrix_to_image
from filter.ColorBlindnessFilter import lms_matrix_inverse
from filter.ColorBlindnessFilter import lms_matrix
from filter.ColorBlindnessFilter import lms_protanopia_matrix
from filter.ColorBlindnessFilter import lms_deuteranopia_matrix


# Matrix to perform protanopia daltonization
rgb_daltonization_prot_matrix = np.array([
    [0, 0, 0],
    [0.7, 1, 0],
    [0.7, 0, 1]
])

# Matrix to perform deuteranopia daltonization
rgb_daltonization_deut_matrix = np.array([
    [0, 0, 0],
    [0.7, 1, 0],
    [0.7, 0, 1]
])


def daltonize_rgb(image, blind_transform, daltonization_matrix):
    """
    Daltonize an image based on the color blind transformation.  This will attempt to correct for some
    of the information loss due to color blindness.
    :param image: The image to correct
    :param blind_transform: The transformation that simulates the color blindness
    :param daltonization_matrix: The transformation to correct for the information loss
    :return:
    """
    image = image/255.
    image_blind = apply_matrix_to_image(
            np.dot(lms_matrix_inverse, np.dot(blind_transform, lms_matrix)),image)
    return np.clip((image + apply_matrix_to_image(daltonization_matrix, image - image_blind))*255.0, 0.0, 255.0)


class DaltonizationFilter(object):
    """
    The daltonization filter will accept two different blindness types: protanopia and deuteranopia.
    This filter will attempt to correct for the loss of information due to color blindness in an image
    by incorporating the lost color channels into the other color channels.
    """
    def __init__(self):
        pass

    @staticmethod
    def get_parameters():
        return [['protanopia'], ['deuteranopia']]

    @staticmethod
    def get_parameter_labels():
        return ['blindness_type']

    @staticmethod
    def transform_image(parameters, image):
        type = parameters[0]
        if type == 'protanopia' or type == 'prot':
            return daltonize_rgb(image, lms_protanopia_matrix, rgb_daltonization_prot_matrix)
        if type == 'deuteranopia' or type == 'deut':
            return daltonize_rgb(image, lms_deuteranopia_matrix, rgb_daltonization_deut_matrix)
        return image
