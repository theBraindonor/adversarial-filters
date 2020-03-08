#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    This file defines a color blindness correction filter that uses Daltonization and LAB color enhancement
     and can be used in experiments and evaluations.
"""

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2020, John Hoff"
__license__ = "MIT License"
__version__ = "1.0.0"


import numpy as np
from skimage.color import lab2rgb
from skimage.color import rgb2lab

from filter.ColorBlindnessFilter import apply_matrix_to_image
from filter.ColorBlindnessFilter import lms_matrix_inverse
from filter.ColorBlindnessFilter import lms_matrix
from filter.ColorBlindnessFilter import lms_protanopia_matrix
from filter.ColorBlindnessFilter import lms_deuteranopia_matrix
from filter.DaltonizationFilter import rgb_daltonization_prot_matrix
from filter.DaltonizationFilter import rgb_daltonization_deut_matrix


def daltonize_rgb_no_clip(image, blind_transform, daltonization_matrix):
    """
    Daltonization without adjusting the image to a float between [0,1]
    :param image: The image to transform
    :param blind_transform: The color blindness simulation
    :param daltonization_matrix: The daltonization transform
    :return:
    """
    image_blind = apply_matrix_to_image(
            np.dot(lms_matrix_inverse, np.dot(blind_transform, lms_matrix)),image)
    return image + apply_matrix_to_image(daltonization_matrix, image - image_blind)


def correct_lab_color(image):
    """
    Enhance the pixels of the image in the LAB color space
    :param image: The image to enhance
    :return:
    """
    lab_image = rgb2lab(image)
    a_max = np.max(lab_image[:, :, 1])
    a_min = np.min(lab_image[:, :, 1])
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            l, a, b = lab_image[i][j]

            if a > 0:
                a_adjust = ((a_max - a) / a_max) * a
                a = a + a_adjust
                if b < 0:
                    b = b - a_adjust
                else:
                    b = b - 1.25 * a_adjust
                l -= 0.25 * a_adjust
            else:
                a_adjust = ((a - a_min) / a_min) * a
                a = a - a_adjust
                b = b + a_adjust

            lab_image[i][j] = [l, a, b]

    return np.clip(lab2rgb(lab_image)*255., 0., 255.)


class DaltonizationLabEnchanceFilter(object):
    """
    The daltonization with lab color enhancement filter will accept two different blindness types: protanopia and
    deuteranopia.  This filter will attempt to correct for the loss of information due to color blindness in an image
    by incorporating the lost color channels into the other color channels.  After adjusting for the loss, it will
    attempt to correct for changes in color using the LAB color space.
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
            return correct_lab_color(
                daltonize_rgb_no_clip(image/255., lms_protanopia_matrix, rgb_daltonization_prot_matrix))
        if type == 'deuteranopia' or type == 'deut':
            return correct_lab_color(
                daltonize_rgb_no_clip(image/255., lms_deuteranopia_matrix, rgb_daltonization_deut_matrix))
        return image
