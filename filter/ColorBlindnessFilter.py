#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    This file defines a color blindness simulation filter to be used in experiments and evaluations.
"""

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2020, John Hoff"
__license__ = "MIT License"
__version__ = "1.0.0"


import numpy as np


# Matrix to perform RGB to LMS conversion
lms_matrix = np.array([
    [17.8824, 43.5161, 4.11935],
    [3.4565, 27.1554, 3.86714],
    [0.02996, 0.18431, 1.46709]
])
lms_matrix_inverse = np.linalg.inv(lms_matrix)

# Matrix to simulate protanopia
lms_protanopia_matrix = np.array([
    [0, 2.02344, -2.53581],
    [0, 1, 0],
    [0, 0, 1]
])

# Matrix to simulate deuteranopia
lms_deuteranopia_matrix = np.array([
    [1, 0, 0],
    [0.494207, 0, 1.24827],
    [0, 0, 1]
])


def apply_matrix_to_image(matrix, image):
    """
    This is an internal utility function that will apply a matrix to each pixel of an array of pixels.  This function
    assumes that each pixel is an array of values.
    :param matrix: An np.ndarray to be applied to each pixel
    :param image: An np.ndarray containing the pixel data.
    :return: An np.ndarray containing the transformed pixels.
    """
    reshaped_image = image.reshape((image.shape[0] * image.shape[1], image.shape[2]))
    return np.dot(matrix, reshaped_image.T).T.reshape(image.shape)


def rgb2lms(image):
    """
    Convert an array of pixels from the RGB colorspace to the LMS colorspace.  This function assumes that
    each pixel is an array of RGB values [0,1].
    :param image: An np.ndarray containing the image data
    :return: An np.ndarray containing the transformed image data
    """
    return np.clip(apply_matrix_to_image(lms_matrix, image), 0.0, 1.0)


def lms2rgb(image):
    """
    Convert an array of pixels from the LMS colorspace to the RGB colorspace.  This function assumes that
    each pixel in an array of LMS values.
    :param image: An np.ndarray containing the image data
    :return: An np.ndarray containing the transformed image data
    """
    return np.clip(apply_matrix_to_image(lms_matrix_inverse, image), 0.0, 1.0)


def apply_rgb_protanopia(image):
    """
    Simulate the affects of protanopia color blindness on an image.  This function assumes that the image is an
    np.ndarray containing RGB values [0,255]
    TODO: Add explanation
    TODO: Add citation
    :param image: An np.ndarray containing the image data
    :return: An np.ndarray containing the transformed image data
    """
    return np.clip(
        apply_matrix_to_image(
            np.dot(lms_matrix_inverse, np.dot(lms_protanopia_matrix, lms_matrix)),image/255.)*255., 0., 255.)


def apply_rgb_deuteranopia(image):
    """
    Simulate the affects of deuteranopia color blindness on an image.  This function assumes that the image is an
    np.ndarray containing RGB values [0,255].
    TODO: Add explanation
    TODO: Add citation
    :param image: An np.ndarray containing the image data
    :return: An np.ndarray containing the transformed image data
    """
    return np.clip(
        apply_matrix_to_image(
            np.dot(lms_matrix_inverse, np.dot(lms_deuteranopia_matrix, lms_matrix)), image/255.)*255., 0., 255.)


class ColorBlindnessFilter(object):
    """
    The color blindness filter will accept two different blindness types: protanopia and deuteranopia.
    The simulation is performed by converting each image into the LMS color space, applying the necessary
    color blindness transformation, and then tranforming back into the RGB color space.
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
            return apply_rgb_protanopia(image)
        if type == 'deuteranopia' or type == 'deut':
            return apply_rgb_deuteranopia(image)
        return image
