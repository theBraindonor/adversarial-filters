#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Expectation over Transformation scoring code.
"""

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2020, John Hoff"
__license__ = "MIT License"
__version__ = "1.0.0"


import numpy as np


class ExpectationOverTransformation(object):
    def __init__(self, lambda_value):
        self.lambda_value = lambda_value
        self.probabilities = list()
        self.distances = list()
        self.scores = None
        self.mean_score = None
        self.max_score = None

    def update(self, probability, distance):
        self.probabilities.append(probability)
        self.distances.append(distance)

    def finalize(self):
        self.probabilities = np.array(self.probabilities)
        self.distances = np.array(self.distances)
        self.scores = np.log(self.probabilities) - (self.lambda_value * self.distances)
        self.mean_score = np.mean(self.scores)
        self.max_score = np.max(self.scores)

    def get_mean_score(self):
        return self.mean_score

    def get_max_score(self):
        return self.max_score

