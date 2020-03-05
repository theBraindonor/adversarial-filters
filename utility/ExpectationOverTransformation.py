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
from sklearn.preprocessing import MinMaxScaler


class ExpectationOverTransformation(object):
    def __init__(self, lambda_value):
        if lambda_value is not None:
            self.lambda_value = lambda_value
        else:
            self.lambda_value = 0.5
        self.probabilities = list()
        self.distances = list()
        self.correctly_classified = 0
        self.scores = None
        self.mean_score = None
        self.max_score = None
        self.sd_score = None
        self.mean_distance = None
        self.max_distance = None
        self.sd_distance = None
        self.mean_penalty = None
        self.max_penalty = None
        self.sd_penalty = None
        self.accuracy = None
        self.raw_eot = None
        self.adjusted_eot = None
        self.max_score_index = None

    def update(self, probability, distance, correctly_classified):
        self.probabilities.append(probability)
        self.distances.append(distance)
        if correctly_classified:
            self.correctly_classified += 1

    def finalize(self):
        self.probabilities = np.array(self.probabilities)
        scaler = MinMaxScaler(feature_range=(0.01, np.max([np.max(self.probabilities), 0.5])))
        self.probabilities = scaler.fit_transform(self.probabilities.reshape(-1, 1)).flatten()
        self.distances = np.array(self.distances)
        self.mean_distance = np.mean(self.distances)
        self.max_distance = np.max(self.distances)
        self.sd_distance = np.std(self.distances)
        self.mean_penalty = np.mean(np.log(self.probabilities))
        self.max_penalty = np.max(np.log(self.probabilities))
        self.sd_penalty = np.std(np.log(self.probabilities))
        self.scores = np.log(self.probabilities) - (self.lambda_value * self.distances)
        self.max_score_index = np.argmax(self.scores)
        self.mean_score = np.mean(self.scores)
        self.max_score = np.max(self.scores)
        self.sd_score = np.std(self.scores)
        self.accuracy = self.correctly_classified / len(self.scores)
        self.raw_eot = self.max_score
        self.adjusted_eot = self.mean_score + self.sd_score

    def get_mean_score(self):
        return self.mean_score

    def get_max_score(self):
        return self.max_score

    def get_sd_score(self):
        return self.sd_score

    def get_mean_penalty(self):
        return self.mean_penalty

    def get_max_penalty(self):
        return self.max_penalty

    def get_sd_penalty(self):
        return self.sd_distance

    def get_mean_distance(self):
        return self.mean_distance

    def get_max_distance(self):
        return self.max_distance

    def get_sd_distance(self):
        return self.sd_distance

    def get_raw_eot(self):
        return self.raw_eot

    def get_adjusted_eot(self):
        return self.adjusted_eot

    def get_accuracy(self):
        return self.accuracy

    def get_max_score_index(self):
        return self.max_score_index

    def get_all_scores(self):
        return {
            'raw_eot': self.raw_eot,
            'adjusted_eot': self.adjusted_eot,
            'accuracy': self.accuracy,
            'max_score_index': self.max_score_index,
            'mean_distance': self.mean_distance,
            'max_distance': self.max_distance,
            'sd_distance': self.sd_distance,
            'mean_penalty': self.mean_penalty,
            'max_penalty': self.max_penalty,
            'sd_penalty': self.sd_penalty,
            'mean_score': self.mean_score,
            'max_score': self.max_score,
            'sd_score': self.sd_score
        }

