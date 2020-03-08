#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Wrapper for scikit-optimize search process
"""

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2020, John Hoff"
__license__ = "MIT License"
__version__ = "1.0.0"


import time

from skopt import gp_minimize

from utility.score import filtered_network_eot


class OptimizationTest(object):
    def __init__(self, images, filter, network_model, preprocessor, lambda_value=1.):
        self.images = images
        self.filter = filter
        self.network_model = network_model
        self.preprocessor = preprocessor
        self.lambda_value = lambda_value

        self.parameter_labels = None
        self.iterations_current = None
        self.iterations_max = None
        self.search_start_time = None
        self.best_score = None
        self.best_additional_scores = None

    def perform_test(self):

        self.parameter_labels = list()
        for label in self.filter.get_parameter_labels():
            self.parameter_labels.append(label)
        self.iterations_current = 1
        self.iterations_max = len(self.filter.get_parameters())
        self.search_start_time = time.time()
        self.best_score = None
        self.best_additional_scores = dict()

        eot_scores = list()

        def objective_function(parameters):\
            # Quick information about the current iteration
            elapsed_time = time.time() - self.search_start_time
            if self.iterations_current == 1:
                estimated_time = 0
            else:
                estimated_time = (elapsed_time / (self.iterations_current - 1)) * self.iterations_max - elapsed_time
            elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
            estimated_time_str = time.strftime("%H:%M:%S", time.gmtime(estimated_time))
            print('Iteration %s of %s (Running: %s, Remaining: %s)'
                  % (self.iterations_current, self.iterations_max, elapsed_time_str, estimated_time_str))
            for i in range(len(parameters)):
                print('%s = %s' % (self.parameter_labels[i], parameters[i]))
            self.iterations_current += 1

            eot_scoring = filtered_network_eot(self.images,
                                               self.filter,
                                               parameters,
                                               self.preprocessor,
                                               self.network_model,
                                               self.lambda_value)
            eot_scores.append(eot_scoring.get_all_scores())

            print('Expectation over Transformation scoring:')
            print(eot_scoring.get_all_scores())

            return -eot_scoring.adjusted_eot

        test_results = list()
        for parameters in self.filter.get_parameters():
            test_results.append([parameters, objective_function(parameters)])

        print(test_results)

        return test_results, eot_scores
