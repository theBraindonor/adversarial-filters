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

import numpy as np
from skopt import gp_minimize

from utility.ExpectationOverTransformation import ExpectationOverTransformation
from utility.score import lab_color_distance


class OptimizationSearch(object):
    def __init__(self, images, filter, network_model, preprocessor, lambda_value = 1/127, random_state=454):
        self.images = images
        self.filter = filter
        self.network_model = network_model
        self.preprocessor = preprocessor
        self.lambda_value = lambda_value
        self.random_state = random_state

        self.dimenion_labels = None
        self.iterations_current = None
        self.iterations_max = None
        self.search_start_time = None
        self.best_score = None
        self.best_additional_scores = None

    def perform_search(self, iterations=25):
        x0 = self.filter.get_default_dimensions()
        y0 = None

        self.dimension_labels = list()
        for dimension in self.filter.get_dimensions():
            self.dimension_labels.append(dimension.name)
        self.iterations_current = 1
        self.iterations_max = iterations
        self.search_start_time = time.time()
        self.best_score = None
        self.best_additional_scores = dict()

        def objective_function(dimensions):\
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
            for i in range(len(dimensions)):
                print('%s = %s' % (self.dimension_labels[i], dimensions[i]))
            self.iterations_current += 1

            misclassified_count = 0
            eot_scoring = ExpectationOverTransformation(self.lambda_value)

            for image in self.images:
                original_image, target = image
                network_input = self.filter.transform_image(dimensions, original_image)
                color_distance = lab_color_distance(original_image, network_input)
                network_input = self.preprocessor(np.expand_dims(network_input, axis=0))
                predictions = self.network_model.predict(network_input)
                target_prediction = predictions[0][target]
                model_prediction = predictions[0][np.argmax(predictions)]
                probability = np.max([model_prediction - target_prediction, 0.001])
                if target_prediction < model_prediction:
                    probability_difference = (model_prediction + 0.999) / 2
                    misclassified_count += 1
                else:
                    probability_difference = (-target_prediction + 1.001) / 2
                eot_scoring.update(probability_difference, color_distance)

            eot_scoring.finalize()

            print('Mean Score: %s' % eot_scoring.get_mean_score())
            print(' Max Score: %s' % eot_scoring.get_max_score())
            print('Accuracy: %s' % ((len(self.images) - misclassified_count) / len(self.images)))
            return -eot_scoring.get_mean_score()

        search_results = gp_minimize(func=objective_function,
                                     dimensions=self.filter.get_dimensions(),
                                     x0=x0, y0=y0,
                                     n_calls=self.iterations_max,
                                     random_state=self.random_state)

        print(search_results)
