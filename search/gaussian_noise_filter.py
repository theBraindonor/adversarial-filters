#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Filters to be used in the underlying testing and transforming
"""

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2020, John Hoff"
__license__ = "MIT License"
__version__ = "1.0.0"

import time

from keras.applications.vgg16 import VGG16, preprocess_input, decode_predictions
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from skopt import gp_minimize

from filter import GaussianNoiseFilter
from utility import lab_color_distance, use_project_path

# TODO: Search a gaussian filter parameter space for the most effective gaussian filter for

if __name__ == '__main__':
    use_project_path()

    images = np.load('data/vgg16_100_correct.npy', allow_pickle=True)

    filter = GaussianNoiseFilter(4545)

    x0 = filter.get_default_dimensions()
    y0 = None

    # TODO: Need to establish what this value should actually be
    lambda_value = 1/127.

    print(len(images))

    network_model = VGG16(weights='imagenet')

    dimension_labels = list()
    for dimension in filter.get_dimensions():
        dimension_labels.append(dimension.name)

    iterations_current = 1
    iterations_max = 25
    search_start_time = time.time()

    def objective_function(dimensions):
        global iterations_current

        elapsed_time = time.time() - search_start_time
        if iterations_current == 1:
            estimated_time = 0
        else:
            estimated_time = (elapsed_time / (iterations_current - 1)) * iterations_max - elapsed_time
        elapsed_time_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        estimated_time_str = time.strftime("%H:%M:%S", time.gmtime(estimated_time))
        print('Iteration %s of %s (Running: %s, Remaining: %s)'
              % (iterations_current, iterations_max, elapsed_time_str, estimated_time_str))
        for i in range(len(dimensions)):
            print('%s = %s' % (dimension_labels[i], dimensions[i]))
        iterations_current += 1

        image_scores = list()
        misclassified = 0
        distances = list()
        probability_differences = list()

        for image in images:
            original_image, target = image
            network_input = filter.transform_image(dimensions, original_image)
            color_distance = lab_color_distance(original_image, network_input)
            distances.append(color_distance)
            network_input = preprocess_input(np.expand_dims(network_input, axis=0))
            predictions = network_model.predict(network_input)
            target_prediction = predictions[0][target]
            model_prediction = predictions[0][np.argmax(predictions)]
            probability = np.max([model_prediction - target_prediction, 0.001])
            if target_prediction < model_prediction:
                probability_differences.append(model_prediction)
                misclassified += 1
            else:
                probability_differences.append(-target_prediction)
            score = np.log(probability) - (lambda_value * color_distance)
            image_scores.append(score)

        print(min(probability_differences))
        print(max(probability_differences))
        probability_differences = np.array(probability_differences)
        distances = np.array(distances)

        scaler = MinMaxScaler(feature_range=(0.01, np.max(probability_differences)))
        scaled_probability_distances = scaler.fit_transform(probability_differences.reshape(-1, 1)).flatten()

        scores = np.log(scaled_probability_distances) - (lambda_value * distances)

        print('Mean Scores: %s' % np.mean(scores))
        print('Mean EOT: %s' % np.mean(image_scores))
        print(' Mean CD: %s' % np.mean(distances))
        print('Accuracy: %s' % ((len(images) - misclassified) / len(images)))
        return -np.mean(scores)

    search_results = gp_minimize(func=objective_function,
                                 dimensions=filter.get_dimensions(),
                                 x0=x0, y0=y0, n_calls=iterations_max, random_state=454)

    print(search_results)
