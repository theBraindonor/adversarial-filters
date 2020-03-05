#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Routines to save filter scores
"""

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2020, John Hoff"
__license__ = "MIT License"
__version__ = "1.0.0"


import pandas as pd


def save_filter_search_scores(filter, results, scores, filename):
    search_df = pd.DataFrame(scores)

    dimension_labels = list()
    for dimension in filter.get_dimensions():
        dimension_labels.append(dimension.name)

    dimensions = dict()
    for dimension in dimension_labels:
        dimensions[dimension] = list()

    for x in results.x_iters:
        for i in range(len(dimension_labels)):
            dimensions[dimension_labels[i]].append(x[i])

    print(dimensions)

    for i in range(len(dimension_labels)):
        search_df['dim_%s' % dimension_labels[i]] = dimensions[dimension_labels[i]]

    search_df.to_csv(filename, index_label='step')
