#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Construct a CSV file with the raw imagenet images and their label data.  This file expects the following
    files to exist in the project which will need to be downloaded independently:
        data/val/ILSVRC2012_val_*.JPEG -> Validation set images
        data/ILSVRC2012_validation_ground_truth.txt -> Validation set labels
        data/meta.mat -> labels to synset mappings
        data/synset_words.txt -> synset to words mappings

    The purpose of this file is to create an easy-to-access CSV that we can use to find images that are correctly
    classified by the imagenet-based network so that we can attack a known-working image.
"""

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2020, John Hoff"
__license__ = "MIT License"
__version__ = "1.0.0"

import glob
import re

import numpy as np
import pandas as pd
from scipy.io import loadmat

from utility import use_project_path

if __name__ == '__main__':

    use_project_path()

    # Grab all of the images and store them in ascending order in a pandas array.
    raw_image_array = np.rot90(np.array([
        [filename.replace('\\', '/') for filename in glob.iglob('data/val/*.JPEG')]
    ]))
    image_df = pd.DataFrame(raw_image_array, columns=['image'])
    image_df.sort_values('image', axis=0, inplace=True)

    # Read in the labels and make sure they are a zero-based index
    raw_label_indexes = np.loadtxt('data/ILSVRC2012_validation_ground_truth.txt', dtype='int')
    image_df['label'] = raw_label_indexes - 1

    # Load the meta.mat and append the correct synset value to the array
    meta = loadmat('data/meta.mat')
    synset_map = dict()
    for i in range(1000):
        label_index = int(meta['synsets'][i,0][0][0][0])
        synset = meta['synsets'][i,0][1][0]
        synset_map[label_index - 1] = synset
    image_df['label'] = image_df['label'].apply(lambda x: synset_map[x])

    # Load the synset_words and add this to the pandas array.  Please note that we are breaking the human
    # readable name to exclude everything appearing after a comma.
    synset_regex = re.compile(r'^(n\d{8}) ([a-zA-Z \-]+),?')
    synset_label_map = dict()
    with open('data/synset_words.txt', 'r') as file:
        for line in file.readlines():
            matches = synset_regex.match(line.strip())
            if matches:
                synset_label = matches.group(1)
                synset_name = matches.group(2)
                synset_label_map[synset_label] = synset_name
    image_df['label_name'] = image_df['label'].apply(lambda x: synset_label_map[x])

    image_df.to_csv('data/full_image_dataset.csv', index=False)
