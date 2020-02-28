#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    This file holds common code for creating the datasets of images for the adversarial attacks.
"""

__author__ = "John Hoff"
__email__ = "john.hoff@braindonor.net"
__copyright__ = "Copyright 2020, John Hoff"
__license__ = "MIT License"
__version__ = "1.0.0"


import argparse

from keras.preprocessing import image
import numpy as np
import pandas as pd

from utility import use_project_path


class DatasetCreator(object):
    def __init__(self, name, network, preprocessor, decoder):
        """
        Create a new DatasetCreator.  This will allow for easy creation of imagenet validation samples.
        :param name: The label for the dataset
        :param network: The neural network constructor to test
        :param preprocessor: The pre-processor for the input images
        :param decoder: The decoder to turn predictions into synset names
        """
        self.name = name
        self.network = network
        self.preprocessor = preprocessor
        self.decoder = decoder

    def create_and_save(self):
        """
        Create and save the dataset using the arguments passed in from the console.  This will shuffle the validation
        data, create a classifier, and select the correct classifications that are seen.  The resulting list
        of images that are correctly classified is then saved to an NPY file.
        :return:
        """
        parser = argparse.ArgumentParser()
        parser.add_argument('-rs', '--random-state', default=1024,
                            help='Random state for the shuffling.')
        parser.add_argument('-n', '--number', default=100,
                            help='Number of correct classifications to store.')
        arguments = vars(parser.parse_args())

        random_state = int(arguments['random_state'])
        number = int(arguments['number'])

        print('')
        print('Starting %s Image and Label Extraction...' % self.name)
        print('')
        print('Parameters:')
        print('    Random State: %s' % random_state)
        print('          Number: %s' % number)
        print('')

        use_project_path()

        # Load and shuffle the validation dataset
        image_df = pd.read_csv('data/full_image_dataset.csv')
        image_df = image_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

        # Create the CNN to be tested
        model = self.network(weights='imagenet')

        accumulator = list()

        # Step through the dataframe and keep all images correctly classified.
        for index, row in image_df.iterrows():
            scaled_image = image.img_to_array(image.load_img(row['image'], target_size=(224, 224)))
            input = self.preprocessor(np.expand_dims(scaled_image.copy(), axis=0))
            prediction = self.decoder(model.predict(input), top=1)[0][0][0]
            if row['label'] == prediction:
                accumulator.append([scaled_image, prediction])
            if len(accumulator) >= number:
                break

        np.save('data/%s_%s_correct.npy' % (self.name, number), np.array(accumulator))
