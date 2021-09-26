#!/usr/bin/env python

import os
import pickle

# Functions from other notebook file.
from shared_functions_server import *

# Move one directory back to the project root.
os.chdir("..")

# Suppress tensorflow log messages.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# GLOBALS
DATA_DIRECTORY = os.path.join(os.getcwd() + '/datasets/vgg-16')
IMAGE_HEIGHT = 100
IMAGE_WIDTH = 100
IMAGE_CHANNELS = 3
CATEGORIES = 10 # represents digits 0-9
DIMENSIONS = 4  # 4-digit CAPTCHA images
TRAINING_EPOCHS = 1
TRAINING_BATCH_SIZE = 32
VALIDATION_BATCH_SIZE = 32
TESTING_BATCH_SIZE = 64
MODEL_NAME = 'vgg-16'




def main():

    # Load the CAPTCHA dataset.
    data_frame = create_captcha_dataframe(DATA_DIRECTORY)

    # Split CAPTCHA dataset into training and validation sets.
    train_indices, validation_indices = shuffle_and_split_data(data_frame)

    # Display the number of samples in each set.
    print('training count: %s, validation count: %s' % (
          len(train_indices), len(validation_indices)))
    
    # Create the baseline untrained model.
    model = create_untrained_vgg16_model(IMAGE_HEIGHT, 
                                         IMAGE_WIDTH, 
                                         IMAGE_CHANNELS,
                                         DIMENSIONS, 
                                         CATEGORIES)

    # Train the model.
    history = train_model(model, 
                          data_frame, 
                          train_indices, 
                          validation_indices, 
                          TRAINING_BATCH_SIZE, 
                          VALIDATION_BATCH_SIZE, 
                          TRAINING_EPOCHS,
                          IMAGE_HEIGHT,
                          IMAGE_WIDTH,
                          CATEGORIES)

    # Save the training history.
    with open('training-history', 'wb') as handle:
        pickle.dump(history.history, handle)

    # Save the model.
    model.save(MODEL_NAME)




main()
