#!/usr/bin/env python

import argparse
import os
import pickle
import sys

# Functions from other notebook file.
from shared_functions_server import *

# Move one directory back to the project root.
os.chdir("..")

# Suppress tensorflow log messages.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# GLOBALS
IMAGE_HEIGHT = 100
IMAGE_WIDTH = 100
IMAGE_CHANNELS = 3
CATEGORIES = 10 # represents digits 0-9
DATA_DIRECTORY = None
DIMENSIONS = None  
TRAINING_EPOCHS = None
TRAINING_BATCH_SIZE = None
VALIDATION_BATCH_SIZE = None
MODEL_NAME = None
TRAINING_HISTORY_FILE_NAME = None




def main():
    """
        Create the CAPTCHA-solving model.
    """
    global DATA_DIRECTORY
    global DIMENSIONS
    global TRAINING_EPOCHS
    global TRAINING_BATCH_SIZE
    global VALIDATION_BATCH_SIZE
    global MODEL_NAME
    global TRAINING_HISTORY_FILE_NAME

    # Get the command-line arguments.
    arguments = parse_command_line_arguments()

    # Assign values retrieved from command-line.
    DATA_DIRECTORY =  os.path.join(os.getcwd() + 
                                  '/datasets/' + arguments.data_directory[0])
    DIMENSIONS = arguments.length[0]
    TRAINING_EPOCHS = arguments.epochs[0]
    TRAINING_BATCH_SIZE = arguments.batch_size[0]
    VALIDATION_BATCH_SIZE = arguments.batch_size[0]
    MODEL_NAME = arguments.model_name[0]
    TRAINING_HISTORY_FILE_NAME = arguments.training_history_file_name[0]
  
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
    with open(TRAINING_HISTORY_FILE_NAME, 'wb') as handle:
        pickle.dump(history.history, handle)

    # Save the model.
    model.save(MODEL_NAME)




def parse_command_line_arguments():
    """ 
        Parse the arguments from the command-line.

        If no arguments are passed, the help screen will
        be shown and the program will be terminated.

    Returns:
        the parser with command-line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-d', '--data_directory', nargs=1, required=True, 
                        help='Name of the directory holding the CAPTCHA dataset.')

    parser.add_argument('-l', '--length', type=int, choices=range(1, 6), nargs=1, required=True, 
                        help='Number of characters for each CAPTCHA image.')

    parser.add_argument('-e', '--epochs', type=int, nargs=1, required=True,
                        help='Number of epochs when training the model.')

    parser.add_argument('-b', '--batch_size', type=int, choices=[16, 32, 64], nargs=1, required=True,
                        help='Number of epochs when training the model.')

    parser.add_argument('-m', '--model_name', nargs=1, required=True,
                        help='Name of the model file when saving to disk.')

    parser.add_argument('-t', '--training_history_file_name', nargs=1, required=True,
                        help='Name of the destination file name for storing training history information.')

    # if no arguments were passed, show the help screen
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    return parser.parse_args()




main()
