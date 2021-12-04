#!/usr/bin/env python3

import argparse
import os
import pickle
import sys

# Functions from other file.
from model_utils import *

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
MODEL_ARCHITECTURE = None
MODEL_NAME = None
TRAINING_HISTORY_FILE_NAME = None




def main():
    """
        Create the CAPTCHA-solving model.
    """
    # Get the command-line arguments.
    arguments = parse_command_line_arguments()

    # Assign values retrieved from the command-line.
    initialize_globals(arguments)
  
    # Load the CAPTCHA dataset.
    data_frame = create_captcha_dataframe(DATA_DIRECTORY)

    # Create the trainable model and display its architecture configuration.
    model = get_trainable_neural_network()
    model.summary()

    # Split CAPTCHA dataset into training and validation sets.
    train_indices, validation_indices = shuffle_and_split_data(data_frame)

    # Display the number of samples in each set.
    print('=================================================================\n')
    print('training count: %s, validation count: %s\n' % (
          len(train_indices), len(validation_indices)))
    print('=================================================================')

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
                          DIMENSIONS,
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
                        help='Name of the sub-directory inside "datasets" holding the CAPTCHA images for training.')

    parser.add_argument('-l', '--length', type=int, choices=range(1, 6), nargs=1, required=True, 
                        help='Number of characters for each CAPTCHA image.')

    parser.add_argument('-e', '--epochs', type=int, nargs=1, required=True,
                        help='Number of epochs when training the model.')

    parser.add_argument('-b', '--batch_size', type=int, choices=[1, 16, 32, 64, 128], nargs=1, required=True,
                        help='Number of samples for the model at each iteration of training.')

    parser.add_argument('-a', '--model_architecture', 
                        choices=['VGG-16', 'MOBILE-NET','RESNET','CAPTCHA-NET', 'T-NET'], nargs=1, required=True,
                        help='Type of neural network architecture for the model.')

    parser.add_argument('-m', '--model_name', nargs=1, required=True,
                        help='Name of the model file when saving to disk.')

    parser.add_argument('-t', '--training_history_file_name', nargs=1, required=True,
                        help='Name of the destination file name for storing training history information.')

    # if no arguments were passed, show the help screen
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    return parser.parse_args()




def initialize_globals(arguments):
    """
        Assign the parsed command-line arguments to the 
        global variables.

    Args:
        arguments: the parsed command-line arguments
    """
    global DATA_DIRECTORY
    global DIMENSIONS
    global TRAINING_EPOCHS
    global TRAINING_BATCH_SIZE
    global VALIDATION_BATCH_SIZE
    global MODEL_NAME
    global MODEL_ARCHITECTURE
    global TRAINING_HISTORY_FILE_NAME

    DATA_DIRECTORY =  os.path.join(os.getcwd() + 
                                  '/datasets/' + arguments.data_directory[0])
    DIMENSIONS = arguments.length[0]
    TRAINING_EPOCHS = arguments.epochs[0]
    TRAINING_BATCH_SIZE = arguments.batch_size[0]
    VALIDATION_BATCH_SIZE = arguments.batch_size[0]
    MODEL_NAME = arguments.model_name[0]
    MODEL_ARCHITECTURE = arguments.model_architecture[0]
    TRAINING_HISTORY_FILE_NAME = arguments.training_history_file_name[0]




def get_trainable_neural_network():
    """
    Returns:
        the appropriate neural network architecture
        based on the value specified by the command-line argument
    """
    model = None

    if MODEL_ARCHITECTURE == 'VGG-16':
        model = create_VGG16_model(IMAGE_HEIGHT, 
                                   IMAGE_WIDTH, 
                                   IMAGE_CHANNELS,
                                   DIMENSIONS, 
                                   CATEGORIES)
    elif MODEL_ARCHITECTURE == 'MOBILE-NET':
        model = create_MOBILE_NET_model(IMAGE_HEIGHT,
                                       IMAGE_WIDTH,
                                       IMAGE_CHANNELS,
                                       DIMENSIONS,
                                       CATEGORIES)
    elif MODEL_ARCHITECTURE == 'RESNET':
        model = create_RESNET_model(IMAGE_HEIGHT,
                                    IMAGE_WIDTH,
                                    IMAGE_CHANNELS,
                                    DIMENSIONS,
                                    CATEGORIES)
    elif MODEL_ARCHITECTURE == 'CAPTCHA-NET':
        model = create_CAPTCHA_NET_model(IMAGE_HEIGHT, 
                                         IMAGE_WIDTH, 
                                         IMAGE_CHANNELS,
                                         DIMENSIONS, 
                                         CATEGORIES)
    elif MODEL_ARCHITECTURE == 'T-NET':  
        model = create_improved_CAPTCHA_NET_model(IMAGE_HEIGHT, 
                                                  IMAGE_WIDTH, 
                                                  IMAGE_CHANNELS,
                                                  DIMENSIONS, 
                                                  CATEGORIES)
    return model




if __name__ == '__main__':
    main()
