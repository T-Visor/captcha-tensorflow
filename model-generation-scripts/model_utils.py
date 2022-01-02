#!/usr/bin/env python3

import glob
import matplotlib.pyplot as pyplot
import math
import numpy
import os
import pandas
import tensorflow

from PIL import Image, ImageDraw
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.applications import MobileNet, ResNet50, VGG16

session = tensorflow.compat.v1.Session()




def create_captcha_dataframe(captcha_images_directory):
    """
    Args:
        captcha_images_directory (str): the full file path to the folder where the captcha images
                                        were generated
    
    Returns:
        a pandas.DataFrame object storing each captcha file name along with its label
    """
    files = glob.glob(os.path.join(captcha_images_directory, '*.png'))
    attributes = list(map(_get_captcha_label, files))

    captcha_dataframe = pandas.DataFrame(attributes)
    captcha_dataframe['file'] = files
    captcha_dataframe.columns = ['label', 'file']
    captcha_dataframe = captcha_dataframe.dropna()
    
    return captcha_dataframe




def _get_captcha_label(file_path):
    """
    (HELPER FUNCTION)

    Precondition: CAPTCHA images were generated using the
                  script found in this project folder
    
    Args:
        file_path (str): the path to the CAPTCHA image
    
    Returns:
        the 'label' for each CAPTCHA denoted by the 
        string in the file name before the '_'
        character

        Example: '9876_image.png' -> '9876' 
    """
    try:
        path, file_name = os.path.split(file_path)
        file_name, extension = os.path.splitext(file_name)
        label, _ = file_name.split("_")
        return label
    except Exception as e:
        print('error while parsing %s. %s' % (file_path, e))
        return None, None




def shuffle_and_split_data(data_frame):
    """
        Shuffle and split the data into 2 sets: training and validation.
    
    Args:
        data_frame (pandas.DataFrame): the data to shuffle and split
    
    Returns:
        2 numpy.ndarray objects -> (train_indices, validation_indices)
        Each hold the index positions for data in the pandas.DataFrame 
    """
    shuffled_indices = numpy.random.permutation(len(data_frame))

    train_up_to = int(len(data_frame) * 0.7)

    train_indices = shuffled_indices[:train_up_to]
    validation_indices = shuffled_indices[train_up_to:]

    return train_indices, validation_indices




def build_TNET_model(image_height, 
                     image_width, 
                     image_channels, 
                     character_length, 
                     categories):
    """
        Builds a simple Convolutional Neural Network (CNN) to recognize CAPTCHA images.

        Arguments to this function specify the characteristics of the input and
        output layers.

        Postcondition: Model must be trained after being built

    Args:
        image_height (int): height (in pixels) of expected input CAPTCHA image 

        image_width (int): width (in pixels) of expected input CAPTCHA image

        image_channels (int): channel count of expected input CAPTCHA image 
                              ('3' for RGB, '1' for grayscale)

        character_length (int): number of characters in expected input CAPTCHA image

        categories (int): number of possible characters in expected input
                          CAPTCHA image, specifying category count in the output layer
                          ('10' for digits 0-9, '26' for alphabet, '36' for alphanumeric)

    Returns:
        a compiled model ready for training
    """
    model = Sequential(name='T-NET')

    model.add(Input(shape=((image_height + 10), image_width, image_channels)))
    
    model.add(Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

    model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))

    model.add(Flatten())
    
    model.add(Dense(units=1024,activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(categories, activation='softmax'))
    
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy',
                  metrics= ['accuracy'])
    
    return model




def build_transfer_learning_model(model_architecture_name,
                                  image_height, 
                                  image_width, 
                                  image_channels,
                                  character_length, 
                                  categories):
    """
        Builds a Convolutional Neural Network (CNN) using a pre-trained model
        (transfer learning) with weights established from ImageNet. The model
        will be restructured to recognize CAPTCHA images.

        Arguments to this function specify the characteristics of the model
        architecture, input layer, and output layer.

        Postcondition: Model must be trained after being built

    Args:
        model_architecture_name (string): name of the pre-trained model to be used
                                          (e.g. MobileNet, VGG16, ResNet50)

        image_height (int): height (in pixels) of expected input CAPTCHA image 

        image_width (int): width (in pixels) of expected input CAPTCHA image

        image_channels (int): channel count of expected input CAPTCHA image 
                              ('3' for RGB, '1' for grayscale)

        character_length (int): number of characters in expected input CAPTCHA image

        categories (int): number of possible characters in expected input
                          CAPTCHA image, specifying category count in the output layer
                          ('10' for digits 0-9, '26' for alphabet, '36' for alphanumeric)

    Returns:
        a compiled model ready for training
    """
    if model_architecture_name == 'MOBILE-NET':
        base_model = MobileNet(weights='imagenet',
                               include_top=False,
                               input_shape=(image_height + 10, 
                                            image_width, 
                                            image_channels))
    elif model_architecture_name == 'RESNET':
        base_model = ResNet50(weights='imagenet',
                              include_top=False,
                              input_shape=(image_height + 10, 
                                           image_width, 
                                           image_channels))
    elif model_architecture_name == 'VGG16':
        base_model = VGG16(weights='imagenet',
                           include_top=False,
                           input_shape=(image_height + 10, 
                                        image_width, 
                                        image_channels))

    flatten_layer = Flatten()
    prediction_layer = Dense(categories, activation='softmax')

    model = Sequential([
        base_model,
        flatten_layer,
        prediction_layer
    ])

    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    return model




def train_captcha_recognition_model(model, 
                                    captcha_dataframe, 
                                    train_indices, 
                                    validation_indices, 
                                    batch_size,
                                    training_epochs,
                                    image_height, 
                                    image_width, 
                                    character_length, 
                                    categories):                   
    """
        Train a Deep Learning model to recognize CAPTCHA images preprocessed
        with the CRABI algorithm using supervised learning.

    Args:
        model (tensorflow.keras.Model): the Deep Learning model to train

        captcha_dataframe (pandas.DataFrame): the dataset for training
    
        train_indices (numpy.ndarray): indices of the CAPTCHA dataset used for training data

        validation_indices (numpy.ndarray): indices of the CAPTCHA dataset used for validation data

        batch_size (int): number of samples to process before the model is updated

        training_epochs (int): number of passes through the entire dataset

        image_height (int): height (in pixels) of expected input CAPTCHA image 

        image_width (int): width (in pixels) of expected input CAPTCHA image

        character_length (int): number of characters in expected input CAPTCHA image

        categories (int): number of possible characters in expected input
                          CAPTCHA image, specifying category count in the output layer
                          ('10' for digits 0-9, '26' for alphabet, '36' for alphanumeric)

    Returns:
        a History object which contains accuracy/loss information from training
    """
    training_set_iterator, validation_set_iterator = _get_CRABI_iterators(captcha_dataframe, 
                                                                          train_indices, 
                                                                          validation_indices,
                                                                          batch_size, 
                                                                          image_height, 
                                                                          image_width,
                                                                          character_length,
                                                                          categories)

    callbacks = [
        ModelCheckpoint("./model_checkpoint", monitor='val_loss')
    ]

    history = model.fit(training_set_iterator,
                        steps_per_epoch=len(train_indices * character_length) // batch_size,
                        epochs=training_epochs,
                        callbacks=callbacks,
                        validation_data=validation_set_iterator,
                        validation_steps=len(validation_indices * character_length) // batch_size)
    
    return history




def _get_CRABI_iterators(captcha_dataframe,
                         train_indices,
                         validation_indices,
                         batch_size,
                         image_height, 
                         image_width, 
                         character_length, 
                         categories):
    """
        (HELPER FUNCTION)

    Args:
        captcha_dataframe (pandas.DataFrame): the dataset for training
    
        train_indices (numpy.ndarray): indices of the CAPTCHA dataset used for training data

        validation_indices (numpy.ndarray): indices of the CAPTCHA dataset used for validation data

        batch_size (int): number of samples to process before the model is updated

        image_height (int): height (in pixels) of expected input CAPTCHA image 

        image_width (int): width (in pixels) of expected input CAPTCHA image

        character_length (int): number of characters in expected input CAPTCHA image

        categories (int): number of possible characters in expected input
                          CAPTCHA image, specifying category count in the output layer
                          ('10' for digits 0-9, '26' for alphabet, '36' for alphanumeric)

    Returns:
        pair of generator objects -> (training_set_iterator, validation_set_iterator)  
    """

    training_set_iterator = generate_CRABI_preprocessed_images(captcha_dataframe, 
                                                               train_indices,
                                                               for_training=True, 
                                                               batch_size=batch_size,
                                                               image_height=image_height,
                                                               image_width=image_width,
                                                               categories=categories)
    
    validation_set_iterator = generate_CRABI_preprocessed_images(captcha_dataframe, 
                                                                 validation_indices,
                                                                 for_training=True, 
                                                                 batch_size=batch_size,
                                                                 image_height=image_height,
                                                                 image_width=image_width,
                                                                 categories=categories)

    return training_set_iterator, validation_set_iterator




def generate_CRABI_preprocessed_images(captcha_dataframe, 
                                       indices, 
                                       for_training, 
                                       batch_size=16, 
                                       image_height=100, 
                                       image_width=100,
                                       categories=10):
    """    
        (GENERATOR FUNCTION)

        Creates an iterator object, which will generate CAPTCHA images using a
        variation of the CRABI (CAPTCHA Recognition with Attached Binary
        Images) algorithm. 

        This mechanism is used to facilitate CAPTCHA-recognition on a per-character basis, 
        by attaching black bars with markers to the bottom of CAPTCHA image copies, and giving each
        image a single-character label.

    Args:
        captcha_dataframe (pandas.DataFrame): contains the file paths to the CAPTCHA images and their labels
        
        indices (int): specifies training indices, testing indices, or validation indices of the DataFrame
        
        for_training (bool): 'True' for training or validation set, 'False' to specify a test set 
        
        batch_size (int): number of data instances to return when iterated upon
        
        image_height (int): height in pixels to resize the CAPTCHA image to
        
        image_width (int): width in pixels to resize the CAPTCHA image to
        
        categories (int): number of possible values for each position in the CAPTCHA image
    
    Returns:
        a concrete iterator to traverse over a CAPTCHA dataset
        
    Yields:
        a pair of lists -> (CRABI-preprocessed CAPTCHA images, single-character labels)
    """
    images, labels = [], []
    
    while True:
        for i in indices:
            captcha = captcha_dataframe.iloc[i]
            file, label = captcha['file'], captcha['label']
            
            captcha_image = Image.open(file).convert('L') # open CAPTCHA image in gray-scale
            captcha_image = captcha_image.resize((image_height, image_width))
                        
            # Get the black bars with marker images to attach to the bottom of each
            # CAPTCHA image copy.
            attacher_images = _get_attacher_images(image_height, image_width, len(label))

            for j in range(len(label)):
                # Create a blank image for CRABI-preprocessing.
                combined_image = Image.new('RGB', (image_width, (image_height + 10)), 'white')

                # Paste the CAPTCHA image first.
                combined_image.paste(captcha_image, (0, 0))

                # Paste the black bar with marker underneath the CAPTCHA image.
                combined_image.paste(attacher_images[j], (0, image_height))
 
                # Normalize the pixel values of the resulting image to values
                # in the range (0, 1) (inclusive).
                combined_image = numpy.array(combined_image) / 255.0

                # Add the resulting image to the current batch.
                images.append(numpy.array(combined_image))

                # Add a 1-character label for the current image.
                labels.append(numpy.array(to_categorical(int(label[j]), categories)))

            if len(images) >= batch_size: 
                yield numpy.array(images), numpy.array(labels)   # Return the current batch.
                images, labels = [], []                          # Make both lists empty for the next batch.
                
        if not for_training:
            break




def _get_attacher_images(captcha_height, captcha_width, character_length):
    """
        (HELPER FUNCTION)

    Args:
        captcha_height (int): height (in pixels) of the CAPTCHA image
        captcha_width (int): width (in pixels) of the CAPTCHA image
        character_length (int): number of characters in the CAPTCHA image

    Returns:
        a list of black bar images with markers responsible for assisting in single-character recognition
        in a CAPTCHA image. These generated images are referred to as the 'external binary images'
        in the implementation of CRABI (CAPTCHA Recognition With Attached Binary Images).
    """
    attacher_width = captcha_width
    attacher_height = 10
    left_side = 0
    right_side = captcha_width
    attacher_images = []

    for i in range(character_length):
        # Start and end coordinates.
        rectangle_shape = [(left_side, 0), ((right_side / character_length), attacher_height)]

        # Create the base image.
        attacher_images.append(Image.new('L', (attacher_width, attacher_height), color='grey'))
  
        # Draw the rectangle.
        rectangle_drawer = ImageDraw.Draw(attacher_images[i])
        rectangle_drawer.rectangle(rectangle_shape, fill ='#ffffff')

        # Move to a new set of image coordinates for drawing the next attacher image.
        left_side = (right_side / character_length) 
        right_side += attacher_width

    return attacher_images
