import matplotlib.pyplot as pyplot
import math
import tensorflow

from crabi_preprocessing_utils import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.applications import MobileNet, ResNet50, VGG16

session = tensorflow.compat.v1.Session()




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

    model.add(Input(shape=((image_height + ATTACHER_HEIGHT), image_width, image_channels)))
    
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
                               input_shape=(image_height + ATTACHER_HEIGHT, 
                                            image_width, 
                                            image_channels))
    elif model_architecture_name == 'RESNET':
        base_model = ResNet50(weights='imagenet',
                              include_top=False,
                              input_shape=(image_height + ATTACHER_HEIGHT, 
                                           image_width, 
                                           image_channels))
    elif model_architecture_name == 'VGG16':
        base_model = VGG16(weights='imagenet',
                           include_top=False,
                           input_shape=(image_height + ATTACHER_HEIGHT, 
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
