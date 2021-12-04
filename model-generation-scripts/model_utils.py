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
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import *
from tensorflow.keras.utils import to_categorical
from keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.vgg16 import VGG16

session = tensorflow.compat.v1.Session()




def get_captcha_label(file_path):
    """
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




def create_captcha_dataframe(captcha_images_directory):
    """
    Args:
        captcha_images_directory (str): the full file path to the folder where the captcha images
                                        were generated
    
    Returns:
        a pandas.DataFrame object storing each captcha file name along with its label
    """
    files = glob.glob(os.path.join(captcha_images_directory, '*.png'))
    attributes = list(map(get_captcha_label, files))

    data_frame = pandas.DataFrame(attributes)
    data_frame['file'] = files
    data_frame.columns = ['label', 'file']
    data_frame = data_frame.dropna()
    
    return data_frame




def shuffle_and_split_data(data_frame):
    """
        Shuffle and split the data into 2 sets: training and validation.
    
    Args:
        data_frame (pandas.DataFrame): the data to shuffle and split
    
    Returns:
        3 numpy.ndarray objects -> (train_indices, validation_indices)
        each hold the index positions for data in the pandas.DataFrame 
    """
    shuffled_indices = numpy.random.permutation(len(data_frame))

    train_up_to = int(len(data_frame) * 0.7)

    train_indices = shuffled_indices[:train_up_to]
    validation_indices = shuffled_indices[train_up_to:]

    return train_indices, validation_indices




def create_CAPTCHA_NET_model(image_height=100, image_width=100, image_channels=3, 
                             character_length=4, categories=10):

    input_layer = tensorflow.keras.Input(shape=(image_height, image_width, image_channels))

    hidden_layers = layers.Conv2D(32, 3, activation='relu')(input_layer)
    hidden_layers = layers.MaxPooling2D((2, 2))(hidden_layers)
    hidden_layers = layers.Conv2D(64, 3, activation='relu')(hidden_layers)
    hidden_layers = layers.MaxPooling2D((2, 2))(hidden_layers)
    hidden_layers = layers.Conv2D(64, 3, activation='relu')(hidden_layers)
    hidden_layers = layers.MaxPooling2D((2, 2))(hidden_layers)

    hidden_layers = layers.Flatten()(hidden_layers)

    hidden_layers = layers.Dense(1024, activation='relu')(hidden_layers)
    hidden_layers = layers.Dense(character_length * categories, activation='softmax')(hidden_layers)
    hidden_layers = layers.Reshape((character_length, categories))(hidden_layers)

    model = models.Model(inputs=input_layer, outputs=hidden_layers, name='CAPTCHA-NET')

    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy',
                  metrics= ['accuracy'])
    
    return model




def create_improved_CAPTCHA_NET_model(image_height=100, 
                                      image_width=100, 
                                      image_channels=1, 
                                      character_length=4, 
                                      categories=10):
    """
        Model creation function modified for new algorithm.
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




def create_VGG16_model(image_height=100, image_width=100, image_channels=3, 
                       character_length=4, categories=10):
    
    base_model = VGG16(weights='imagenet',
                       include_top=False,
                       input_shape=(image_height + 10, image_width, image_channels))
    base_model.trainable = False

    flatten_layer = layers.Flatten()
    dropout_layer = Dropout(0.5)
    prediction_layer = Dense(categories, activation='softmax')

    model = Sequential([
        base_model,
        flatten_layer,
        dropout_layer,
        prediction_layer
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model



def get_captcha_generator(data_frame, indices, for_training, batch_size=16, image_height=100, image_width=100,
                                    categories=10):
    """    
    Args:
        data_frame (pandas.DataFrame): contains the file paths to the CAPTCHA images and their labels
        
        indices (int): specifies training indices, testing indices, or validation indices of the DataFrame
        
        for_training (bool): 'True' for training or validation set, 'False' to specify a test set 
        
        batch_size (int): number of data instances to return when iterated upon
        
        image_height (int): height in pixels to resize the CAPTCHA image to
        
        image_width (int): width in pixels to resize the CAPTCHA image to
        
        categories (int): number of possible values for each position in the CAPTCHA image
    
    Returns:
        a concrete iterator which is responsible for traversing over a CAPTCHA
        dataset
        
    Yields:
        a pair of lists -> (CAPTCHA images, labels)
    """
    images, labels = [], []
    
    while True:
        for i in indices:
            captcha = data_frame.iloc[i]
            file, label = captcha['file'], captcha['label']
            
            # Open and convert the CAPTCHA image to gray-scale.
            captcha_image = Image.open(file).convert('L')
        
            # Resize, in-case the generated CAPTCHA image has different
            # dimensions than what the neural network expects.
            captcha_image = captcha_image.resize((image_height, image_width))
                        
            # Get the attacher images which will be binded to the CAPTCHA image copies.
            attacher_images = _get_attacher_images(image_height, image_width, len(label))

            for j in range(len(label)):
                # Create a new gray-scale image which will combine the 
                # CAPTCHA image and meta-data image.
                #combined_image = Image.new('L', (image_width, (image_height + 10)), 'white')
                combined_image = Image.new('RGB', (image_width, (image_height + 10)), 'white')

                # Paste the CAPTCHA image first.
                combined_image.paste(captcha_image, (0, 0))

                # Paste the attacher image underneath the CAPTCHA image.
                combined_image.paste(attacher_images[j], (0, image_height))
 
                # Normalize the pixel values of the resulting image to values
                # in the range (0, 1) (inclusive).
                combined_image = numpy.array(combined_image) / 255.0

                # By default, gray-scale images which are converted to numpy
                # arrays will only contain two dimensions (height, width). 
                #
                # This instruction will manually add a third dimension 
                # (color channel) since it is required by the neural network.
                #
                # The value '1' specifies a single color channel for gray-scale
                # images.
                #combined_image = combined_image.reshape(*combined_image.shape, 1)

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
    Args:
        captcha_height (int): height (in pixels) of the CAPTCHA image
        captcha_width (int): width (in pixels) of the CAPTCHA image
        character_length (int): number of characters in the CAPTCHA image

    Returns:
        a list of 'attacher' images responsible for assisting in single-character recognition
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




def train_model(model, 
                data_frame, 
                train_indices, 
                validation_indices, 
                training_batch_size,
                validation_batch_size,
                training_epochs,
                image_height, 
                image_width, 
                character_length, 
                categories):
    
    training_set_generator = get_captcha_generator(data_frame, 
                                                   train_indices,
                                                   for_training=True, 
                                                   batch_size=training_batch_size,
                                                   image_height=image_height,
                                                   image_width=image_width,
                                                   categories=categories)
    
    validation_set_generator = get_captcha_generator(data_frame, 
                                                     validation_indices,
                                                     for_training=True, 
                                                     batch_size=validation_batch_size,
                                                     image_height=image_height,
                                                     image_width=image_width,
                                                     categories=categories)

    callbacks = [
        ModelCheckpoint("./model_checkpoint", monitor='val_loss')
    ]

    history = model.fit(training_set_generator,
                        steps_per_epoch=len(train_indices * character_length) // training_batch_size,
                        epochs=training_epochs,
                        callbacks=callbacks,
                        validation_data=validation_set_generator,
                        validation_steps=len(validation_indices * character_length) // validation_batch_size)
    
    return history
