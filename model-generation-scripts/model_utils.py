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

session = tensorflow.compat.v1.Session()




def get_captcha_label(file_path):
    """
    Precondition: CAPTCHA images were generated using the
                  'generator.py' script found in this project folder
    
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
    files = glob.glob(os.path.join(captcha_images_directory, "*.png"))
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




def _create_initial_convolutional_layers(input_shape):
    x = layers.Conv2D(16, 3, activation='relu')(input_shape)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = Model(inputs=input_shape, outputs=x)

    return x




def create_VGG16_model(image_height=100, image_width=100, image_channels=3, 
                       character_length=4, categories=10):
    
    model = Sequential(name='VGG-16')
    
    model.add(Input(shape=(image_height, image_width, image_channels)))

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(data_format="channels_last", pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(data_format="channels_last", pool_size=(2, 2)))
    
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(data_format="channels_last", pool_size=(2, 2)))
    
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D(data_format="channels_last", pool_size=(2, 2)))

    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    #model.add(MaxPooling2D(data_format="channels_last", pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(character_length * categories))
    model.add(Activation('softmax'))
    
    model.add(Reshape((character_length, categories)))

    optimizer = RMSprop(lr=1e-4)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model




def get_alternate_captcha_generator(data_frame, indices, for_training, batch_size=16, image_height=100, image_width=100,
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
        a generator object for producing CAPTCHA images along with their labels
        
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
                        
            # Get the meta-data images to attach to the CAPTCHA image copies.
            metadata_images = _create_metadata_images(image_height, image_width, len(label))

            for j in range(len(label)):
                # Create a new gray-scale image which will combine the 
                # CAPTCHA image and meta-data image.
                combined_image = Image.new('L', (image_width, (image_height + 10)), 'white')

                # Paste the CAPTCHA image first.
                combined_image.paste(captcha_image, (0, 0))

                # Paste the meta-data image underneath the CAPTCHA image.
                combined_image.paste(metadata_images[j], (0, image_height))
 
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
                combined_image = combined_image.reshape(*combined_image.shape, 1)

                # Add the resulting image to the current batch.
                images.append(numpy.array(combined_image))

                # Add a 1-character label for the current image.
                labels.append(numpy.array(to_categorical(int(label[j]), categories)))

            if len(images) >= batch_size: 
                yield numpy.array(images), numpy.array(labels)   # Return the current batch.
                images, labels = [], []                          # Make both lists empty for the next batch.
                
        if not for_training:
            break




def _create_metadata_images(captcha_height, captcha_width, character_length):
    meta_width = captcha_width
    meta_height = 10
    left_side = 0
    right_side = captcha_width
    metadata_images = []

    for i in range(character_length):
        # Start and end coordinates.
        rectangle_shape = [(left_side, 0), ((right_side / character_length), meta_height)]

        # Create the base image.
        metadata_images.append(Image.new('L', (meta_width, meta_height), color='grey'))
  
        # Draw the rectangle.
        rectangle_drawer = ImageDraw.Draw(metadata_images[i])
        rectangle_drawer.rectangle(rectangle_shape, fill ='#ffffff')

        # Move to a new set of image coordinates for drawing the next metadata image.
        left_side = (right_side / character_length) 
        right_side += meta_width

    return metadata_images






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
        a generator object for producing CAPTCHA images along with their labels
        
    Yields:
        a pair of lists -> (CAPTCHA images, labels)
    """
    images, labels = [], []
    
    while True:
        for i in indices:
            captcha = data_frame.iloc[i]
            file, label = captcha['file'], captcha['label']
            
            captcha_image = Image.open(file)
            captcha_image = captcha_image.resize((image_height, image_width))
            captcha_image = numpy.array(captcha_image) / 255.0
            
            images.append(numpy.array(captcha_image))
            labels.append(numpy.array([numpy.array(to_categorical(int(i), categories)) for i in label]))
            
            if len(images) >= batch_size:
                yield numpy.array(images), numpy.array(labels)
                images, labels = [], []
                
        if not for_training:
            break




# TODO: add parameters to satisfy what is required for 'get_captcha_generator' function
def train_model(model, data_frame, train_indices, validation_indices, 
                training_batch_size, validation_batch_size, training_epochs,
                image_height, image_width, categories):
    
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
                        steps_per_epoch=len(train_indices)//training_batch_size,
                        epochs=training_epochs,
                        callbacks=callbacks,
                        validation_data=validation_set_generator,
                        validation_steps=len(validation_indices)//validation_batch_size)
    
    return history




# TODO: add parameters to satisfy what is required for 'get_captcha_generator' function
def train_model_alternative(model, data_frame, train_indices, validation_indices, 
                            training_batch_size, validation_batch_size, training_epochs,
                            image_height, image_width, character_length, categories):
    
    training_set_generator = get_alternate_captcha_generator(data_frame, 
                                                   train_indices,
                                                   for_training=True, 
                                                   batch_size=training_batch_size,
                                                   image_height=image_height,
                                                   image_width=image_width,
                                                   categories=categories)
    
    validation_set_generator = get_alternate_captcha_generator(data_frame, 
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




def plot_training_history(history):
    figure, axes = pyplot.subplots(1, 2, figsize=(20, 5))

    axes[0].plot(history.history['acc'], label='Training accuracy')
    axes[0].plot(history.history['val_acc'], label='Validation accuracy')
    axes[0].set_xlabel('Epochs')
    axes[0].legend() 

    axes[1].plot(history.history['loss'], label='Training loss')
    axes[1].plot(history.history['val_loss'], label='Validation loss')
    axes[1].set_xlabel('Epochs')
    axes[1].legend()




# TODO: add parameters to satisfy what is required for 'get_captcha_generator' function
def get_prediction_results(model, data_frame, test_indices, testing_batch_size, 
                           image_height, image_width, categories):
    
    testing_set_generator = get_captcha_generator(data_frame, 
                                                  test_indices,
                                                  for_training=True, 
                                                  batch_size=testing_batch_size,
                                                  image_height=image_height,
                                                  image_width=image_width,
                                                  categories=categories)

    captcha_images, captcha_text = next(testing_set_generator)

    predictions = model.predict_on_batch(captcha_images)

    true_values = tensorflow.math.argmax(captcha_text, axis=-1)
    predictions = tensorflow.math.argmax(predictions, axis=-1)
    
    return captcha_images, predictions, true_values




def display_predictions_from_model(captcha_images, predictions, true_values, total_to_display=30, columns=5):
    """
        Display a plot showing the results of the model's predictions.
        Each subplot will contain the CAPTCHA image, the model's prediction value, and the true value (label).
        
    Args:
        captcha_images (PNG image): the CAPTCHA image file
        
        predictions (EagerTensor): the prediction value made by the model
        
        true_values (EagerTensor): the label associated with the CAPTCHA image
        
        total_to_display (int): total number of subplots
        
        columns (int): number of columns in the plot
    """
    
    with session.as_default():
        random_indices = numpy.random.permutation(total_to_display)
        rows = math.ceil(total_to_display / columns)

        figure, axes = pyplot.subplots(rows, columns, figsize=(15, 20))
    
        for i, image_index in enumerate(random_indices):
            result = axes.flat[i]
            result.imshow(captcha_images[image_index])
        
            if tensorflow.executing_eagerly():
                result.set_title('prediction: {}'.format(
                                 ''.join(map(str, predictions[image_index].numpy()))))
                result.set_xlabel('true value: {}'.format(
                                  ''.join(map(str, true_values[image_index].numpy()))))
            else:
                result.set_title('prediction: {}'.format(
                                 ''.join(map(str, predictions[image_index].eval()))))
                result.set_xlabel('true value: {}'.format(
                                  ''.join(map(str, true_values[image_index].eval()))))

            result.set_xticks([])
            result.set_yticks([])
