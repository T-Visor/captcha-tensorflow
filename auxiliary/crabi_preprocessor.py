import numpy
from PIL import Image, ImageDraw
from tensorflow.keras.utils import to_categorical

ATTACHER_HEIGHT = 10

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
                combined_image = Image.new('RGB', (image_width, (image_height + ATTACHER_HEIGHT)), 'white')

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
    attacher_height = ATTACHER_HEIGHT
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
