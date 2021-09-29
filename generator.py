#!/usr/bin/env python3

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm, tnrange
import argparse
import os
import random 
import string
import sys
import uuid

#==========================================================================================
# GLOBAL VARIABLES

FOREST_GREEN = (64, 107, 76)
SEA_BLUE = (0, 87, 128)
DARK_INDIGO = (0, 3, 82)
PINK = (191, 0, 255)
LIGHT_GREEN = (72, 189, 0)
ORANGE = (189, 107, 0)
RED = (189, 41, 0)
DARK_BLUE = (0, 3, 82)
POINT_COLORS = ['black', 'red', 'blue', 'green', FOREST_GREEN, SEA_BLUE, DARK_BLUE]
LINE_POINT_COLORS = [FOREST_GREEN, SEA_BLUE, DARK_INDIGO, PINK, LIGHT_GREEN, ORANGE, RED]

IMAGE_HEIGHT = 100
IMAGE_WIDTH = 100

FONTS = [r'fonts/DejaVuSans.ttf',
         r'fonts/DejaVuSerif.ttf',
         r'fonts/OpenSans-Bold.ttf',
         r'fonts/OpenSans-Light.ttf',
         r'fonts/System San Francisco Display Regular.ttf',
         r'fonts/MesloLGS-NF-Italic.ttf'
        ]

# Variables which will be set from command-line arguments.
DESTINATION_DIRECTORY = None
ITERATIONS = None
CAPTCHA_LENGTH = None

# '1' represents the most significant digit,
# append this value with 0's to determine the total
# number of unique CAPTCHAs to generate.
UNIQUE_VALUES = '1' 

#==========================================================================================

# Get a random location on the CAPTCHA image
get_image_location = lambda : (random.randrange(0, IMAGE_WIDTH),
                               random.randrange(0, IMAGE_HEIGHT))

# Get a random font
get_font = lambda : (random.choice(FONTS)) 




def main():
    """
        Generate the CAPTCHA dataset.
    """
    global UNIQUE_VALUES
    global CAPTCHA_LENGTH
    global DESTINATION_DIRECTORY

    # Get the command-line arguments.
    arguments = parse_command_line_arguments()

    # Assign values retrieved from the command-line.
    ITERATIONS = arguments.iterations[0]
    CAPTCHA_LENGTH = arguments.length[0]
    DESTINATION_DIRECTORY = 'datasets/' + arguments.destination[0] + '/'

    # Destination directory to store CAPTCHA image dataset.
    os.makedirs(DESTINATION_DIRECTORY, exist_ok=True)

    # Append zeroes to get the total number of unique 
    # CAPTCHA instances.
    for i in range(0, CAPTCHA_LENGTH):
        UNIQUE_VALUES += '0'

    # Convert from string to number.
    UNIQUE_VALUES = int(UNIQUE_VALUES)

    # Display CAPTCHA generation information.
    print('--------------------------------------')
    print('GENERATING CAPTCHA IMAGES\n')
    print('Unique values: {}'.format(str(UNIQUE_VALUES)))
    print('CAPTCHA character length: {}'.format(str(CAPTCHA_LENGTH)))
    print('Iterations: {}'.format(str(ITERATIONS)))
    print('Total number of samples: {}\n'.format(str(ITERATIONS * UNIQUE_VALUES)))
    print('Saving to: ' + DESTINATION_DIRECTORY)
    print('--------------------------------------')

    # Generate the numeric CAPTCHAs.
    with tqdm(total=(ITERATIONS * UNIQUE_VALUES)) as progress_bar:
        for _ in range(0, ITERATIONS):
            for i in range (0, UNIQUE_VALUES):
                generate_numeric_captcha_image(i)
                progress_bar.update(1)




def parse_command_line_arguments():
    """ 
        Parse the arguments from the command-line.

        If no arguments are passed, the help screen will
        be shown and the program will be terminated.

    Returns:
        the parser with command-line arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--iterations', type=int, nargs=1, required=True, 
                        help='The number of times to repeat unique CAPTCHA set generation.')

    parser.add_argument('-l', '--length', type=int, nargs=1, required=True, 
                        help='Number of characters for each CAPTCHA image.')

    parser.add_argument('-d', '--destination', nargs=1, required=True,
                        help='The name of the destination directory within \
                              "datasets" to save the CAPTCHA images to')

    # if no arguments were passed, show the help screen
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    return parser.parse_args()




def generate_numeric_captcha_image(number):
    """
        Create a CAPTCHA image sample by drawing out colored digits, and
        make obstructions by placing random colored dots and lines throughout.

    Args:
        number (str): series of one or more digits represented as a single string
    """
    # Create a colored image with white background.
    captcha_image = Image.new('RGB', (IMAGE_WIDTH, IMAGE_HEIGHT), color="white")
    illustrator = ImageDraw.Draw(captcha_image)

    # Draw the colored number and fill any digit places
    # according to the specified CAPTCHA length.
    #
    # EXAMPLE: if '25' was the number passed and the total
    #          CAPTCHA length is '4', its new value is '0025'.
    captcha_text = str(number)
    captcha_text = captcha_text.zfill(CAPTCHA_LENGTH)
    draw_colored_text(illustrator, captcha_text)

    # Draw some random lines.
    for _ in range(5,random.randrange(6, 10)):
        illustrator.line((get_image_location(), get_image_location()), fill=random.choice(LINE_POINT_COLORS), width=random.randrange(1,3))

    # Draw some random points.
    for _ in range(10,random.randrange(11, 20)):
        illustrator.point((get_image_location(), get_image_location(), get_image_location(), get_image_location(), 
                           get_image_location(), get_image_location(), get_image_location(), 
                           get_image_location(), get_image_location(), get_image_location()), 
                           fill=random.choice(POINT_COLORS))

    # Save the newly generate CAPTCHA image with a unique identifier.
    captcha_image.save(DESTINATION_DIRECTORY + '/' + captcha_text + '_' +
                       str(uuid.uuid4()) + '.png')




def draw_colored_text(illustrator, captcha_string):
    """
        Draw colored characters on the CAPTCHA image using the supplied
        CAPTCHA string.
        
    Args:
        illustrator: object capable of drawing on the CAPTCHA image
        captcha_string: the characters to write on the CAPTCHA image
    """
    x_position = IMAGE_WIDTH / (CAPTCHA_LENGTH + 2)
    y_position = IMAGE_HEIGHT / 3 # middle height
    font_size = int(IMAGE_HEIGHT * 0.3)

    for i in range(len(captcha_string)):
        text_color = random.choice(POINT_COLORS)
        character = captcha_string[i]
        font = ImageFont.truetype(get_font(), font_size)
        illustrator.text((x_position, y_position), character, fill=text_color, font=font)
        x_position = x_position + (IMAGE_WIDTH / (CAPTCHA_LENGTH + 1))




main()
