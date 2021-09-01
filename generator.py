#/usr/bin/env python3

from PIL import Image, ImageDraw, ImageFont
import argparse
import os
import random 
import string
import sys

# RGB values for custom colors
FOREST_GREEN = (64, 107, 76)
SEA_BLUE = (0, 87, 128)
DARK_INDIGO = (0, 3, 82)
PINK = (191, 0, 255)
LIGHT_GREEN = (72, 189, 0)
ORANGE = (189, 107, 0)
RED = (189, 41, 0)
DARK_BLUE = (0, 3, 82)

# lambda function - used to pick a random location in image
getit = lambda : (random.randrange(0, 80), random.randrange(0, 60))

POINT_COLORS = ['black', 'red', 'blue', 'green', FOREST_GREEN, SEA_BLUE, DARK_BLUE]
LINE_POINT_COLORS = [FOREST_GREEN, SEA_BLUE, DARK_INDIGO, PINK, LIGHT_GREEN, ORANGE, RED]
FONT_NAME = ''
DESTINATION_DIRECTORY = 'datasets/'




def main():
    parser = parse_commandline_argument()

    global FONT_NAME 
    FONT_NAME = parser.font[0]

    global DESTINATION_DIRECTORY
    DESTINATION_DIRECTORY = DESTINATION_DIRECTORY + os.path.splitext(FONT_NAME)[0]

    print('Saving to: ' + DESTINATION_DIRECTORY)

    os.makedirs(DESTINATION_DIRECTORY, exist_ok=True)

    print('Generating CAPTCHA images')
    i = 1
    while (i < 10000):
        generate_captcha_image()
        i += 1




def parse_commandline_argument():
    """ 
        Parse the arguments from the command-line.

        If no arguments are passed, the help screen will
        be shown and the program will be terminated.

    Returns:
        the parser with the command-line argument
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('-f', '--font', nargs=1, required=True, 
                        help='The font file with extension ".ttf" to use \
                              (example: "DejaVuSansMono.ttf")')
    
    # if no arguments were passed, show the help screen
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    return parser.parse_args()




def generate_captcha_image():
    
    # create a colored 100x100 image with white background
    captcha_image = Image.new('RGB', (80, 60), color="white")
    illustrater = ImageDraw.Draw(captcha_image)

    # generate a randomly colored string
    captcha_string = generate_random_string(4, 'numeric')
    get_colored_text(illustrater, captcha_string)

    # draw some random lines
    for i in range(5,random.randrange(6, 10)):
        illustrater.line((getit(), getit()), fill=random.choice(LINE_POINT_COLORS), width=random.randrange(1,3))

    # draw some random points
    for i in range(10,random.randrange(11, 20)):
        illustrater.point((getit(), getit(), getit(), getit(), getit(), getit(), getit(), getit(), getit(), getit()), fill=random.choice(POINT_COLORS))

    # save the newly generate CAPTCHA image
    captcha_image.save(DESTINATION_DIRECTORY + '/' + captcha_string + '_image.png')




def generate_random_string(captcha_length: int, character_set: str) -> str:

    captcha_string = None

    if character_set == 'alphanumeric':
        captcha_string += (string.ascii_lowercase + string.ascii_uppercase)
    elif character_set == 'alphabet':
        captcha_string += (string.ascii_lowercase + string.ascii_uppercase)
    else:
        captcha_string = string.digits

    generate_random_string = ''.join(random.choices(captcha_string, k=captcha_length))
    return generate_random_string




def get_colored_text(illustrater, captcha_string):
    text_colors = random.choice(POINT_COLORS)
    font = ImageFont.truetype('fonts/' + FONT_NAME, 18)
    illustrater.text((20,20), captcha_string, fill=text_colors, font=font)




main()
