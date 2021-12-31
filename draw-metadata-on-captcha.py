#!/usr/bin/env python3
from PIL import Image, ImageDraw

# importing image object from PIL
import math
import cv2
from PIL import Image, ImageDraw
  
width, height = 100, 10
character_length = 4
left_side = 0
right_side = width

metadata_images = []

for i in range(character_length):
    # Start and end coordinates.
    rectangle_shape = [(left_side, 0), ((right_side / character_length), height)]

    # Create the base image.
    metadata_images.append(Image.new('L', (width, height), color='grey'))
  
    # Draw the rectangle.
    rectangle_drawer = ImageDraw.Draw(metadata_images[i])
    rectangle_drawer.rectangle(rectangle_shape, fill ='#ffffff')

    # Move to a new set of image coordinates for drawing the next metadata image.
    left_side = (right_side / character_length) 
    right_side += width

for i in range(character_length):
    # Load the CAPTCHA image.
    captcha_image = Image.open('1234_5ee7b284-3dfa-420f-b4b1-d32ca48d9459.png').convert('L')

    # Create a new image which combines the CAPTCHA image and metadata image.
    combined_image = Image.new('L', (100, 110), 'white')

    # Paste the CAPTCHA image first.
    combined_image.paste(captcha_image, (0, 0))

    # Paste the metadata image underneath the CAPTCHA image.
    combined_image.paste(metadata_images[i], (0, 100))

    # Save the resulting.
    combined_image.save('combined-image' + str(i) + '.png')
