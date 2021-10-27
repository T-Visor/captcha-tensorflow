#!/usr/bin/env python3
from PIL import Image, ImageDraw

# importing image object from PIL
import math
import cv2
from PIL import Image, ImageDraw
  
w, h = 100, 20
character_length = 4
left_side = 0
right_side = w


for i in range(character_length):
    #shape = [(0, 0), ((w / 4), h)] # start and end coordinates
    shape = [(left_side, 0), ((right_side / character_length), h)]

    # Create base image.
    img = Image.new('RGB', (w, h), color='green')
  
    # Draw the rectangle.
    img1 = ImageDraw.Draw(img)  
    img1.rectangle(shape, fill ='#ffffff')

    # Save the image.
    img.save('image' + str(i) + '.png')

    left_side = right_side / character_length
    right_side += w

img1 = Image.open('image0.png').convert('L')
img2 = Image.open('image1.png').convert('L')
img3 = Image.new('L', (100, 40), 'white')

img3.paste(img1, (0, 0))
img3.paste(img2, (0, 20))

img3.save('combined_image.png')
