#/usr/bin/env python3

from PIL import Image, ImageDraw, ImageFont
import string, random 

# RGB values for color
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

# pick random colors for points
colors = ['black', 'red', 'blue', 'green', (64, 107, 76), (0, 87, 128), (0, 3, 82)]
#colors = ['black', 'red', 'blue', 'green', FOREST_GREEN, SEA_BLUE, DARK_BLUE]

# pick a random colors for lines
fill_color = [(64, 107, 76), (0, 87, 128), (0, 3, 82), (191, 0, 255), (72, 189, 0), (189, 107, 0), (189, 41, 0)]
#fill_color = [FOREST_GREEN, SEA_BLUE, DARK_INDIGO, PINK, LIGHT_GREEN, ORANGE, RED]




def main():
    i = 1
    while (i < 10):
        generate_captcha_image()
        i += 1




def generate_captcha_image():
    
    # create a colored 100x100 image with white background
    captcha_image = Image.new('RGB', (80, 60), color="white")
    illustrater = ImageDraw.Draw(captcha_image)

    # generate a randomly colored string
    captcha_string = generate_random_string(4, 'numeric')
    get_colored_text(illustrater, captcha_string)

    # draw some random lines
    for i in range(5,random.randrange(6, 10)):
        illustrater.line((getit(), getit()), fill=random.choice(fill_color), width=random.randrange(1,3))

    # draw some random points
    for i in range(10,random.randrange(11, 20)):
        illustrater.point((getit(), getit(), getit(), getit(), getit(), getit(), getit(), getit(), getit(), getit()), fill=random.choice(colors))

    # save the newly generate CAPTCHA image
    captcha_image.save('captcha_img/' + captcha_string + '_image.png')




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
    text_colors = random.choice(colors)
    font_name = 'DejaVuSansMono.ttf'
    font = ImageFont.truetype(font_name, 18)
    illustrater.text((20,20), captcha_string, fill=text_colors, font=font)




main()
