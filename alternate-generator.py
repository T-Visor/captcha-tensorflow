#!/usr/bin/env python3

import captcha
from captcha.image import ImageCaptcha

FONTS =['/usr/share/fonts/TTF/System San Francisco Display Regular.ttf',
        '/usr/share/fonts/TTF/MesloLGS-NF-Bold-Italic.ttf',
        '/usr/share/fonts/TTF/DejaVuSerif.ttf']


# change this back to 10,000 later
for i in range(0, 10):
    captcha_text = str(i)
    captcha_text = captcha_text.zfill(4)
    print(captcha_text)

    # Create an image instance of the given size
    image = ImageCaptcha(width = 80, height = 60, fonts=FONTS)

    # generate the image of the given text
    data = image.generate(captcha_text)  

    # write the image on the given file and save it
    image.write(captcha_text, captcha_text + '_image.png')
