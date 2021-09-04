#!/usr/bin/env python3

import captcha
from captcha.image import ImageCaptcha

FONTS =['/usr/share/fonts/TTF/System San Francisco Display Regular.ttf',
        '/usr/share/fonts/TTF/MesloLGS-NF-Bold-Italic.ttf',
        '/usr/share/fonts/TTF/DejaVuSerif.ttf']




for number in range(0, 10000):
    # 4 significant digits for every number.
    captcha_text = str(number)
    captcha_text = captcha_text.zfill(4)

    # Create and save the CAPTCHA image.
    image = ImageCaptcha(width = 80, height = 60, fonts=FONTS)
    data = image.generate(captcha_text)  
    image.write(captcha_text, 'datasets/training/' + captcha_text + '_alt.png')
