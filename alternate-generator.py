#!/usr/bin/env python3

import captcha
import uuid
from captcha.image import ImageCaptcha

#FONTS =['/usr/share/fonts/TTF/System San Francisco Display Regular.ttf',
#        '/usr/share/fonts/TTF/MesloLGS-NF-Bold-Italic.ttf',
#        '/usr/share/fonts/TTF/DejaVuSerif.ttf']

DESTINATION = 'datasets/complex/'




for number in range(0, 10000):
    # 4 significant digits for every number.
    captcha_text = str(number)
    captcha_text = captcha_text.zfill(4)

    image = ImageCaptcha(width = 120, height = 100) 

    data = image.generate(captcha_text)  
    image.write(captcha_text, DESTINATION + captcha_text + '_' +
                str(uuid.uuid4()) + '.png')
