#!/usr/bin/env python3

import sys
import subprocess
from pathlib import Path

training_set_folder = 'heterogeneous'
training_set_exists = Path('datasets/' + training_set_folder).exists()

testing_set_folder = 'heterogeneous-test'
testing_set_exists = Path('datasets/' + testing_set_folder).exists()

# Create the heterogeneous training and testing datasets
if not training_set_exists:
    subprocess.call([sys.executable, 'create_captcha_images.py', '-i', '1', '-l', '4', '-t', 'SIMPLE', '-d', training_set_folder])
    subprocess.call([sys.executable, 'create_captcha_images.py', '-i', '2', '-l', '4', '-t', 'COMPLEX', '-d', training_set_folder])
    subprocess.call([sys.executable, 'create_captcha_images.py', '-i', '2', '-l', '4', '-t', 'MONOCHROME', '-d', training_set_folder])
if not testing_set_exists:
    subprocess.call([sys.executable, 'create_captcha_images.py', '-i', '1', '-l', '4', '-t', 'SIMPLE', '-d', testing_set_folder])
    subprocess.call([sys.executable, 'create_captcha_images.py', '-i', '1', '-l', '4', '-t', 'COMPLEX', '-d', testing_set_folder])
    subprocess.call([sys.executable, 'create_captcha_images.py', '-i', '1', '-l', '4', '-t', 'MONOCHROME', '-d', testing_set_folder])

# Train MobileNet model
subprocess.call([sys.executable, 
                 'create_captcha_recognition_model.py', 
                 '-d', training_set_folder, 
                 '-l', '4', 
                 '-e', '50', 
                 '-b', '32', 
                 '-a', 'MOBILE-NET',
                 '-m', 'mobilenet', 
                 '-t', 'mobilenet-history'])
