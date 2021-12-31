# CAPTCHA Solving Using TensorFlow

Original Author: Jackon Yang (2017)

Further Modified By: Turhan Kimbrough (2021)

---
## Requirements
- Python 3
- Python 'pip' package manager
- Jupyter Notebook
- Run the command 'pip install -r requirements.txt' in the project root directory to install the necessary Python modules

## Generate Dataset
- Run the 'captcha-generator.py' file and supply it with command-line arguments
- The script will generate all numeric combinations of a fixed-digit CAPTCHA length for each iteration 
- Example: 10,000 images for 4-digit CAPTCHAs, 1,000 images for 3-digit CAPTCHAs
- Users can specify which CAPTCHA library for generation at run-time, allowing mixed datasets
- All CAPTCHA images will be saved to a sub-directory within 'datasets' inside the project

![alt text for screen
readers](https://github.com/T-Visor/captcha-tensorflow/blob/master/pictures/captcha-generator-screenshot.png "script screenshot")

## Train the Model
- Run the 'create-captcha-solving-model.py' file and supply it with command-line arguments
- The script will train a convolutional neural network using the CAPTCHA images generated from the prior script
- Note: make sure the CAPTCHA length argument and data directory name argument is consistent to the arguments used in 'captcha-generator.py'
- Users can specify the batch size, model architecture (out of a set of choices), and number of training epochs at run-time
- When the script finishes, the training history will be saved in a serialized format (using 'pickle' library) along with the trained model. Both will be saved in the root of the project folder.

![alt text for screen
readers](https://github.com/T-Visor/captcha-tensorflow/blob/master/pictures/create-captcha-solving-model-screenshot.png "script screenshot")

## Model inference
- No standardized template has been created for model inference (at least yet)
- Examples of Jupyter notebook files with model inference can be found in the 'notebooks' subdirectory. The notebook files contain 'evaluation' in their file names.
