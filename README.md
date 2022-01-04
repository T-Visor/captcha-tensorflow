# DeCRUEHD  Framework

**D**eep **C**APTCHA **R**ecognition **U**sing **E**ncapsulated Preprocessing and **H**eterogeneous **D**atasets

A research effort for using Deep Learning (DL) techniques to recognize text-based CAPTCHAs. 

### Research Contributions:
1. The capability to generate 'Heterogeneous' CAPTCHA image samples, whereby different CAPTCHA schemes are employed to create a diversified labelled dataset.
2. Integrating the CRABI algorithm (**C**APTCHA **R**ecognition with **A**ttached **B**inary **I**mages) to preprocess CAPTCHA samples by attaching black and white bars as markers to the bottom of CAPTCHA image copies. This allows for CAPTCHA-text recognition on a per-character basis without the use of segmentation. 
3. Demonstrating the effectiveness of this CAPTCHA-recognition pipeline through transfer (continuous) learning. This project uses Convolutional Neural Networks (CNNs) to recognize characters in CAPTCHA images.

---
## Requirements
- Python 3
- Python 'pip' package manager
- Jupyter Notebook
- Python modules found in **requirements.txt** file

## Generate Dataset
- Run the **create_captcha_images.py** file with command-line arguments
- The script will generate all numeric combinations of a fixed-digit CAPTCHA length for each iteration 
- Example: 10,000 images for 4-digit CAPTCHAs, 1,000 images for 3-digit CAPTCHAs
- Users specify the CAPTCHA library to use for image generation at run-time
- All CAPTCHA images will be saved to a sub-directory within 'datasets' inside the project

## Train the Model
- Run the **create_captcha_recognition_model.py** file with command-line arguments
- The script will train a CNN to recognize CAPTCHA images
- Note: make sure the CAPTCHA length argument and data directory name argument are consistent with arguments used to generate the training dataset
- Users can specify the batch size, model architecture, and number of training epochs at run-time
- When the script finishes, the training history will be saved in a serialized format (using 'pickle' library) along with the trained model. Both will be saved in the root of the project folder.

## Model inference
- Examples of Jupyter notebook files with model inference can be found in the **notebooks** subdirectory. The notebook files contain 'evaluation' in their file names.
