import os
import glob
import numpy
import pandas

def create_captcha_dataframe(captcha_images_directory):
    """
    Args:
        captcha_images_directory (str): the full file path to the folder where the captcha images
                                        were generated
    
    Returns:
        a pandas.DataFrame object storing each captcha file name along with its label
    """
    files = glob.glob(os.path.join(captcha_images_directory, '*.png'))
    attributes = list(map(_get_captcha_label, files))

    captcha_dataframe = pandas.DataFrame(attributes)
    captcha_dataframe['file'] = files
    captcha_dataframe.columns = ['label', 'file']
    captcha_dataframe = captcha_dataframe.dropna()
    
    return captcha_dataframe




def _get_captcha_label(file_path):
    """
    (HELPER FUNCTION)

    Precondition: CAPTCHA images were generated using the
                  script found in this project folder
    
    Args:
        file_path (str): the path to the CAPTCHA image
    
    Returns:
        the 'label' for each CAPTCHA denoted by the 
        string in the file name before the '_'
        character

        Example: '9876_image.png' -> '9876' 
    """
    try:
        path, file_name = os.path.split(file_path)
        file_name, extension = os.path.splitext(file_name)
        label, _ = file_name.split("_")
        return label
    except Exception as e:
        print('error while parsing %s. %s' % (file_path, e))
        return None, None




def shuffle_and_split_data(data_frame):
    """
        Shuffle and split the data into 2 sets: training and validation.
    
    Args:
        data_frame (pandas.DataFrame): the data to shuffle and split
    
    Returns:
        2 numpy.ndarray objects -> (train_indices, validation_indices)
        Each hold the index positions for data in the pandas.DataFrame 
    """
    shuffled_indices = numpy.random.permutation(len(data_frame))

    train_up_to = int(len(data_frame) * 0.7)

    train_indices = shuffled_indices[:train_up_to]
    validation_indices = shuffled_indices[train_up_to:]

    return train_indices, validation_indices

