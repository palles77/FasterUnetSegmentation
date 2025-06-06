import numpy as np

def three_channels_to_one_hot_encoding(input_window):
    """
    Convert a three-channel input image to a one-hot encoded output image.
    
    Args:
        input_window (numpy array): A numpy array of shape (height, width, 3)
                                    representing a 3-channel input image.
    
    Returns:
        output_window (numpy array): A numpy array of shape (height, width, 2)
                                     representing a one-hot encoded output image.
    """
    shape = input_window.shape
    output_window = np.zeros((shape[0], shape[1], 2), dtype=np.uint8)
    
    white_pixels = (input_window == (255, 255, 255)).all(axis=-1)
    black_pixels = (input_window == (0, 0, 0)).all(axis=-1)

    output_window[white_pixels, 1] = 1
    output_window[black_pixels, 0] = 1

    return output_window

def one_hot_encoding_to_three_channels(input_window):
    """
    Convert a one-hot encoded input image to a three-channel output image.
    
    Args:
        input_window (numpy array): A numpy array of shape (height, width, 2)
                                    representing a one-hot encoded input image.
    
    Returns:
        output_window (numpy array): A numpy array of shape (height, width, 3)
                                     representing a three-channel output image.
    """
    shape = input_window.shape
    output_window = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    
    black_pixels = (input_window[..., 0] == 1)
    white_pixels = (input_window[..., 1] == 1)

    output_window[black_pixels] = (0, 0, 0)
    output_window[white_pixels] = (255, 255, 255)

    return output_window