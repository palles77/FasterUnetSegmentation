import sys
sys.path.append('./') if './' not in sys.path else None
sys.path.append('../') if '../' not in sys.path else None

import os
import numpy as np
import cv2

#===============================================================#
#                                                               #
#===============================================================# 
def binarize_image(input_image_path, output_image_path):
    # Read the image as a NumPy array
    image = cv2.imread(input_image_path)
    
    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    threshold = np.mean(grayscale_image)/3-5
    _, mask = cv2.threshold(grayscale_image, max(0,threshold), 1, cv2.THRESH_BINARY)
    nn_mask = np.zeros((mask.shape[0]+2,mask.shape[1]+2),np.uint8)
    new_mask = (1-mask).astype(np.uint8)
    width = new_mask.shape[1]
    height = new_mask.shape[0]
    _, new_mask, _, _ = cv2.floodFill(new_mask, nn_mask, (0,0), (0), cv2.FLOODFILL_MASK_ONLY)
    _, new_mask, _, _ = cv2.floodFill(new_mask, nn_mask, (width - 1, height - 1), (0), cv2.FLOODFILL_MASK_ONLY)
    _, new_mask, _, _ = cv2.floodFill(new_mask, nn_mask, (width - 1, 0), (0), cv2.FLOODFILL_MASK_ONLY)
    _, new_mask, _, _ = cv2.floodFill(new_mask, nn_mask, (0, height - 1), (0), cv2.FLOODFILL_MASK_ONLY)

    mask = mask + new_mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (60,  60))
    mask = cv2.erode(mask, kernel)
    mask = cv2.dilate(mask, kernel)    
    
    # Invert the mask
    inverted_mask = cv2.bitwise_not(mask)
    
    # Create a blank image with the same size as the original image
    binarized_image = np.zeros_like(image)    
    
    # Set the pixels in the output image to black or white based on the mask
    binarized_image[inverted_mask == 0] = [0, 0, 0]
    binarized_image[inverted_mask == 255] = [255, 255, 255]
    
    # now revert the binarized_image to its negative
    binarized_image = cv2.bitwise_not(binarized_image)
    
    # Save the binarized image in PNG format
    cv2.imwrite(output_image_path, binarized_image)

#===============================================================#
#                                                               #
#===============================================================# 
def is_window_outside_mask(window_array):
    
    # create a mask where pixels with all three channels are black, all others are white    
    mask = np.where(np.all(window_array == 0, axis=-1), 0, 1).astype(np.uint8)
    sum = np.sum(mask)
    
    height = window_array.shape[0]
    width = window_array.shape[1]
    threshold = int(round(height * width * 0.25))
    result = sum < threshold 
    
    return result

#===============================================================#
#                                                               #
#===============================================================# 
def remove_outer_rim(input_image_name: str,
                     input_extension: str = ".png",
                     output_extension: str = ".png"):
    
    no_rim_input_image_name = None
    if not input_image_name.endswith(input_extension):
        return no_rim_input_image_name
    
    try:
    
        input_image_short_name = os.path.basename(input_image_name)
        input_image_name_without_ext = os.path.splitext(input_image_short_name)[0]
        input_image_directory = os.path.dirname(input_image_name) 
    
        binarized_input_image_name = os.path.join(input_image_directory, input_image_name_without_ext + "_no_rim_binarized" + output_extension)
        no_rim_input_image_name = os.path.join(input_image_directory, input_image_name_without_ext + "_no_rim" + output_extension)
        binarize_image(input_image_name, binarized_input_image_name)
        
        # Read the image as a NumPy array
        binarized_input_image = cv2.imread(binarized_input_image_name)
                
        # find min x of white pixel in binarized_input_image
        white_pixel_indices = np.where(binarized_input_image == 255)
        min_x = np.min(white_pixel_indices[1])
        max_x = np.max(white_pixel_indices[1])
        min_y = np.min(white_pixel_indices[0])
        max_y = np.max(white_pixel_indices[0])
        
        # now cut down the input_image_name to the rectange (min_x, min_y, max_x, max_y))
        input_image = cv2.imread(input_image_name)
        no_rim_image = input_image[min_y:max_y, min_x:max_x]
        
        # now create a mask where pixels with all grayscale_no_rim_image below threshold are black, all others are as they are
        no_rim_binarized_image = binarized_input_image[min_y:max_y, min_x:max_x]
        _, no_rim_mask = cv2.threshold(no_rim_binarized_image, 0, 255, cv2.THRESH_BINARY)        
        
        # now write the mask to the no_rim_image
        no_rim_image[np.all(no_rim_mask == [0, 0, 0], axis=-1)] = [0, 0, 0]
        cv2.imwrite(no_rim_input_image_name, no_rim_image)
        
        if os.path.exists(binarized_input_image_name):
            os.remove(binarized_input_image_name)    
    
    except Exception as e:
        print("Exception in remove_outer_rim: " + str(e))
        
    return no_rim_input_image_name

#===============================================================#
#                                                               #
#===============================================================# 
def get_outer_rim_rect(input_image_name: str, output_extension: str = ".png"):
    try:
        input_image_short_name = os.path.basename(input_image_name)
        input_image_name_without_ext = os.path.splitext(input_image_short_name)[0]
        input_image_directory = os.path.dirname(input_image_name) 
    
        binarized_input_image_name = os.path.join(input_image_directory, input_image_name_without_ext + "_no_rim_binarized" + output_extension)
        no_rim_input_image_name = os.path.join(input_image_directory, input_image_name_without_ext + "_no_rim" + output_extension)
        binarize_image(input_image_name, binarized_input_image_name)
        
        # Read the image as a NumPy array
        binarized_input_image = cv2.imread(binarized_input_image_name)
                
        # find min x of white pixel in binarized_input_image
        white_pixel_indices = np.where(binarized_input_image == 255)
        min_x = np.min(white_pixel_indices[1])
        max_x = np.max(white_pixel_indices[1])
        min_y = np.min(white_pixel_indices[0])
        max_y = np.max(white_pixel_indices[0])
        
        if os.path.exists(binarized_input_image_name):
            os.remove(binarized_input_image_name)
            
        if os.path.exists(no_rim_input_image_name):
            os.remove(no_rim_input_image_name)
        
        return min_x, min_y, max_x, max_y

    except Exception as e:
        print("Exception in get_outer_rim_rect: " + str(e))
        return None
