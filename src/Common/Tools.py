import sys
sys.path.append('./') if './' not in sys.path else None
sys.path.append('../') if '../' not in sys.path else None

from ImageOperations.MaskFinder import binarize_image
from ImageOperations.ContrastDefault import process_image_with_default_contrast
from Learning.PretrainingDataGenerator import *
from Learning.Unet import *
from Parameters.DefaultParameters import *
from Common.FileTools import *
import scipy.misc
from random import randint
from PIL import Image
import numpy
import os
from os import path
import math
from PIL import Image

#------------------------------------------------------
# generate training data
#------------------------------------------------------
def generate_data(cml_args):    
    
    x_images_dir = cml_args.x_dir
    y_images_dir = cml_args.y_dir
    width = cml_args.width
    height = cml_args.height
    max_generation_images_count = cml_args.max_generation_images_count
      
    # create a subdirectory in X directory with processed images
    processed_x_images_dir = path.join(x_images_dir, "processed")
    processed_y_images_dir = path.join(y_images_dir, "processed")
    empty_or_create_directory(processed_x_images_dir)
    empty_or_create_directory(processed_y_images_dir)
    
    fei = PretrainingDataGenerator()   

    # First check if both directories contain identical file names
    files_x = [f for f in os.listdir(x_images_dir) if os.path.isfile(os.path.join(x_images_dir, f)) and f.endswith('.png')]
    files_y = [f for f in os.listdir(y_images_dir) if os.path.isfile(os.path.join(y_images_dir, f)) and f.endswith('.png')]
    
    files_x_filtered = [os.path.basename(file) for file in files_x]
    files_x_filtered_sorted = sorted(files_x_filtered)
    
    files_y_filtered = [os.path.basename(file) for file in files_y]
    files_y_filtered_sorted = sorted(files_y_filtered)
    
    # now make sure both arrays match
    if files_x_filtered_sorted != files_y_filtered_sorted:
        print(f"Error: {x_images_dir} x and {y_images_dir}' directories do not contain identical file names.")
        return
       
    # apply the same filter to X directory
    print(f"Processing {len(files_x)} images in {x_images_dir} and {y_images_dir} directories...")
    len_max = len(files_x)
    if max_generation_images_count != -1:
        len_max = min(max_generation_images_count, len(files_x))
    count = 0
    for file in files_x:
        if count >= len_max:
            break
        if count % 10 == 0:
            print(f"Processing image {file} which is {count} of {len_max}")
        copied_x_full_path = path.join(processed_x_images_dir, file)
        shutil.copy(path.join(x_images_dir, file), copied_x_full_path)
        processed_file_name = process_x_training_image(copied_x_full_path, height, width, cml_args)
        shutil.move(processed_file_name, copied_x_full_path)
        count += 1

    # apply scaling down to Y directory
    count = 0
    for file in files_y:
        if count >= len_max:
            break
        if count % 10 == 0:
            print(f"Processing image {file} which is {count} of {len_max}")
        copied_y_full_path = path.join(processed_y_images_dir, file)
        shutil.copy(path.join(y_images_dir, file), copied_y_full_path)
        processed_file_name = process_y_training_image(copied_y_full_path, height, width)
        shutil.move(processed_file_name, copied_y_full_path)
        count += 1
    
    fei.write_training_data(
        processed_x_images_dir, processed_y_images_dir,
        cml_args.generate_dir, max_generation_images_count, 
        height, width, cml_args.window_size,
        cml_args.windows_per_image_on_average, cml_args.augmentations_count,
        cml_args.min_fei_prct_window, cml_args.max_non_fei_prct_window, cml_args.fei_window_percentage,
        cml_args.checkup_img_dir, cml_args.generated_images_to_save, 
        cml_args.generated_windows_to_save, cml_args.generation_images_per_npz, cml_args.percentage_train)
    
    empty_or_create_directory(processed_x_images_dir)
    empty_or_create_directory(processed_y_images_dir)

#------------------------------------------------------
# save Fei color and B&W circled Fei images
# images_count - how many random images pairs
# data_x - 4D tensor with training images
# data_y - 4D tensor with B&W destination images
# generated_images_dir - directory where to save images
#------------------------------------------------------
def save_images(images_count, data_x, data_y, generated_images_dir):

    # a sample file name just to create directory if it does not exist
    empty_or_create_directory(path.join(generated_images_dir))

    for i in range(0, images_count):
        index = randint(0, data_x.shape[0] - 1)
        print("Index is {}".format(index))

        # outfile_x is an original colour random file
        image_x_to_save = path.join(generated_images_dir, 'outfile_{}_x.png'.format(index))
        data_x_to_save = data_x[index, :, :, :]
        image_x = Image.fromarray(numpy.uint8(data_x_to_save))
        image_x.save(image_x_to_save)
        
        # outfile_y is a B&W file with circles centred where Fei are for the random file above
        image_y_to_save = path.join(generated_images_dir, 'outfile_{}_y.png'.format(index))
        data_y_to_save = data_y[index, :, :, :]
        image_y = Image.fromarray(numpy.uint8(data_y_to_save))
        image_y.save(image_y_to_save)

#------------------------------------------------------
# save Fei/non Fei images
# images_count - how many random images pairs
# data_x - 4D tensor with training images
# data_y - 4D tensor with B&W destination images
#------------------------------------------------------
def save_images_overlay(images_count, data_x, data_y):

    for i in range(0, images_count):
        index = randint(0, data_x.shape[0] - 1)
        print("Index is {}".format(index))

        data_x[ : , :, :, 1] = data_y[ : , : , : , 1]
        scipy.misc.imsave('outfile_x_{}.jpg'.format(index), data_x[index] * 255.0)

#===============================================================#
#                                                               #
#===============================================================# 
def print_parameters(training_params, hyper_params, image_processing_params, segmentation_params):

    print("Training Parameters:")
    for key, value in training_params.items():
        print("Key: {}, Value: {}".format(key, value))
        
    print("Hyper Parameters:")
    for key, value in hyper_params.items():
        print("Key: {}, Value: {}".format(key, value))
        
    print("Image Processing Parameters:")
    for key, value in image_processing_params.items():
        print("Key: {}, Value: {}".format(key, value))
        
    print("Segmentation Parameters:")
    for key, value in segmentation_params.items():
        print("Key: {}, Value: {}".format(key, value))


#===============================================================#
#                                                               #
#===============================================================# 
def load_unet(
    cml_args):   

    unet = Unet(cml_args)

    return unet

#===============================================================#
#                                                               #
#===============================================================# 
def process_x_training_image(
    training_image_path, 
    dest_width,
    dest_height,
    cml_args):
    """
    Processes a training image by resizing it to the specified dimensions and applying
    additional image processing parameters.

    Args:
        training_image_path (str): The file path to the training image.
        dest_height (int): The desired height of the processed image.
        dest_width (int): The desired width of the processed image.
        image_processing_params (dict): A dictionary of additional image processing parameters.

    Returns:
        processed_image: The processed image.
    """
        
    # Rescale
    img = cv2.imread(training_image_path)
    scaled_dim = (dest_width, dest_height)
    resized = cv2.resize(img, scaled_dim, interpolation = cv2.INTER_AREA)       
    scaled_down_image_name = training_image_path.replace(".png", "_scaled.png")
    cv2.imwrite(scaled_down_image_name, resized)
    
    # Apply default contrast
    scaled_down_normalized_image_name = training_image_path.replace(".png", "_scaled_normalized.png")
    process_image_with_default_contrast(cml_args, scaled_down_image_name, scaled_down_normalized_image_name) 
    
    # Now binarize
    scaled_down_mask_image_name = training_image_path.replace(".png", "_scaled_mask.png")
    binarize_image(scaled_down_image_name, scaled_down_mask_image_name)
    
    # Now let's apply black and white mask
    mask = cv2.imread(scaled_down_mask_image_name, cv2.IMREAD_GRAYSCALE)
    
    # Read the scaled down normalized image
    image = cv2.imread(scaled_down_normalized_image_name)

    # Apply the mask
    masked_image = cv2.bitwise_and(image, image, mask=mask)

    # Save or display the masked image as needed
    scaled_down_normalized_masked_image_name = training_image_path.replace(".png", "_scaled_normalized_masked.png")
    cv2.imwrite(scaled_down_normalized_masked_image_name, masked_image)
    
    # print(f"Processed training image {training_image_path} to {scaled_down_normalized_masked_image_name}")
    
    # delete the intermediate files
    try:
        os.remove(scaled_down_image_name)
        os.remove(scaled_down_mask_image_name)
        os.remove(scaled_down_normalized_image_name)
    except OSError as e:
        print(f"Error: {e.strerror} - {e.filename}")
    
    return scaled_down_normalized_masked_image_name

#===============================================================#
#                                                               #
#===============================================================# 
def process_y_training_image(
    training_image_path, 
    dest_height, 
    dest_width):
    """
    Processes a training image by resizing it to the specified dimensions and applying
    additional image processing parameters.

    Args:
        training_image_path (str): The file path to the training image.
        dest_height (int): The desired height of the processed image.
        dest_width (int): The desired width of the processed image.

    Returns:
        processed_image: The processed image.
    """
        
    # Rescale
    img = cv2.imread(training_image_path)
    scaled_dim = (dest_width, dest_height)
    resized = cv2.resize(img, scaled_dim, interpolation = cv2.INTER_AREA)       
    scaled_down_image_name = training_image_path.replace(".png", "_scaled.png")
    cv2.imwrite(scaled_down_image_name, resized)
    
    return scaled_down_image_name