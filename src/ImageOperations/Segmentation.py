import sys
sys.path.append('./') if './' not in sys.path else None
sys.path.append('../') if '../' not in sys.path else None

import string
import time
import glob as glob_module
import shutil
from Learning.Cnn import Cnn
from ImageOperations.ContrastDefault import process_image_with_default_contrast
from Learning.PretrainingDataGenerator import *
from Learning.Unet import *
from Parameters.DefaultParameters import *
from Common.FileTools import *
from ImageOperations.SingleRegionFinder import get_brightest_region, save_brightest_region
from ImageOperations.MaskFinder import binarize_image
import os

#===============================================================#
#                                                               #
#===============================================================# 
def pre_segment_image_into_region(
                    unet,
                    input_full_file_name,
                    cml_args):
    
    file_name_suffix = cml_args.segmentation_suffix
    input_short_file_name = os.path.basename(input_full_file_name)
    log_entry = f"[{input_short_file_name}][{file_name_suffix}][Progress][Starting segmentation]"
    print(log_entry)
    rough_sliding_step_1 = cml_args.segment_rough_sliding_step_1
    
    # Let's create a temporary directory with a current date and timestamp
    rand_str = ''.join(random.choice(string.ascii_lowercase) for i in range(10))
    temp_dir_name = os.path.join(cml_args.segmentation_output_dir, f'temp_{rand_str}')
    empty_or_create_directory(temp_dir_name)
    
    # short name of the input 
    input_short_file_name = os.path.basename(input_full_file_name)
    
    # Inside that directory create a scaled versions of the input image from scale of 10% up to 75% increasing by 2%
    img = cv2.imread(input_full_file_name)
    width = int(img.shape[1])
    height = int(img.shape[0])
   
    #--------------------------------------------------------#
    # rough segmentation                                     #
    #--------------------------------------------------------#     
    percentage_step = cml_args.segment_rough_percent_step_1
    min_percent = cml_args.segment_min_percent_scale
    max_percent = cml_args.segment_max_percent_scale
    
    if (width / unet.window_size) < 6 or (height / unet.window_size) < 6:
        # If the image is too small, we need to increase the max_percent twice
        max_percent = (int)(max_percent * 1.5)
        rough_sliding_step_1 = (int)(rough_sliding_step_1 / 2)
        
    perform_rough_segmentation(
        unet, cml_args, temp_dir_name, input_full_file_name, 
        rough_sliding_step_1, min_percent, max_percent, percentage_step)
    
    # We will take the segmented images and move them to the training directory
    # so we can calculate Jaccard distance between segmented binary files and target segmentation
    # to train the CNN model to understand which segmentation is better than others.
    # by associating scoring from 0 to 1 for each segmented image.
    segmented_pattern = os.path.join(temp_dir_name, 'output_*_segmented_binary*.png')
    segmented_binary_files = glob_module.glob(segmented_pattern)
    for segmented_binary_file_name in segmented_binary_files:
        input_short_file_name = os.path.basename(input_full_file_name)
        input_short_file_name_no_ext = os.path.splitext(input_short_file_name)[0]
        segmented_binary_file_name = os.path.basename(segmented_binary_file_name)
        percentage = int(segmented_binary_file_name.split('_')[1])
        destination_segmented_binary_file = input_short_file_name.replace(input_short_file_name_no_ext, f'{input_short_file_name_no_ext}_{percentage}') 
        destination_segmented_binary_file_full_path = os.path.join(cml_args.segmentation_training_dir, destination_segmented_binary_file)
        segmented_binary_file_name_full_path = os.path.join(temp_dir_name, segmented_binary_file_name)
        shutil.copy(segmented_binary_file_name_full_path, destination_segmented_binary_file_full_path)
        ground_truth_file = os.path.join(cml_args.segmentation_input_ground_dir, input_short_file_name)
        ground_truth_file_base_name = os.path.basename(ground_truth_file)
        destination_ground_truth_file = ground_truth_file_base_name.replace(".png", "_truth.png")
        destination_ground_truth_file_full_path = os.path.join(cml_args.segmentation_training_dir, destination_ground_truth_file)
        shutil.copy(ground_truth_file, destination_ground_truth_file_full_path)
        
#===============================================================#
#                                                               #
#===============================================================# 
def segment_image_into_region(
                    unet,
                    input_full_file_name,
                    cml_args):
    
    file_name_suffix = cml_args.segmentation_suffix
    input_short_file_name = os.path.basename(input_full_file_name)
    log_entry = f"[{input_short_file_name}][{file_name_suffix}][Progress][Starting segmentation]"
    print(log_entry)
    consider_single_region = cml_args.consider_single_region
    consider_oval_region = cml_args.consider_oval_region
    eccentricity_level = cml_args.eccentricity_level
    rough_sliding_step_1 = cml_args.segment_rough_sliding_step_1
    rough_sliding_step_2 = cml_args.segment_rough_sliding_step_2
    
    # Let's create a temporary directory with a current date and timestamp
    rand_str = ''.join(random.choice(string.ascii_lowercase) for i in range(10))
    temp_dir_name = os.path.join(cml_args.segmentation_output_dir, f'temp_{rand_str}')
    empty_or_create_directory(temp_dir_name)
    
    # short name of the input 
    input_short_file_name = os.path.basename(input_full_file_name)
    
    # Inside that directory create a scaled versions of the input image from scale of 10% up to 75% increasing by 2%
    img = cv2.imread(input_full_file_name)
    width = int(img.shape[1])
    height = int(img.shape[0])
   
    #--------------------------------------------------------#
    # rough segmentation                                     #
    #--------------------------------------------------------#     
    (segmented_grey_image_file_name, segmented_binary_image_file_name, _) = \
        get_segmented_and_grey_image_file_name(cml_args.segmentation_output_dir, input_short_file_name, file_name_suffix, rough_sliding_step_1)   

    percentage_step = cml_args.segment_rough_percent_step_1
    min_percent = cml_args.segment_min_percent_scale
    max_percent = cml_args.segment_max_percent_scale
    optional_cnn_model = None
    if os.path.exists(cml_args.cnn_model_file):
        optional_cnn_model = Cnn(cml_args.segmentation_training_dir, cml_args.models_dir, cml_args.cnn_model_file, 
            cml_args.epochs, cml_args.batch_size, cml_args.learning_rate, cml_args.window_size,
            cml_args.cnn_min_window_count, cml_args.cnn_min_pixels_per_window_prct)
        
        model_state_dict = torch.load(cml_args.cnn_model_file, weights_only=True)
        optional_cnn_model.model.load_state_dict(model_state_dict)
        optional_cnn_model.model.eval()
    
    if (width / unet.window_size) < 6 or (height / unet.window_size) < 6:
        # If the image is too small, we need to increase the max_percent twice
        min_percent = (int)(min_percent * cml_args.segment_small_image_scale_factor)
        max_percent = (int)(max_percent * cml_args.segment_small_image_scale_factor)
        percentage_step =(int)(percentage_step * cml_args.segment_small_image_scale_factor)
        
    rough_percentage_stage_1 = perform_rough_segmentation(unet, cml_args, temp_dir_name, input_full_file_name,                                                           
                                                          rough_sliding_step_1, min_percent, max_percent, percentage_step, optional_cnn_model)

    #--------------------------------------------------------#
    # detailed segmentation                                  #
    #--------------------------------------------------------#
    percentage_step = cml_args.segment_rough_percent_step_2
    min_percent = rough_percentage_stage_1 - cml_args.segment_rough_interval
    max_percent = rough_percentage_stage_1 + cml_args.segment_rough_interval
    
    rough_percentage_stage_2 = perform_rough_segmentation(unet, cml_args, temp_dir_name, input_full_file_name, 
                                                          rough_sliding_step_2, min_percent, max_percent, percentage_step, optional_cnn_model)
    
    #--------------------------------------------------------#
    # final segmentation with very small step                #
    #--------------------------------------------------------#
    
    # If we have a rough percentage, we can do detailed segmentation.    
    log_entry = f"[{input_short_file_name}][{file_name_suffix}][Progress][Starting final segmentation]"
    print(log_entry)
        
    (object_statistic, regions_count, detailed_grey_file_name, detailed_binary_file_name) = segment_image_by_percentage(
        unet, cml_args, temp_dir_name, input_full_file_name, 
        cml_args.segment_detailed_sliding_step, rough_percentage_stage_2, optional_cnn_model)
    
    log_entry = f"[{input_short_file_name}][{file_name_suffix}][Progress][Finished final segmentation with statistic: {object_statistic}, with regions count: {regions_count}]"
    print(log_entry)

    scaled_down_grey_image = cv2.imread(detailed_grey_file_name)
    dim = (width, height)
    resized = cv2.resize(scaled_down_grey_image, dim, interpolation = cv2.INTER_LANCZOS4)
    # check if file exists, and delete it if it does
    if os.path.exists(segmented_grey_image_file_name):
        os.remove(segmented_grey_image_file_name)
    cv2.imwrite(segmented_grey_image_file_name, resized)

    scaled_down_binary_image = cv2.imread(detailed_binary_file_name)
    resized = cv2.resize(scaled_down_binary_image, dim, interpolation = cv2.INTER_LANCZOS4)
    if os.path.exists(segmented_binary_image_file_name):
        os.remove(segmented_binary_image_file_name)
    cv2.imwrite(segmented_binary_image_file_name, resized) 
    
    if consider_single_region:
        # We need to narrow down the results for the user
        brightest_region, _ = get_brightest_region(
            segmented_grey_image_file_name, segmented_binary_image_file_name, 
            eccentricity_level, consider_oval_region)
        
        if brightest_region is not None:
            
            log_entry = f"[{input_short_file_name}][{file_name_suffix}][Progress][Final segmentation image has been masked to the brightest region with bounding box: {brightest_region.bbox}]"
            print(log_entry)
            
            brightest_region_segmented_grey_image_file_name = save_brightest_region(brightest_region, segmented_grey_image_file_name)
            shutil.move(brightest_region_segmented_grey_image_file_name, segmented_grey_image_file_name)
            brightest_region_segmented_binary_image_file_name = save_brightest_region(brightest_region, segmented_binary_image_file_name)
            shutil.move(brightest_region_segmented_binary_image_file_name, segmented_binary_image_file_name)
        
    return (segmented_grey_image_file_name, segmented_binary_image_file_name)        

#===============================================================#
#                                                               #
#===============================================================# 
def segment_image_by_voting_into_region(
                    unet,
                    input_full_file_name,
                    cml_args):
    
    file_name_suffix = cml_args.segmentation_suffix
    input_short_file_name = os.path.basename(input_full_file_name)
    log_entry = f"[{input_short_file_name}][{file_name_suffix}][Progress][Starting segmentation]"
    print(log_entry)
    
    # Let's create a temporary directory with a current date and timestamp
    rand_str = ''.join(random.choice(string.ascii_lowercase) for i in range(10))
    temp_dir_name = os.path.join(cml_args.segmentation_output_dir, f'temp_{rand_str}')
    empty_or_create_directory(temp_dir_name)
    
    # short name of the input 
    input_short_file_name = os.path.basename(input_full_file_name)
    
    # Inside that directory create a scaled versions of the input image from scale of 10% up to 75% increasing by 2%
    img = cv2.imread(input_full_file_name)
    width = int(img.shape[1])
    height = int(img.shape[0])
   
    #--------------------------------------------------------#
    # rough segmentation                                     #
    #--------------------------------------------------------#     
    (segmented_grey_image_file_name, segmented_binary_image_file_name, _) = \
        get_by_voting_segmented_and_grey_image_file_name(cml_args.segmentation_output_dir, input_short_file_name, file_name_suffix, cml_args.segment_voting_percentage)

    percentage_step = cml_args.segment_rough_percent_step_1
    min_percent = cml_args.segment_min_percent_scale
    max_percent = cml_args.segment_max_percent_scale
    window_cnn_model = None
    if os.path.exists(cml_args.cnn_model_file):
        window_cnn_model = Cnn(cml_args.segmentation_training_dir, cml_args.models_dir, cml_args.cnn_model_file, 
            cml_args.epochs, cml_args.batch_size, cml_args.learning_rate, cml_args.window_size,
            cml_args.cnn_min_window_count, cml_args.cnn_min_pixels_per_window_prct)
        model_state_dict = torch.load(cml_args.cnn_model_file, weights_only=True)
        window_cnn_model.model.load_state_dict(model_state_dict)
        window_cnn_model.model.eval()
    
    if (width / unet.window_size) < 6 or (height / unet.window_size) < 6:
        # If the image is too small, we need to increase the max_percent twice
        min_percent = (int)(min_percent * cml_args.segment_small_image_scale_factor)
        max_percent = (int)(max_percent * cml_args.segment_small_image_scale_factor)
        percentage_step = (int)(percentage_step * cml_args.segment_small_image_scale_factor)
        
    (rough_percentage_stage_1, _, _) = perform_rough_segmentation_by_voting(unet, cml_args, temp_dir_name, input_full_file_name,                                                           
        min_percent, max_percent, percentage_step, window_cnn_model)

    #--------------------------------------------------------#
    # detailed segmentation                                  #
    #--------------------------------------------------------#
    percentage_step = cml_args.segment_rough_percent_step_2
    min_percent = rough_percentage_stage_1 - cml_args.segment_rough_interval
    max_percent = rough_percentage_stage_1 + cml_args.segment_rough_interval

    (rough_percentage_stage_2, detailed_grey_file_name, detailed_binary_file_name) = perform_rough_segmentation_by_voting(unet, cml_args, temp_dir_name, input_full_file_name,
        min_percent, max_percent, percentage_step, window_cnn_model)    
   
    log_entry = f"[{input_short_file_name}][{file_name_suffix}][Progress][Finished final segmentation with statistic: {rough_percentage_stage_2}]"
    print(log_entry)

    scaled_down_grey_image = cv2.imread(detailed_grey_file_name)
    dim = (width, height)
    resized = cv2.resize(scaled_down_grey_image, dim, interpolation = cv2.INTER_LANCZOS4)
    # check if file exists, and delete it if it does
    if os.path.exists(segmented_grey_image_file_name):
        os.remove(segmented_grey_image_file_name)
    cv2.imwrite(segmented_grey_image_file_name, resized)

    scaled_down_binary_image = cv2.imread(detailed_binary_file_name)
    resized = cv2.resize(scaled_down_binary_image, dim, interpolation = cv2.INTER_LANCZOS4)
    if os.path.exists(segmented_binary_image_file_name):
        os.remove(segmented_binary_image_file_name)
    cv2.imwrite(segmented_binary_image_file_name, resized)
        
    return (segmented_grey_image_file_name, segmented_binary_image_file_name)

#===============================================================#
#                                                               #
#===============================================================# 
def segment_simple_image_into_region_no_voting(
                    unet,
                    input_full_file_name,
                    cml_args):
    
    file_name_suffix = cml_args.segmentation_suffix
    input_short_file_name = os.path.basename(input_full_file_name)
    log_entry = f"[{input_short_file_name}][{file_name_suffix}][Progress][Starting segmentation]"
    print(log_entry)
    
    # Let's create a temporary directory with a current date and timestamp
    rand_str = ''.join(random.choice(string.ascii_lowercase) for i in range(10))
    temp_dir_name = os.path.join(cml_args.segmentation_output_dir, f'temp_{rand_str}')
    empty_or_create_directory(temp_dir_name)
    
    # short name of the input 
    input_short_file_name = os.path.basename(input_full_file_name)
   
    #--------------------------------------------------------#
    # final segmentation with very small step                #
    #--------------------------------------------------------#
    
    # If we have a rough percentage, we can do detailed segmentation.    
    log_entry = f"[{input_short_file_name}][{file_name_suffix}][Progress][Starting final segmentation]"
    print(log_entry)
        
    (detailed_grey_file_name, detailed_binary_file_name) = segment_image_by_not_voting_percentage(
        unet, cml_args, temp_dir_name, input_full_file_name, 100)
    
    log_entry = f"[{input_short_file_name}][{file_name_suffix}][Progress][Finished final segmentation]"
    print(log_entry)
        
    return (detailed_grey_file_name, detailed_binary_file_name)

#===============================================================#
#                                                               #
#===============================================================# 
def segment_image_by_percentage(
    unet,
    cml_args,
    temp_dir_name,
    input_full_file_name,
    sliding_window_step,
    scale_percent):
    
    print(f"Scaling {input_full_file_name} to {scale_percent} percent.")
    
    # Inside that directory create a scaled versions of the input image from scale of 10% up to 75% increasing by 2%
    img = cv2.imread(input_full_file_name)
    width = int(img.shape[1])
    height = int(img.shape[0])

    # Resize the image
    scaled_width = int(width * scale_percent / 100)
    scaled_height = int(height * scale_percent / 100)
    scaled_dim = (scaled_width, scaled_height)
    resized = cv2.resize(img, scaled_dim, interpolation = cv2.INTER_AREA)       

    # Create a unique output file name based on the scale percent
    scaled_down_image_name = f"{temp_dir_name}/output_{scale_percent}.png"
    cv2.imwrite(scaled_down_image_name, resized)
    
    # Now let's have a masked version of the image    
    scaled_down_mask_image_name = f"{temp_dir_name}/output_{scale_percent}_bw.png"
    binarize_image(scaled_down_image_name, scaled_down_mask_image_name)
    
    scaled_down_normalized_image_name = f"{temp_dir_name}/output_{scale_percent}n.png"
    process_image_with_default_contrast(cml_args, scaled_down_image_name, scaled_down_normalized_image_name)    
    os.remove(scaled_down_image_name)
    shutil.move(scaled_down_normalized_image_name, scaled_down_image_name)   
    
    print(f"Segmentation start {scaled_down_image_name}")
    start_time = time.time()
    
    (scaled_down_grey_segment_file_name, scaled_down_binary_segment_file_name) = \
        segment_image_no_contrast(unet, cml_args, scaled_down_image_name, scaled_down_mask_image_name, temp_dir_name, sliding_window_step)
        
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Segmentation end {scaled_down_image_name}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")    
            
    return (scaled_down_grey_segment_file_name, scaled_down_binary_segment_file_name)

#===============================================================#
#                                                               #
#===============================================================# 
def segment_image_by_not_voting_percentage(
    unet,
    cml_args,
    temp_dir_name,
    input_full_file_name,
    scale_percent):
    """
    Segments an input image using a UNet model and a voting-based threshold, with optional CNN-based quality assessment.

    This function performs the following steps:
        1. Loads and rescales the input image to the specified scale_percent.
        2. Applies binarization and contrast normalization to the scaled image.
        3. Segments the image using the provided UNet model.
        4. Returns file paths for the resulting images.

    Args:
        unet: The UNet model instance used for segmentation.
        cml_args: Command-line arguments or configuration object with segmentation parameters.
        temp_dir_name (str): Temporary directory for storing intermediate files.
        input_full_file_name (str): Path to the input image file.
        scale_percent (int): Percentage to scale the input image before segmentation.

    Returns:
        tuple: (scaled_down_grey_segment_file_name, scaled_down_binary_segment_file_name)
            - scaled_down_grey_segment_file_name (str): Path to the segmented grayscale image.
            - scaled_down_binary_segment_file_name (str): Path to the segmented binary image.
    """    
    
    print(f"Scaling {input_full_file_name} to {scale_percent} percent.")
    
    # Inside that directory create a scaled versions of the input image from scale of 10% up to 75% increasing by 2%
    img = cv2.imread(input_full_file_name)
    width = int(img.shape[1])
    height = int(img.shape[0])

    # Resize the image
    scaled_width = int(width * scale_percent / 100)
    scaled_height = int(height * scale_percent / 100)
    scaled_dim = (scaled_width, scaled_height)
    resized = cv2.resize(img, scaled_dim, interpolation = cv2.INTER_AREA)       

    # Create a unique output file name based on the scale percent
    scaled_down_image_name = f"{temp_dir_name}/output_{scale_percent}.png"
    cv2.imwrite(scaled_down_image_name, resized)
    
    # Now let's have a masked version of the image    
    scaled_down_mask_image_name = f"{temp_dir_name}/output_{scale_percent}_bw.png"
    binarize_image(scaled_down_image_name, scaled_down_mask_image_name)
    
    scaled_down_normalized_image_name = f"{temp_dir_name}/output_{scale_percent}n.png"
    process_image_with_default_contrast(cml_args, scaled_down_image_name, scaled_down_normalized_image_name)    
    os.remove(scaled_down_image_name)
    shutil.move(scaled_down_normalized_image_name, scaled_down_image_name)   
    
    print(f"Segmentation start {scaled_down_image_name}")
    start_time = time.time()
    
    (scaled_down_grey_segment_file_name, scaled_down_binary_segment_file_name) = \
        segment_image_by_not_voting_no_contrast(unet, cml_args, scaled_down_image_name, scaled_down_mask_image_name, temp_dir_name)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Segmentation end {scaled_down_image_name}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    
    return (scaled_down_grey_segment_file_name, scaled_down_binary_segment_file_name)

#===============================================================#
#                                                               #
#===============================================================# 
def segment_image_by_voting_percentage(
    unet,
    cml_args,
    temp_dir_name,
    input_full_file_name,
    scale_percent,
    window_cnn_model):
    """
    Segments an input image using a UNet model and a voting-based threshold, with optional CNN-based quality assessment.

    This function performs the following steps:
        1. Loads and rescales the input image to the specified scale_percent.
        2. Applies binarization and contrast normalization to the scaled image.
        3. Segments the image using the provided UNet model.
        4. Optionally evaluates the segmentation quality using a CNN model, or by calculating the ratio of white pixels.
        5. Returns segmentation statistics and file paths for the resulting images.

    Args:
        unet: The UNet model instance used for segmentation.
        cml_args: Command-line arguments or configuration object with segmentation parameters.
        temp_dir_name (str): Temporary directory for storing intermediate files.
        input_full_file_name (str): Path to the input image file.
        scale_percent (int): Percentage to scale the input image before segmentation.
        voting_percentage (float): Voting threshold for post-processing (if applicable).
        window_cnn_model (Cnn, optional): Optional CNN model for assessing segmentation quality.

    Returns:
        tuple: (object_ratio, regions_count, scaled_down_grey_segment_file_name, scaled_down_binary_segment_file_name)
            - object_ratio (float): Ratio of segmented (white) pixels or CNN-assessed score.
            - regions_count (int): Number of detected regions in the segmentation.
            - scaled_down_grey_segment_file_name (str): Path to the segmented grayscale image.
            - scaled_down_binary_segment_file_name (str): Path to the segmented binary image.
    """    
    
    print(f"Scaling {input_full_file_name} to {scale_percent} percent.")
    
    # Inside that directory create a scaled versions of the input image from scale of 10% up to 75% increasing by 2%
    img = cv2.imread(input_full_file_name)
    width = int(img.shape[1])
    height = int(img.shape[0])

    # Resize the image
    scaled_width = int(width * scale_percent / 100)
    scaled_height = int(height * scale_percent / 100)
    scaled_dim = (scaled_width, scaled_height)
    resized = cv2.resize(img, scaled_dim, interpolation = cv2.INTER_AREA)       

    # Create a unique output file name based on the scale percent
    scaled_down_image_name = f"{temp_dir_name}/output_{scale_percent}.png"
    cv2.imwrite(scaled_down_image_name, resized)
    
    # Now let's have a masked version of the image    
    scaled_down_mask_image_name = f"{temp_dir_name}/output_{scale_percent}_bw.png"
    binarize_image(scaled_down_image_name, scaled_down_mask_image_name)
    
    scaled_down_normalized_image_name = f"{temp_dir_name}/output_{scale_percent}n.png"
    process_image_with_default_contrast(cml_args, scaled_down_image_name, scaled_down_normalized_image_name)    
    os.remove(scaled_down_image_name)
    shutil.move(scaled_down_normalized_image_name, scaled_down_image_name)   
    
    print(f"Segmentation start {scaled_down_image_name}")
    start_time = time.time()
    
    (scaled_down_grey_segment_file_name, scaled_down_binary_segment_file_name, segmentation_score) = \
        segment_image_by_voting_no_contrast(unet, cml_args, scaled_down_image_name, scaled_down_mask_image_name, temp_dir_name, window_cnn_model)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Segmentation end {scaled_down_image_name}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    
    return (segmentation_score, scaled_down_grey_segment_file_name, scaled_down_binary_segment_file_name)

#===============================================================#
#                                                               #
#===============================================================# 
def get_segmented_and_grey_image_file_name(segmented_dir, image_file_name, file_name_suffix, step):

    short_image_file_name = os.path.basename(image_file_name)
    short_image_grey_file_name = short_image_file_name.replace(".png", f"_{file_name_suffix}_segmented_grey_{step}.png").replace("__", "_")
    short_image_binary_file_name = short_image_file_name.replace(".png", f"_{file_name_suffix}_segmented_binary_{step}.png").replace("__", "_")
    
    segmented_grey_file_name = os.path.join(segmented_dir, short_image_grey_file_name)
    segmented_binary_file_name = os.path.join(segmented_dir, short_image_binary_file_name)
    segmented_partially_name_prefix = os.path.join(segmented_dir, short_image_file_name.replace(".png", ""))

    return (segmented_grey_file_name, segmented_binary_file_name, segmented_partially_name_prefix)

#===============================================================#
#                                                               #
#===============================================================# 
def get_by_voting_segmented_and_grey_image_file_name(
    segmented_dir, 
    image_file_name, 
    file_name_suffix, percentage):
    """
    Generates standardized file names for segmented grayscale and binary images produced by voting-based segmentation.

    This utility function constructs output file paths for both the grayscale and binary segmentation results,
    appending a suffix and voting percentage to the original image file name. It ensures consistent naming
    for downstream processing and result tracking.

    Args:
        segmented_dir (str): Directory where the segmented images will be saved.
        image_file_name (str): Original input image file name (can be a full or relative path).
        file_name_suffix (str): Suffix to append to the file name (e.g., indicating segmentation stage or method).
        percentage (int or float): Voting percentage used for segmentation, included in the file name.

    Returns:
        tuple:
            - segmented_grey_file_name (str): Full path to the output grayscale segmented image.
            - segmented_binary_file_name (str): Full path to the output binary segmented image.
            - segmented_partially_name_prefix (str): Prefix path for any additional related files.
    """

    short_image_file_name = os.path.basename(image_file_name)
    short_image_grey_file_name = short_image_file_name.replace(".png", f"_{file_name_suffix}_segmented_grey_vp_{percentage}.png").replace("__", "_")
    short_image_binary_file_name = short_image_file_name.replace(".png", f"_{file_name_suffix}_segmented_binary_vp_{percentage}.png").replace("__", "_")
    
    segmented_grey_file_name = os.path.join(segmented_dir, short_image_grey_file_name)
    segmented_binary_file_name = os.path.join(segmented_dir, short_image_binary_file_name)
    segmented_partially_name_prefix = os.path.join(segmented_dir, short_image_file_name.replace(".png", ""))

    return (segmented_grey_file_name, segmented_binary_file_name, segmented_partially_name_prefix)

#===============================================================#
#                                                               #
#===============================================================# 
def get_by_not_voting_segmented_and_grey_image_file_name(
    segmented_dir, 
    image_file_name, 
    file_name_suffix):
    """
    Generates standardized file names for segmented grayscale and binary images produced by not voting-based segmentation.

    This utility function constructs output file paths for both the grayscale and binary segmentation results,
    appending a suffix and voting percentage to the original image file name. It ensures consistent naming
    for downstream processing and result tracking.

    Args:
        segmented_dir (str): Directory where the segmented images will be saved.
        image_file_name (str): Original input image file name (can be a full or relative path).
        file_name_suffix (str): Suffix to append to the file name (e.g., indicating segmentation stage or method).
        percentage (int or float): Voting percentage used for segmentation, included in the file name.

    Returns:
        tuple:
            - segmented_grey_file_name (str): Full path to the output grayscale segmented image.
            - segmented_binary_file_name (str): Full path to the output binary segmented image.
            - segmented_partially_name_prefix (str): Prefix path for any additional related files.
    """

    short_image_file_name = os.path.basename(image_file_name)
    short_image_grey_file_name = short_image_file_name.replace(".png", f"_{file_name_suffix}_segmented_grey_not_voting.png").replace("__", "_")
    short_image_binary_file_name = short_image_file_name.replace(".png", f"_{file_name_suffix}_segmented_binary_not_voting.png").replace("__", "_")
    
    segmented_grey_file_name = os.path.join(segmented_dir, short_image_grey_file_name)
    segmented_binary_file_name = os.path.join(segmented_dir, short_image_binary_file_name)
    segmented_partially_name_prefix = os.path.join(segmented_dir, short_image_file_name.replace(".png", ""))

    return (segmented_grey_file_name, segmented_binary_file_name, segmented_partially_name_prefix)

#===============================================================#
#                                                               #
#===============================================================# 
def segment_image_no_contrast(unet,
                  cml_args,
                  input_full_file_name,
                  input_mask_file_name,
                  segment_dir,
                  sliding_window_step):
  
    # check if segment_dir exists  
    if not os.path.exists(segment_dir):
        os.makedirs(segment_dir)    
    
    short_image_name = os.path.basename(input_full_file_name)

    (segmented_grey_file_name, segmented_binary_file_name, _) = \
        get_segmented_and_grey_image_file_name(segment_dir, short_image_name, "", sliding_window_step)

    print('input_file_name = {}, output_grey_file_name = {}, output_binary_file_name = {}'.
        format(short_image_name, segmented_grey_file_name, segmented_binary_file_name))

    try:
        unet.segment(cml_args, input_full_file_name, input_mask_file_name, segmented_grey_file_name,
                     segmented_binary_file_name, sliding_window_step)
    except Exception as e:
        print("Exception while segmenting image: {}".format(e))
        (segmented_grey_file_name, segmented_binary_file_name) = (None, None)      
    
    return (segmented_grey_file_name, segmented_binary_file_name)

#===============================================================#
#                                                               #
#===============================================================# 
def segment_image_by_not_voting_no_contrast(unet,
                  cml_args,
                  input_full_file_name,
                  input_mask_file_name,
                  segment_dir):
  
    # check if segment_dir exists  
    if not os.path.exists(segment_dir):
        os.makedirs(segment_dir)    
    
    short_image_name = os.path.basename(input_full_file_name)

    (segmented_grey_file_name, segmented_binary_file_name, _) = \
        get_by_not_voting_segmented_and_grey_image_file_name(segment_dir, short_image_name, "")

    print('input_file_name = {}, output_grey_file_name = {}, output_binary_file_name = {}'.
        format(short_image_name, segmented_grey_file_name, segmented_binary_file_name))

    try:
        unet.segment_by_not_voting(cml_args, 
            input_full_file_name, input_mask_file_name, segmented_grey_file_name, segmented_binary_file_name)
    except Exception as e:
        print("Exception while segmenting image: {}".format(e))
        (segmented_grey_file_name, segmented_binary_file_name) = (None, None)
        exit(1)

    return (segmented_grey_file_name, segmented_binary_file_name)

#===============================================================#
#                                                               #
#===============================================================# 
def segment_image_by_voting_no_contrast(unet,
                  cml_args,
                  input_full_file_name,
                  input_mask_file_name,
                  segment_dir,
                  window_cnn_model):
  
    # check if segment_dir exists  
    if not os.path.exists(segment_dir):
        os.makedirs(segment_dir)    
    
    short_image_name = os.path.basename(input_full_file_name)

    (segmented_grey_file_name, segmented_binary_file_name, _) = \
        get_by_voting_segmented_and_grey_image_file_name(segment_dir, short_image_name, "", cml_args.segment_voting_percentage)

    print('input_file_name = {}, output_grey_file_name = {}, output_binary_file_name = {}'.
        format(short_image_name, segmented_grey_file_name, segmented_binary_file_name))

    segmentation_score = 0
    try:
        segmentation_score = unet.segment_by_voting(cml_args, input_full_file_name, input_mask_file_name, segmented_grey_file_name,
            segmented_binary_file_name, window_cnn_model)
    except Exception as e:
        print("Exception while segmenting image: {}".format(e))
        (segmented_grey_file_name, segmented_binary_file_name) = (None, None)
        exit(1)

    return (segmented_grey_file_name, segmented_binary_file_name, segmentation_score)

#--------------------------------------------------------#
#                                                        #
#--------------------------------------------------------#
def perform_rough_segmentation(
    unet, 
    cml_args,
    temp_dir_name, 
    input_full_file_name, 
    rough_sliding_step,
    min_percent, 
    max_percent, 
    percentage_step):
    """
    Performs rough segmentation on an input image using a sliding window approach with a large step size and a small step size.
    
    Args:
        unet (keras.models.Model): The trained UNet model to use for segmentation.
        cml_args (argparse.Namespace): The command line arguments to use for segmentation.
        temp_dir_name (str): The name of the temporary directory to use for storing intermediate files.
        input_full_file_name (str): The full path to the input image file.
        rough_sliding_step (int): The sliding step size to use for rough segmentation.
        min_percent (int): The minimum percentage of the image to use for segmentation.
        max_percent (int): The maximum percentage of the image to use for segmentation.
        percentage_step (int): The percentage step size to use for segmentation.
                
    Returns:
        int: The optimal rough segmentation percentage.
    """
    
    # Perform rough segmentation with large step size.
    input_short_file_name = os.path.basename(input_full_file_name)
    file_name_suffix = cml_args.segmentation_suffix
    log_entry = f"[{input_short_file_name}][{file_name_suffix}][Progress][Rough segmentation with {percentage_step} percent increase]"
    print(log_entry)
    
    for percentage in range(min_percent, max_percent + percentage_step, percentage_step):
        
        try:
            segment_image_by_percentage(
                unet, cml_args, temp_dir_name, input_full_file_name, rough_sliding_step, percentage)
                
        except Exception as error:
            log_entry = f"[{input_short_file_name}][{file_name_suffix}][Fail][Error: {str(error)}]"
            print(log_entry)

    
    log_entry = f"[{input_short_file_name}][{file_name_suffix}][Progress][Out of rough segmentation with step: {percentage_step}]"
    print(log_entry)

#--------------------------------------------------------#
#                                                        #
#--------------------------------------------------------#
def perform_rough_segmentation_by_voting(
    unet, 
    cml_args,
    temp_dir_name, 
    input_full_file_name, 
    min_percent, 
    max_percent, 
    percentage_step,
    optional_cnn_model = None):
    """
    Performs rough segmentation on an input image using a sliding window approach with a large step size and a small step size.
    
    Args:
        unet (keras.models.Model): The trained UNet model to use for segmentation.
        cml_args (argparse.Namespace): The command line arguments to use for segmentation.
        temp_dir_name (str): The name of the temporary directory to use for storing intermediate files.
        input_full_file_name (str): The full path to the input image file.
        min_percent (int): The minimum percentage of the image to use for segmentation.
        max_percent (int): The maximum percentage of the image to use for segmentation.
        percentage_step (int): The percentage step size to use for segmentation.
        optional_cnn_model (Cnn): The optional CNN model to use for segmentation quality assessement.
                
    Returns:
        int: The optimal rough segmentation percentage.
    """
    
    # Perform rough segmentation with large step size.
    input_short_file_name = os.path.basename(input_full_file_name)
    file_name_suffix = cml_args.segmentation_suffix
    log_entry = f"[{input_short_file_name}][{file_name_suffix}][Progress][Rough segmentation with {percentage_step} percent increase]"
    print(log_entry)
    # For single region segmentation we consider average brightness of the most fitting region.
    # For non single region segmentation we consider the area covered by the segmented object
    rough_object_statistics = []
    optimal_rough_object_statistic = 0
    optimal_rough_percentage = min_percent
    
    for percentage in range(min_percent, max_percent + 1, percentage_step):
        
        try:
            (segmentation_score, scaled_down_grey_segment_file_name, scaled_down_binary_segment_file_name) = segment_image_by_voting_percentage(
                unet, cml_args, temp_dir_name, input_full_file_name, percentage, optional_cnn_model)
  
            rough_object_statistics.append((percentage, segmentation_score, scaled_down_grey_segment_file_name, scaled_down_binary_segment_file_name))
                
        except Exception as error:
            log_entry = f"[{input_short_file_name}][{file_name_suffix}][Fail][Error: {str(error)}]"
            print(log_entry)
    
    output_scaled_down_grey_segment_file_name = None
    output_scaled_down_binary_segment_file_name = None        
    for (percentage, statistic, scaled_down_grey_segment_file_name, scaled_down_binary_segment_file_name) in rough_object_statistics:
        print(f"Percentage: {percentage}, statistic: {statistic}")        
        if statistic >= optimal_rough_object_statistic:
            # For single region use average brightness to determine optimal percentage
            optimal_rough_object_statistic = statistic
            optimal_rough_percentage = percentage   
            output_scaled_down_grey_segment_file_name = scaled_down_grey_segment_file_name
            output_scaled_down_binary_segment_file_name = scaled_down_binary_segment_file_name     
   
    log_entry = f"[{input_short_file_name}][{file_name_suffix}][Progress][Out of rough segmentation with step: {percentage_step}, chose percentage: {optimal_rough_percentage}, and optimal statistic: {optimal_rough_object_statistic}]"
    print(log_entry)

    return (optimal_rough_percentage, output_scaled_down_grey_segment_file_name, output_scaled_down_binary_segment_file_name)

#--------------------------------------------------------#
#                                                        #
#--------------------------------------------------------#
def perform_segmentation_by_not_voting(
    unet, 
    cml_args,
    temp_dir_name, 
    input_full_file_name):
    """
    Performs rough segmentation on an input image using a sliding window approach with step size equal to window size.
    
    Args:
        unet (keras.models.Model): The trained UNet model to use for segmentation.
        cml_args (argparse.Namespace): The command line arguments to use for segmentation.
        temp_dir_name (str): The name of the temporary directory to use for storing intermediate files.
        input_full_file_name (str): The full path to the input image file.
                
    Returns:
        int: The optimal rough segmentation percentage.
    """
    
    # Perform rough segmentation with large step size.
    input_short_file_name = os.path.basename(input_full_file_name)
    file_name_suffix = cml_args.segmentation_suffix
    log_entry = f"[{input_short_file_name}][{file_name_suffix}][Progress][Rough segmentation with no percent increase]"
    print(log_entry)
        
    try:
        (scaled_down_grey_segment_file_name, scaled_down_binary_segment_file_name) = segment_image_by_not_voting_percentage(
            unet, cml_args, temp_dir_name, input_full_file_name, 100.0)
            
    except Exception as error:
        log_entry = f"[{input_short_file_name}][{file_name_suffix}][Fail][Error: {str(error)}]"
        print(log_entry)
    
    output_scaled_down_grey_segment_file_name = scaled_down_grey_segment_file_name
    output_scaled_down_binary_segment_file_name = scaled_down_binary_segment_file_name        

    return (output_scaled_down_grey_segment_file_name, output_scaled_down_binary_segment_file_name)
