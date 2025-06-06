import os
from PIL import Image
from Parameters.DefaultParameters import *
import numpy
import cv2
from ImageOperations.Contrast import Contrast

#===============================================================#
#                                                               #
#===============================================================#
def process_image_with_default_contrast(
    cml_args,    
    input_path, 
    output_path = None):
    
    if os.path.exists(input_path) == False:
        print(f"Input path does not exist: ", input_path)
        return
    
    # Load image
    try:
        with Image.open(input_path) as pil_image:
            input_image = cv2.cvtColor(numpy.array(pil_image), cv2.COLOR_BGR2RGB)
    except:
        print(f"Error loading image: ", input_path)
        return   
        
    # Normalization
    alpha = cml_args.norm_alpha
    beta = cml_args.norm_beta
    norm_type = cml_args.norm_type
    
    # Clahe
    clip_limit = cml_args.clahe_clip_limit
    tile_grid_size = cml_args.clahe_tile_grid_size
    
    # Denoising
    patch_size = cml_args.denoise_patch_size
    patch_distance = cml_args.denoise_patch_distance
    denoise_strength = cml_args.denoise_strength
    
    # Median Filtering
    kernel_size = cml_args.median_ksize
    
    if output_path is None:
        # Get the input path without extenstion
        output_path = input_path[:input_path.rfind(".")]
        norm_str = f"norm_a_{alpha}_b_{beta}_nt_{norm_type}"
        clahe_str = f"clahe_cl_{clip_limit}_tgs_{tile_grid_size}"
        denoise_str = f"denoise_ps_{patch_size}_pd_{patch_distance}_ds_{denoise_strength}"
        median_str = f"median_k_{kernel_size}"
        output_path = f"{output_path}_{norm_str}_{clahe_str}_{denoise_str}_{median_str}.png"
    
    contrast = Contrast()
    output_image = input_image
    
    for i in range(0, 4):
        if (cml_args.norm_order == i):
            output_image = contrast.apply_normalization(output_image, alpha / 100.0, beta - 100, norm_type)
        elif (cml_args.clahe_order == i):
            output_image = contrast.apply_clahe(output_image, clip_limit, tile_grid_size)
        elif (cml_args.denoise_order == i):
            output_image = contrast.apply_noise_filtering(output_image, denoise_strength / 100.0, patch_size, patch_distance)
        elif (cml_args.median_order == i):
            output_image = contrast.apply_median_filter(output_image, kernel_size)

    cv2.imwrite(output_path, output_image)
