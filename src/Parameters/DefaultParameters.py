import sys
sys.path.append('./') if './' not in sys.path else None
sys.path.append('../') if '../' not in sys.path else None

from os import path

paths_params = {
    "input_png_x_dir"                 : path.join('data', 'parsed_dataset', 'fives', 'original'),
    "input_png_y_dir"                 : path.join('data', 'parsed_dataset', 'fives', 'ground'),
    "generate_dir"                    : path.join('data', 'generate'),
    "models_dir"                      : path.join('data', 'models'),    
    "unet_model_file"                 : 'unet_model_weights_vs.pt',
    "cnn_model_file"                  : 'cnn_model_weights_vs.pt',
    "checkup_img_dir"                 : path.join('data', 'checkup_images'),
    "segmentation_input_original_dir" : path.join('data', 'parsed_dataset', 'hrf', 'original'), 
    "segmentation_input_ground_dir"   : path.join('data', 'parsed_dataset', 'hrf', 'ground'), 
    "segmentation_training_dir"       : path.join('data', 'segmented', 'hrf', 'training'),       
    "segmentation_output_dir"         : path.join('data', 'segmented', 'hrf'),
    "segmentation_summary_file"       : path.join('data', 'segmented', 'hrf', 'segmentation_summary.txt'),
    "segmentation_dbl_file"           : path.join('data', 'parsed_dataset', 'list.dbl')     
}

hyper_params = {
    "max_generation_images_count"      : -1,     # if value is -1 then take all generation images
    "generation_images_per_npz"        : 4,      # number of images to generate per npz file
    "max_training_images_count"        : -1,     # if value is -1 then take all training images
    "max_testing_images_count"         : -1,     # if value is -1 then take all validation images
    "num_classes"                      : 2,
    "cnn_inference_batch_size"         : 256,
    "batch_size"                       : 256,
    "epochs"                           : 40,
    "l2_regularisation"                : 0.005,
    "dropout"                          : 0.55,
    "learning_rate"                    : 0.0001,
    "percentage_train"                 : 75,
    "windows_per_image_on_average"     : 32,
    "window_size"                      : 128,
    "min_fei_prct_window"              : 15.00,
    "max_non_fei_prct_window"          : 5.0,
    "fei_window_percentage"            : 50.00,
    "filters_count"                    : 16,
    "kernel_size"                      : 5,
    "augmentations_count"              : 3,
    "generated_images_to_save"         : 900,    # if -1, then no saving  
    "generated_windows_to_save"        : 900,     # if -1, then no saving
    "cnn_min_window_count"             : 25,
    "cnn_min_pixels_per_window_prct"   : 15.0
}

image_processing_params = {
    "width"   : 1024,
    "height"  : 1024,
    "norm"    : {"alpha": 7, "beta": 17, "norm_type": "cv2.NORM_L1", "order": -1},
    "denoise" : {"patch_size": 3, "patch_distance": 5, "strength": 7, "order": -1},
    "clahe"   : {"clip_limit": 5, "tile_grid_size": 4, "order": -1},    
    "median"  : {"ksize": 3, "order": -1}
}

segmentation_params = {
    "consider_single_region"             : False,
    "consider_oval_region"               : False,
    "eccentricity_level"                 : 0.8,
    "threshold_level"                    : 127,
    "segment_min_percent_scale"          : 20,
    "segment_max_percent_scale"          : 100,
    "segment_rough_percent_step_1"       : 10,
    "segment_rough_percent_step_2"       : 2,
    "segment_rough_interval"             : 5,
    "segment_rough_sliding_step_1"       : 64,
    "segment_rough_sliding_step_2"       : 32,
    "segment_detailed_sliding_step"      : 16,
    "segment_voting_percentage"          : 100,
    "segment_jaccard_windows_percentage" : 10,
    "segment_voting_percentage_mult"     : 3,
    "segment_voting_overlap_percentage"  : 50,
    "segment_small_image_scale_factor"   : 2.0,
    "padding_to_remove"                  : 0,
    "segmentation_suffix"                : "_vs",
    "segment_multiple_params"            : "900_20_100_20_10_4_128_64_16,900_20_100_20_10_4_64_64_16",
    "segment_simple_multiple_params"     : "128_64_32_16"
}