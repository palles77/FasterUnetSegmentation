import sys
sys.path.append('./') if './' not in sys.path else None
sys.path.append('../') if '../' not in sys.path else None

from Common.Tools import print_parameters
from Parameters.DefaultParameters import *
import argparse
def create_arguments():
    parser = argparse.ArgumentParser(description="Train and test UNET model")

    # Paths paramete    # These parameters are used to generate data for training
    parser.add_argument('-c', '--cleanup',
                        help='with this option program will cleanup directories',
                        dest='cleanup', default=False, action='store_true')
    
    parser.add_argument('-g', '--generate',
                        help='with this option program will only generate data for the 1st iteration of unsupervised training',
                        dest='generate_data', default=False, action='store_true')

    parser.add_argument('-xd', '--x-dir',
                        help="relative path to input directory with x data", type=str,
                        dest='x_dir', default=paths_params['input_png_x_dir'])

    parser.add_argument('-yd', '--y-dir',
                        help="relative path to input directory with y data", type=str,
                        dest='y_dir', default=paths_params['input_png_y_dir'])

    parser.add_argument('-tih', '--training-image-height',
                        help="height of training image", type=int,
                        dest='height', default=image_processing_params['height'])

    parser.add_argument('-tiw', '--training-image-width',
                        help="width of the training image", type=int,
                        dest='width', default=image_processing_params['width'])
    
    parser.add_argument('-gits', '--generated-images-to-save',
                        help="generated images to save count", type=int,
                        dest='generated_images_to_save', default=hyper_params['generated_images_to_save'])
    
    parser.add_argument('-gwts', '--generated-windows-to-save',
                        help="generated windows to save count", type=int,
                        dest='generated_windows_to_save', default=hyper_params['generated_windows_to_save'])

    # These are common parameters used to generate data and also to train UNet model
    parser.add_argument('-gd', '--generate-dir',
                    help="relative path to generate directory", type=str,
                    dest='generate_dir', default=paths_params['generate_dir'])

    # These parameters are used to train Unet model
    parser.add_argument('-t', '--train',
                        help='with this option the program will train Unet model and store it',
                        dest='train', action='store_true')
    
    parser.add_argument('-md', '--models-dir',
                        help="relative path to models directory", type=str,
                        dest='models_dir', default=paths_params['models_dir'])        

    parser.add_argument('-umf', '--unet-model-file',
                        help="relative path to output filename with unet model, relative to models_dir", type=str,
                        dest='unet_model_file', default=paths_params['unet_model_file'])
    
    parser.add_argument('-cmf', '--cnn-model-file',
                        help="relative path to output filename with cnn model, relative to models_dir", type=str,
                        dest='cnn_model_file', default=paths_params['cnn_model_file'])

    parser.add_argument('-ci', '--checkup-img-dir',
                        help="directory for checkup images", type=str,
                        dest='checkup_img_dir', default=paths_params['checkup_img_dir'])

    # Hyperparameters
    parser.add_argument("-gipn", "--generate-images-per-npz",
                        help="number of images to generate per npz file", type=int,
                        dest="generation_images_per_npz", default=hyper_params["generation_images_per_npz"])
    
    parser.add_argument("-mgic", "--maximum-generation-img-count",
                        help="maximum generation images count", type=int,
                        dest="max_generation_images_count", default=hyper_params["max_generation_images_count"])
    
    parser.add_argument("-mtrc", "--maximum-train-img-count",
                        help="maximum training images count", type=int,
                        dest="max_training_images_count", default=hyper_params["max_training_images_count"])

    parser.add_argument("-mtsc", "--maximum-test-img-count",
                        help="maximum testing images count", type=int,
                        dest="max_testing_images_count", default=hyper_params["max_testing_images_count"])     
    
    parser.add_argument("-nc", "--num-classes",
                        help="number of unet pixel classes", type=int,
                        dest="num_classes", default=hyper_params["num_classes"]) 
    
    parser.add_argument("-bs", "--batch-size",
                        help="batch size for training", type=int,
                        dest="batch_size", default=hyper_params["batch_size"])
    
    parser.add_argument("-cibs", "--cnn-inference-batch-size",
                    help="cnn batch size for inference", type=int,
                    dest="cnn_inference_batch_size", default=hyper_params["cnn_inference_batch_size"])

    parser.add_argument("-ep", "--epochs",
                        help="number of epochs", type=int,
                        dest="epochs", default=hyper_params["epochs"])
    
    parser.add_argument("-l2r", "--l2-regularisation",
                        help="l2 regularisation", type=float,
                        dest="l2_regularisation", default=hyper_params["l2_regularisation"])

    parser.add_argument("-drp", "--dropout",
                        help="dropout", type=float,
                        dest="dropout", default=hyper_params["dropout"])
    
    parser.add_argument("-lr", "--learning-rate",
                        help="learning rate", type=float,
                        dest="learning_rate", default=hyper_params["learning_rate"])        
    
    parser.add_argument("-pt", "-percentage-train",
                        help="percentage of learning data to become trainining data", type=int,
                        dest="percentage_train", default=hyper_params["percentage_train"])
    
    parser.add_argument("-wpi", "--windows-per-image",
                        help="windows per image on average", type=int,
                        dest="windows_per_image_on_average", default=hyper_params["windows_per_image_on_average"])
    
    parser.add_argument("-ws", "--window-size",
                        help="window size", type=int,
                        dest="window_size", default=hyper_params["window_size"])
   
    parser.add_argument("-mfp", "--min-fei-percentage",
                        help="minimal percentage of pixels in a window for it to be considered fei", type=float,
                        dest="min_fei_prct_window", default=hyper_params["min_fei_prct_window"])

    parser.add_argument("-mnp", "--max-non-fei-percentage",
                        help="maximum percentage of fei pixels in a window for it to be considered non fei", type=float,
                        dest="max_non_fei_prct_window", default=hyper_params["max_non_fei_prct_window"])
    
    parser.add_argument("-fwp", "--fei-windows-percentage",
                        help="percentage of windows per image randomly chosen which need to be considered fei", type=float,
                        dest="fei_window_percentage", default=hyper_params["fei_window_percentage"]) 
    
    parser.add_argument("-fc", "--filters-count",
                        help="filters in unet count", type=int,
                        dest="filters_count", default=hyper_params["filters_count"])
    
    parser.add_argument("-ks", "--kernel-size",
                        help="kernel size", type=int,
                        dest="kernel_size", default=hyper_params["kernel_size"])    

    parser.add_argument("-ac", "--augmentations-count",
                        help="augmentations count", type=int,
                        dest="augmentations_count", default=hyper_params["augmentations_count"])
    
    parser.add_argument("-cmwc", "--cnn-min-window-count",
                        help="cnn minimum count of windows to cover input training b&w image", type=int,
                        dest="cnn_min_window_count", default=hyper_params["cnn_min_window_count"])

    parser.add_argument("-cmpp", "--cnn-min-pixels-per-window-prct",
                        help="cnn minimum percentage of pixels in a window for cnn training", type=float,
                        dest="cnn_min_pixels_per_window_prct", default=hyper_params["cnn_min_pixels_per_window_prct"])    
    
    # Image processing parameters

    parser.add_argument('-sg', '--segment',
                        help='with this option the program will segment images locally without uploading, downloading them',
                        dest='segment_locally', action='store_true')

    parser.add_argument("-na", "--norm-alpha",
                        help="normalization alpha", type=int,
                        dest="norm_alpha", default=image_processing_params["norm"]["alpha"])

    parser.add_argument("-nb", "--norm-beta",
                        help="normalization beta", type=int,
                        dest="norm_beta", default=image_processing_params["norm"]["beta"])

    parser.add_argument("-nt", "--norm-type",
                        help="normalization type", type=str,
                        dest="norm_type", default=image_processing_params["norm"]["norm_type"])
    
    parser.add_argument("-no", "--norm-order",
                        help="normalization order (from 0 to 3, -1 means disabled)", type=str,
                        dest="norm_order", default=image_processing_params["norm"]["order"])    

    parser.add_argument("-cl", "--clahe-clip-limit",
                        help="CLAHE clip limit", type=int,
                        dest="clahe_clip_limit", default=image_processing_params["clahe"]["clip_limit"])

    parser.add_argument("-ct", "--clahe-tile-grid-size",
                        help="CLAHE tile grid size", type=int,
                        dest="clahe_tile_grid_size", default=image_processing_params["clahe"]["tile_grid_size"])
    
    parser.add_argument("-co", "--clahe-order",
                        help="CLAHE order (from 0 to 3, -1 means disabled)", type=str,
                        dest="clahe_order", default=image_processing_params["clahe"]["order"])    

    parser.add_argument("-dp", "--denoise-patch-size",
                        help="denoise patch size", type=int,
                        dest="denoise_patch_size", default=image_processing_params["denoise"]["patch_size"])

    parser.add_argument("-dd", "--denoise-patch-distance",
                        help="denoise patch distance", type=int,
                        dest="denoise_patch_distance", default=image_processing_params["denoise"]["patch_distance"])

    parser.add_argument("-ds", "--denoise-strength",
                        help="denoise strength", type=int,
                        dest="denoise_strength", default=image_processing_params["denoise"]["strength"])    

    parser.add_argument("-do", "--denoise-order",
                        help="denoise order (from 0 to 3, -1 means disabled)", type=int,
                        dest="denoise_order", default=image_processing_params["denoise"]["order"])

    parser.add_argument("-mk", "--median-ksize",
                        help="median filter kernel size", type=int,
                        dest="median_ksize", default=image_processing_params["median"]["ksize"])
    
    parser.add_argument("-mo", "--median-order",
                        help="median order (from 0 to 3, -1 means disabled)", type=int,
                        dest="median_order", default=image_processing_params["median"]["order"])
        
    # Segmentation CNN training parameters
    parser.add_argument('-spt', '--segmentation-pretraining',
                        help='with this option program will perform dummy segmentation along for further CNN model training for best segmentation result pickup',
                        dest='segmentation_pretraining', default=False, action='store_true')
    parser.add_argument('-st', '--segmentation-training',
                        help='with this option program will use dummy segmentation for CNN model training for best segmentation result pickup',
                        dest='segmentation_training', default=False, action='store_true')
    
    parser.add_argument("-std", "--segmentation-training-dir",
                        help="segmentation training directory", type=str,
                        dest="segmentation_training_dir", default=paths_params["segmentation_training_dir"])

    # Segmentation parameters
    parser.add_argument("-siod", "--segmentation-input-original-dir",
                        help="segmentation input original directory", type=str,
                        dest="segmentation_input_original_dir", default=paths_params["segmentation_input_original_dir"])
    
    parser.add_argument("-sigd", "--segmentation-input-ground-dir",
                        help="segmentation input ground directory", type=str,
                        dest="segmentation_input_ground_dir", default=paths_params["segmentation_input_ground_dir"])

    parser.add_argument("-sod", "--segmentation-output-dir",
                        help="segmentation output directory", type=str,
                        dest="segmentation_output_dir", default=paths_params["segmentation_output_dir"])
    
    parser.add_argument("-ssf", "--segmentation-summary-file",
                        help="segmentation summary file", type=str,
                        dest="segmentation_summary_file", default=paths_params["segmentation_summary_file"])

    parser.add_argument("-sdf", "--segmentation-dbl-file",
                        help="segmentation dbl file", type=str,
                        dest="segmentation_dbl_file", default=paths_params["segmentation_dbl_file"])

    parser.add_argument("-csr", "--consider-single-region",
                        help="consider single region", type=bool,
                        dest="consider_single_region", default=segmentation_params["consider_single_region"])

    parser.add_argument("-cor", "--consider-oval-region",
                        help="consider oval region", type=bool,
                        dest="consider_oval_region", default=segmentation_params["consider_oval_region"])

    parser.add_argument("-el", "--eccentricity-level",
                        help="eccentricity level", type=float,
                        dest="eccentricity_level", default=segmentation_params["eccentricity_level"])

    parser.add_argument("-tl", "--threshold-level",
                        help="threshold level", type=int,
                        dest="threshold_level", default=segmentation_params["threshold_level"])

    parser.add_argument("-ssisf", "--small-image-scale-factor",
                        help="segment small image scale factor", type=float,
                        dest="segment_small_image_scale_factor", default=segmentation_params["segment_small_image_scale_factor"])

    parser.add_argument("-smps1", "--segment-min-percent-scale",
                        help="segment min percent scale", type=int,
                        dest="segment_min_percent_scale", default=segmentation_params["segment_min_percent_scale"])

    parser.add_argument("-smps2", "--segment-max-percent-scale",
                        help="segment max percent scale", type=int,
                        dest="segment_max_percent_scale", default=segmentation_params["segment_max_percent_scale"])

    parser.add_argument("-srps1", "--segment-rough-percent-step-1",
                        help="segment rough percent step 1", type=int,
                        dest="segment_rough_percent_step_1", default=segmentation_params["segment_rough_percent_step_1"])

    parser.add_argument("-srps2", "--segment-rough-percent-step-2",
                        help="segment rough percent step 2", type=int,
                        dest="segment_rough_percent_step_2", default=segmentation_params["segment_rough_percent_step_2"])

    parser.add_argument("-sri", "--segment-rough-interval",
                        help="segment rough interval", type=int,
                        dest="segment_rough_interval", default=segmentation_params["segment_rough_interval"])

    parser.add_argument("-srss1", "--segment-rough-sliding-step-1",
                        help="segment rough sliding step 1", type=int,
                        dest="segment_rough_sliding_step_1", default=segmentation_params["segment_rough_sliding_step_1"])

    parser.add_argument("-srss2", "--segment-rough-sliding-step-2",
                        help="segment rough sliding step 2", type=int,
                        dest="segment_rough_sliding_step_2", default=segmentation_params["segment_rough_sliding_step_2"])

    parser.add_argument("-sdss", "--segment-detailed-sliding-step",
                        help="segment detailed sliding step", type=int,
                        dest="segment_detailed_sliding_step", default=segmentation_params["segment_detailed_sliding_step"])

    parser.add_argument("-sjwp", "--segment-jaccard-windows-percentage",
                        help="segmented images percentage of windows which will be calculated by jaccard estimation", type=int,
                        dest="segment_jaccard_windows_percentage", default=segmentation_params["segment_jaccard_windows_percentage"])

    parser.add_argument("-svop", "--segment-voting-overlap-percentage",
                        help="voting windows subsequent overlap percentage", type=int,
                        dest="segment_voting_overlap_percentage", default=segmentation_params["segment_voting_overlap_percentage"])  

    parser.add_argument("-svp", "--segment-voting-percentage",
                        help="segment image cover voting percentage", type=int,
                        dest="segment_voting_percentage", default=segmentation_params["segment_voting_percentage"])    

    parser.add_argument("-svpm", "--segment-voting-percentage-mult",
                        help="segment image cover voting percentage multiplier", type=float,
                        dest="segment_voting_percentage_mult", default=segmentation_params["segment_voting_percentage_mult"])

    parser.add_argument("-ptr", "--padding-to-remove",
                        help="padding to remove", type=int,
                        dest="padding_to_remove", default=segmentation_params["padding_to_remove"])

    parser.add_argument("-ss", "--segmentation-suffix",
                        help="segmentation suffix", type=str,
                        dest="segmentation_suffix", default=segmentation_params["segmentation_suffix"])
    
    # multiple batches segmentation parameters   
    parser.add_argument('-sgm', '--segment-multiple',
                        help='with this option the program will do complex segmentation of images locally, but under many conditions',
                        dest='segment_multiple', action='store_true')
    
    parser.add_argument('-sgms', '--segment-multiple-simple',
                        help='with this option the program will do simple segmentation of images locally, but under many conditions',
                        dest='segment_multiple_simple', action='store_true')
    
    parser.add_argument('-sgmp', '--segment-multiple-params',
                        help='parameters for complex segmentation in batches. more details in src/data/results/README.md',
                        dest='segment_multiple_params', default=segmentation_params["segment_multiple_params"])
    
    parser.add_argument('-sgsmp', '--segment-simple-multiple-params',
                        help='parameters for simple segmentation in batches. more details in src/data/results/README.md',
                        dest='segment_simple_multiple_params', default=segmentation_params["segment_simple_multiple_params"])

    args = parser.parse_args()
    # print current parameters
    print('Default parameters are:')
    print_parameters(paths_params, hyper_params, image_processing_params, segmentation_params)

    print('Modified parameters passed from command line are: ')
    for arg, value in vars(args).items():
        print(f"  {arg} = {value}")

    return args