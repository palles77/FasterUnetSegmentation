import sys
sys.path.append('./') if './' not in sys.path else None
sys.path.append('../') if '../' not in sys.path else None

import glob as glob_module
from Common.FileTools import *
from Common.Parser import create_arguments
from Common.Tools import *
from ImageOperations.BatchSegmentations import *
from Parameters.DefaultParameters import *
from Learning.Cnn import Cnn

#===============================================================#
#                                                               #
#===============================================================# 
"""
Main routine.
"""

# Validate parameters

how_many_windows_per_training_npz = hyper_params["windows_per_image_on_average"] * (hyper_params["augmentations_count"] + 1) * \
                                    hyper_params["generation_images_per_npz"]
print(f"how_many_windows_per_training_npz: {how_many_windows_per_training_npz}")
if how_many_windows_per_training_npz % hyper_params["batch_size"] != 0:
    raise Exception("The number of windows generated per npz combined training file should be a multiplication of batch size")
if hyper_params["generated_images_to_save"] % hyper_params["generation_images_per_npz"] != 0:
    raise Exception("The number of generated images to save should be a multiplication of generation images per npz file")

cml_args = create_arguments()

#===============================================================#
#                                                               #
#===============================================================# 
if cml_args.cleanup:
    processed_x_images_dir = path.join(cml_args.x_dir, "processed")
    processed_y_images_dir = path.join(cml_args.y_dir, "processed")    
    empty_or_create_directory(processed_x_images_dir)
    empty_or_create_directory(processed_y_images_dir)
    empty_or_create_directory(cml_args.generate_dir)

    empty_or_create_directory(cml_args.models_dir, False)
    empty_or_create_directory(cml_args.checkup_img_dir)
    empty_or_create_directory(cml_args.segmentation_output_dir)

#===============================================================#
#                                                               #
#===============================================================# 
if cml_args.generate_data:
    
    create_directory(cml_args.generate_dir)
    create_directory(cml_args.x_dir)
    create_directory(cml_args.y_dir)

    print("Generating data...")

    # step 1 - generate
    generate_data(cml_args)

#===============================================================#
# Step 3                                                        #
#===============================================================# 
if cml_args.train:  
    
    # step 3 - train
    unet = load_unet(cml_args)

    # train is only enabled via command line
    zipped_model = unet.train(cml_args)
    
#===============================================================#
# Step 4                                                        #
#===============================================================# 
if cml_args.segmentation_pretraining:  
    
    create_directory(cml_args.segmentation_training_dir)
    
    # This is online segmentation with vessels model
    cur_dir = os.getcwd()
    pre_segment_local_files_list_by_specific_model(cml_args)
    os.chdir(cur_dir)
    
#===============================================================#
# Step 5                                                        #
#===============================================================# 
if cml_args.segmentation_training:  
    
    create_directory(cml_args.segmentation_training_dir)    
    cur_dir = os.getcwd()
  
    cnn_trainer = Cnn(
        cml_args.segmentation_training_dir, 
        cml_args.models_dir, 
        cml_args.cnn_model_file, 
        cml_args.epochs, 
        cml_args.batch_size, 
        cml_args.learning_rate,
        cml_args.window_size,
        cml_args.cnn_min_window_count,
        cml_args.cnn_min_pixels_per_window_prct
    )
    
    cnn_trainer.run()

    # iterate through all images in images_dir and run prediction
    # Load the best model state
    cnn_trainer.model.load_state_dict(torch.load(os.path.join(cnn_trainer.save_dir, cml_args.cnn_model_file)))
    cnn_trainer.model.eval()

    # Iterate through all images in images_dir and run prediction
    image_paths = glob_module.glob(os.path.join(cml_args.segmentation_training_dir, "*.png"))
    for image_path in image_paths:
        jaccard_distance = cnn_trainer.predict(image_path)
        print(f"Image: {image_path}, Predicted Jaccard Distance: {jaccard_distance:.6f}")
        
    os.chdir(cur_dir)

#===============================================================#
# STEP 6                                                        #
#===============================================================# 
if cml_args.segment_locally:
    
    create_directory(cml_args.segmentation_output_dir)
    
    # This is online segmentation with vessels model
    cur_dir = os.getcwd()
    segment_local_files_list_by_voting_specific_model(cml_args)
    os.chdir(cur_dir)
    
#===============================================================#
# STEP 7                                                        #
#===============================================================# 
if cml_args.segment_multiple:
    
    create_directory(cml_args.segmentation_output_dir)
    
    # This is online segmentation with vessels model
    cur_dir = os.getcwd()
    segment_local_files_list_by_specific_model_multiple_times(cml_args)
    os.chdir(cur_dir)
    
#===============================================================#
# STEP 8                                                        #
#===============================================================# 
if cml_args.segment_multiple_simple:
    
    create_directory(cml_args.segmentation_output_dir)
    
    # This is online segmentation with vessels model
    cur_dir = os.getcwd()
    segment_local_files_list_by_not_voting_specific_model(cml_args)
    os.chdir(cur_dir)    
  
print("Finished")