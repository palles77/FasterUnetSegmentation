import sys
sys.path.append('./') if './' not in sys.path else None
sys.path.append('../') if '../' not in sys.path else None

from Common.FileTools import *
from Common.Tools import *
from Parameters.DefaultParameters import *
from ImageOperations.Segmentation import * #pre_segment_image_into_region, segment_image_by_voting_into_region, segment_image_into_region, segment_simple_image_into_region_no_voting
from Learning.Cnn import Cnn

#===============================================================#
#                                                               #
#===============================================================#  
def rename_output_segmented_file_name(
    input_file_name, 
    input_segment_file_name, 
    first_file_name_suffix, 
    second_file_name_suffix,
    extension = ".png"):
    """
    Renames the output segmented file name based on the input file name and suffixes.

    Args:
        input_file_name (str): The name of the input file.
        input_segment_file_name (str): The name of the input segmented file.
        first_file_name_suffix (str): The first suffix to add to the output file name.
        second_file_name_suffix (str): The second suffix to add to the output file name.

    Returns:
        str: The name of the output segmented file.
    """
    # Create the output file name by replacing the file extension with the suffixes
    if extension[0] == '.':
        extension = extension[1:]
    segment_file_name_directory = os.path.dirname(input_segment_file_name)
    base_file_name = os.path.basename(input_file_name)
    base_file_name = os.path.splitext(base_file_name)[0]
    output_segment_file_name = f"{base_file_name}{first_file_name_suffix}{second_file_name_suffix}.{extension}"
    output_segment_file_name = os.path.join(segment_file_name_directory, output_segment_file_name)
    
    # Rename the input segmented file to the output file name
    if input_segment_file_name != output_segment_file_name:
        shutil.move(input_segment_file_name, output_segment_file_name)
    
    # Return the name of the output segmented file
    return output_segment_file_name

#===============================================================#
#                                                               #
#===============================================================# 
def segment_local_files_list_by_specific_model(
    cml_args):
    """
    Segments a list of bitmap files using a specific model and saves the segmented images to Azure Blob Storage.

    Args:
        cml_args (object): The command line arguments.

    Returns:
        None
    """
    
    unet_model_path = os.path.join(cml_args.unet_model_file)
    
    # This is local processing   
    empty_or_create_directory(cml_args.segmentation_output_dir, remove_if_exists=False)
    
    unet = load_unet(cml_args)

    # load the model created
    model_state_dict = torch.load(unet_model_path, weights_only=True)
    unet.model.load_state_dict(model_state_dict)
    
    files_to_segment = [f for f in os.listdir(cml_args.segmentation_input_original_dir) if os.path.isfile(os.path.join(cml_args.segmentation_input_original_dir, f)) and f.endswith('.png')]
    files_to_segment_filtered = [os.path.basename(file) for file in files_to_segment]
    
    if cml_args.segmentation_dbl_file is not None:  
        files_to_segment_filtered = []
        # open cml_args.segment_dbl_file
        with open(cml_args.segmentation_dbl_file, 'r') as segment_dbl_file:
            # parent_dir = os.path.dirname(cml_args.segmentation_dbl_file)
            # os.chdir(parent_dir)
            # Iterate over each line in the file
            for png_file_name in segment_dbl_file:
            
                png_file_name = png_file_name.strip()
                file_extension = png_file_name.split('.')[-1]
                if file_extension.lower() != 'png':
                    print(f"Error: {png_file_name} is not a png file. It will not be processed")
                    continue
                                        
                files_to_segment_filtered.append(png_file_name)
                
    # Now we have the list of files to segment
    with open(cml_args.segmentation_summary_file, 'w') as output_file:
        std_out_capture = StdOutCapture(output_file)
        sys.stdout = std_out_capture
        jaccard_distances = []  # List to store Jaccard distances
        for png_file_name in files_to_segment_filtered:
            print(f"Segmenting {png_file_name}...")
            png_full_file_name = os.path.join(cml_args.segmentation_input_original_dir, png_file_name)
            try:
                (segmented_grey_file_name, segmented_binary_file_name) = segment_image_into_region(
                    unet, png_full_file_name, cml_args)
            
                rename_output_segmented_file_name(png_file_name, segmented_grey_file_name, cml_args.segmentation_suffix, "_gr")
                segmented_binary_file_name = rename_output_segmented_file_name(png_file_name, segmented_binary_file_name, cml_args.segmentation_suffix, "_bn")
                
                # check if ground truth file exists
                ground_file_name = os.path.join(cml_args.segmentation_input_ground_dir, png_file_name)
                if os.path.exists(ground_file_name):
                    image = cv2.imread(png_full_file_name)
                    image_size = (image.shape[1], image.shape[0])  # (width, height)

                    jaccard_distance = Cnn.compute_jaccard_distance(image_size, ground_file_name, segmented_binary_file_name)
                    jaccard_distances.append(jaccard_distance)  # Store the Jaccard distance
                    print(f"Ground image file: {png_file_name}, segmented file Jaccard Distance: {jaccard_distance:.6f}")
                
            except Exception as e:
                print(f"Failed to segment file: {png_file_name}, with error: {str(e)}")
        
            # Calculate and print the average Jaccard distance
            if jaccard_distances:
                average_jaccard_distance = sum(jaccard_distances) / len(jaccard_distances)
                print(f"Average Jaccard Distance: {average_jaccard_distance:.6f}")
            else:
                print("No Jaccard distances were calculated.")

#===============================================================#
#                                                               #
#===============================================================# 
def segment_local_files_list_by_voting_specific_model(
    cml_args):
    """
    Segments a list of bitmap files using a specific model and voting.

    Args:
        cml_args (object): The command line arguments.

    Returns:
        None
    """
    
    unet_model_path = os.path.join(cml_args.unet_model_file)
    
    # This is local processing   
    empty_or_create_directory(cml_args.segmentation_output_dir, remove_if_exists=False)
    
    unet = load_unet(cml_args)

    # load the model created
    model_state_dict = torch.load(unet_model_path, weights_only=True)
    unet.model.load_state_dict(model_state_dict)
    
    files_to_segment = [f for f in os.listdir(cml_args.segmentation_input_original_dir) if os.path.isfile(os.path.join(cml_args.segmentation_input_original_dir, f)) and f.endswith('.png')]
    files_to_segment_filtered = [os.path.basename(file) for file in files_to_segment]
    
    if cml_args.segmentation_dbl_file is not None:  
        files_to_segment_filtered = []
        # open cml_args.segment_dbl_file
        with open(cml_args.segmentation_dbl_file, 'r') as segment_dbl_file:
            # parent_dir = os.path.dirname(cml_args.segmentation_dbl_file)
            # os.chdir(parent_dir)
            # Iterate over each line in the file
            for png_file_name in segment_dbl_file:
            
                png_file_name = png_file_name.strip()
                file_extension = png_file_name.split('.')[-1]
                if file_extension.lower() != 'png':
                    print(f"Error: {png_file_name} is not a png file. It will not be processed")
                    continue
                                        
                files_to_segment_filtered.append(png_file_name)
                
    # Now we have the list of files to segment
    with open(cml_args.segmentation_summary_file, 'w') as output_file:
        std_out_capture = StdOutCapture(output_file)
        sys.stdout = std_out_capture
        jaccard_distances = []  # List to store Jaccard distances
        for png_file_name in files_to_segment_filtered:
            print(f"Segmenting {png_file_name}...")
            png_full_file_name = os.path.join(cml_args.segmentation_input_original_dir, png_file_name)
            try:
                (segmented_grey_file_name, segmented_binary_file_name) = segment_image_by_voting_into_region(
                    unet, png_full_file_name, cml_args)
            
                rename_output_segmented_file_name(png_file_name, segmented_grey_file_name, cml_args.segmentation_suffix, "_gr")
                segmented_binary_file_name = rename_output_segmented_file_name(png_file_name, segmented_binary_file_name, cml_args.segmentation_suffix, "_bn")
                
                # check if ground truth file exists
                ground_file_name = os.path.join(cml_args.segmentation_input_ground_dir, png_file_name)
                if os.path.exists(ground_file_name):
                    image = cv2.imread(png_full_file_name)
                    image_size = (image.shape[1], image.shape[0])  # (width, height)

                    jaccard_distance = Cnn.compute_jaccard_distance(image_size, ground_file_name, segmented_binary_file_name)
                    jaccard_distances.append(jaccard_distance)  # Store the Jaccard distance
                    print(f"Ground image file: {png_file_name}, segmented file Jaccard Distance: {jaccard_distance:.6f}")
                
            except Exception as e:
                print(f"Failed to segment file: {png_file_name}, with error: {str(e)}")
        
            # Calculate and print the average Jaccard distance
            if jaccard_distances:
                average_jaccard_distance = sum(jaccard_distances) / len(jaccard_distances)
                print(f"Average Jaccard Distance: {average_jaccard_distance:.6f}")
            else:
                print("No Jaccard distances were calculated.")

#===============================================================#
#                                                               #
#===============================================================#           
def segment_local_files_list_by_not_voting_specific_model(cml_args):
    """
    Segments simple a list of bitmap files using a specific model and saves the segmented images to local, by not voting on windows.

    Args:
        cml_args (object): The command line arguments.

    Returns:
        None
    """
        
    unet_model_path = os.path.join(cml_args.unet_model_file)
    cml_args.segmentation_summary_file = os.path.join(cml_args.segmentation_output_dir, 'summary.txt')
    
    # This is local processing   
    empty_or_create_directory(cml_args.segmentation_output_dir, remove_if_exists=False)
    
    unet = load_unet(cml_args)

    # load the model created
    model_state_dict = torch.load(unet_model_path, weights_only=True)
    unet.model.load_state_dict(model_state_dict)
    
    files_to_segment = [f for f in os.listdir(cml_args.segmentation_input_original_dir) if os.path.isfile(os.path.join(cml_args.segmentation_input_original_dir, f)) and f.endswith('.png')]
    files_to_segment_filtered = [os.path.basename(file) for file in files_to_segment]
    
    if cml_args.segmentation_dbl_file is not None:  
        files_to_segment_filtered = []
        # open cml_args.segment_dbl_file
        with open(cml_args.segmentation_dbl_file, 'r') as segment_dbl_file:
            # parent_dir = os.path.dirname(cml_args.segmentation_dbl_file)
            # os.chdir(parent_dir)
            # Iterate over each line in the file
            for png_file_name in segment_dbl_file:
            
                png_file_name = png_file_name.strip()
                file_extension = png_file_name.split('.')[-1]
                if file_extension.lower() != 'png':
                    print(f"Error: {png_file_name} is not a png file. It will not be processed")
                    continue
                                        
                files_to_segment_filtered.append(png_file_name)
                
    # Now we have the list of files to segment
    with open(cml_args.segmentation_summary_file, 'w') as output_file:
        std_out_capture = StdOutCapture(output_file)
        sys.stdout = std_out_capture
        jaccard_distances = []  # List to store Jaccard distances
        for png_file_name in files_to_segment_filtered:
            print(f"Segmenting {png_file_name}...")
            png_full_file_name = os.path.join(cml_args.segmentation_input_original_dir, png_file_name)
            try:
                (segmented_grey_file_name, segmented_binary_file_name) = segment_simple_image_into_region_no_voting(
                    unet, png_full_file_name, cml_args)
            
                rename_output_segmented_file_name(png_file_name, segmented_grey_file_name, cml_args.segmentation_suffix, "_gr")
                segmented_binary_file_name = rename_output_segmented_file_name(png_file_name, segmented_binary_file_name, cml_args.segmentation_suffix, "_bn")
                
                # check if ground truth file exists
                ground_file_name = os.path.join(cml_args.segmentation_input_ground_dir, png_file_name)
                if os.path.exists(ground_file_name):
                    image = cv2.imread(png_full_file_name)
                    image_size = (image.shape[1], image.shape[0])  # (width, height)

                    jaccard_distance = Cnn.compute_jaccard_distance(image_size, ground_file_name, segmented_binary_file_name)
                    jaccard_distances.append(jaccard_distance)  # Store the Jaccard distance
                    print(f"Ground image file: {png_file_name}, segmented file Jaccard Distance: {jaccard_distance:.6f}")
                
            except Exception as e:
                print(f"Failed to segment file: {png_file_name}, with error: {str(e)}")
        
            # Calculate and print the average Jaccard distance
            if jaccard_distances:
                average_jaccard_distance = sum(jaccard_distances) / len(jaccard_distances)
                print(f"Average Jaccard Distance: {average_jaccard_distance:.6f}")
            else:
                print("No Jaccard distances were calculated.")
      
#===============================================================#
#                                                               #
#===============================================================# 
def segment_local_files_list_by_specific_model_multiple_times(cml_args):
    """
    Segments a list of bitmap files using a specific model and saves the segmented images to local disc.
    Segments the images in batches.

    Args:
        cml_args (object): The command line arguments.

    Returns:
        None
    """
    
    batches_string = cml_args.segment_multiple_params
    batches = batches_string.split(',')
    base_segment_dir = cml_args.segmentation_output_dir
    for batch in batches:
        batch_parameters = batch.split('_')
        cml_args.segment_min_percent_scale = int(batch_parameters[1])
        cml_args.segment_max_percent_scale = int(batch_parameters[2])
        cml_args.segment_rough_percent_step_1 = int(batch_parameters[3])
        cml_args.segment_rough_interval = int(batch_parameters[4])
        cml_args.segment_rough_percent_step_2 = int(batch_parameters[5])

        # Means how much percentage at minimum a random window should overlap to be considered a valid detection
        cml_args.segment_voting_overlap_percentage = int(batch_parameters[6])
        
        # How big the area of the original image should be covered by random windows
        cml_args.segment_voting_percentage = int(batch_parameters[7])
        
        # How much area of the image should be increased to multiply random windows
        # It needs to be higher than 0.0 for random windows to be considered in calculations in voting mechanism
        # When zero than random windows are taken as they were generated
        cml_args.segment_voting_percentage_mult = int(batch_parameters[8])
        
        # How many basic windows should have their Jaccard index estimated to be included in image segmentation assesment
        # With this we can control speed of image segmentation assessment as it is quite costly
        cml_args.segment_jaccard_windows_percentage = int(batch_parameters[9])
        
        cml_args.segmentation_output_dir = os.path.join(base_segment_dir, batch)
        cml_args.segmentation_summary_file = os.path.join(cml_args.segmentation_output_dir, 'summary.txt')
        
        create_directory(cml_args.segmentation_output_dir)
        segment_local_files_list_by_voting_specific_model(cml_args)
            
#===============================================================#
#                                                               #
#===============================================================# 
def pre_segment_local_files_list_by_specific_model(
    cml_args):
    """
    Pre segments a list of bitmap files using a specific model and saves the segmented images to Azure Blob Storage.

    Args:
        cml_args (object): The command line arguments.

    Returns:
        None
    """
    
    unet_model_path = os.path.join(cml_args.unet_model_file)
    
    # This is local processing   
    empty_or_create_directory(cml_args.segmentation_output_dir, remove_if_exists=False)
    
    unet = load_unet(cml_args)

    # load the model created
    model_state_dict = torch.load(unet_model_path, weights_only=True)
    unet.model.load_state_dict(model_state_dict)
    
    files_to_segment = [f for f in os.listdir(cml_args.segmentation_input_original_dir) if os.path.isfile(os.path.join(cml_args.segmentation_input_original_dir, f)) and f.endswith('.png')]
    files_to_segment_filtered = [os.path.basename(file) for file in files_to_segment]
    
    if cml_args.segmentation_dbl_file is not None:  
        files_to_segment_filtered = []      
        # open cml_args.segment_dbl_file
        with open(cml_args.segmentation_dbl_file, 'r') as segment_dbl_file:
            # parent_dir = os.path.dirname(cml_args.segmentation_dbl_file)
            # os.chdir(parent_dir)
            # Iterate over each line in the file
            for png_file_name in segment_dbl_file:
            
                png_file_name = png_file_name.strip()
                file_extension = png_file_name.split('.')[-1]
                if file_extension.lower() != 'png':
                    print(f"Error: {png_file_name} is not a png file. It will not be processed")
                    continue
                                        
                files_to_segment_filtered.append(png_file_name)
                
    # Now we have the list of files to segment
    for png_file_name in files_to_segment_filtered:       
        print(f"Segmenting {png_file_name}...")
        png_full_file_name = os.path.join(cml_args.segmentation_input_original_dir, png_file_name)
        try:
            pre_segment_image_into_region(unet, png_full_file_name, cml_args)
        except Exception as e:
            print(f"Failed to segment file: {png_file_name}, with error: {str(e)}")   