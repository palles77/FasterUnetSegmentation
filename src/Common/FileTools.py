import errno
import os
import shutil
import zipfile

#-----------------------------------
# creates a parent directory
# for a given file name
# file_name - directory name
#-----------------------------------
def empty_or_create_directory(dir_name, remove_if_exists = True):
        
    # remove directory if it exists
    if remove_if_exists and os.path.exists(dir_name):
        try:
            shutil.rmtree(dir_name)
        except OSError as e:
            print("Error: %s : %s" % (dir_name, e.strerror))    
        
    # setup empty directory
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)

        # Guard against race condition
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
   
#-----------------------------------
# create a directory
#-----------------------------------         
def create_directory(directory):
    
    if not os.path.exists(directory):
        os.makedirs(directory)
        
#----------------------------------
# Function for creating a zip file
#----------------------------------
def create_zip(files, zip_filename):
    with zipfile.ZipFile(zip_filename, 'w') as zipf:
        for file in files:
            zipf.write(file, arcname=file)
            
#----------------------------------
# Function for creating a zip file
#----------------------------------
def create_high_compression_zip(files, zip_filename):
    with zipfile.ZipFile(zip_filename, 'w', compression=zipfile.ZIP_DEFLATED) as zipf:
        for file in files:
            zipf.write(file, arcname=file, compress_type=zipfile.ZIP_DEFLATED)
            
#-----------------------------------------------------
# Function for extracting a zip file
#-----------------------------------------------------
def extract_zip(zip_filename, extract_dir):
    with zipfile.ZipFile(zip_filename, 'r') as zipf:
        zipf.extractall(extract_dir)
        
#-----------------------------------------------------
# Function searching for mode_file_name from main dir
#-----------------------------------------------------
def retrieve_and_clean(destination_dir, search_dir, model_file_name):
    """Searches for model_file_name within the contents of main_dir, moves it to main_dir,
    and deletes all other files in main_dir.
    
    Args:
        main_dir (str): The directory to search for model_file_name and clean.
        model_file_name (str): The name of the file to retrieve and move.
    """
    # Get a list of all files and directories in main_dir
    files = os.listdir(search_dir)
    
    # Initialize variables to hold the path of the model file and the paths of all other files
    model_file_path = None
    other_file_paths = []
    
    # Iterate over all files and directories in main_dir
    for f in files:
        # If f is a directory, recursively search it for the model file
        if os.path.isdir(os.path.join(search_dir, f)):
            sub_model_file_path, sub_other_file_paths = retrieve_and_clean(
                destination_dir, os.path.join(search_dir, f), model_file_name)
            
            # If the model file was found in this subdirectory, set its path and add the other file paths to the list
            if sub_model_file_path is not None:
                model_file_path = sub_model_file_path
                other_file_paths.extend(sub_other_file_paths)
        
        # If f is the model file, set its path
        elif f == model_file_name:
            model_file_path = os.path.join(search_dir, f)
        
        # If f is not a directory or the model file, add its path to the list of other file paths
        else:
            other_file_paths.append(os.path.join(search_dir, f))
    
    # If the model file was found, move it to main_dir and delete all other files
    if model_file_path is not None:
        # Move the model file to main_dir
        parent_dir = os.path.dirname(model_file_path)
        destination_path = os.path.join(destination_dir, os.path.basename(model_file_path))
        if parent_dir != destination_dir and not os.path.exists(destination_path):
            shutil.move(model_file_path, destination_dir)
        
        # Delete all other files and directories in search_dir
        if search_dir != destination_dir:
            try:
                deletion_dir_or_files = os.listdir(search_dir)
                for deletion_dir_or_file in deletion_dir_or_files:
                    full_deletion_dir_or_file = os.path.join(search_dir, deletion_dir_or_file)
                    if os.path.exists(full_deletion_dir_or_file):
                        if os.path.isfile(full_deletion_dir_or_file):
                            os.remove(full_deletion_dir_or_file)
                        elif os.path.isdir(full_deletion_dir_or_file):
                            shutil.rmtree(full_deletion_dir_or_file)
                shutil.rmtree(search_dir)   
            except Exception as error: 
                print(f"Error in retrieve_and_clean with '{search_dir}', details: '{error}'")
                        
    # Return the path of the model file and the paths of all other files in main_dir
    return model_file_path, other_file_paths
