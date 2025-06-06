import sys
sys.path.append('./') if './' not in sys.path else None
sys.path.append('../') if '../' not in sys.path else None

import random
import glob
import os
import shutil
from os import listdir, path
import numpy as np
from random import randint
from typing import Tuple
import cv2

# PyTorch Imports
import torchvision.transforms.functional as F
from PIL import Image

from Learning.WindowConversion import three_channels_to_one_hot_encoding
from ImageOperations.MaskFinder import is_window_outside_mask


class PretrainingDataGenerator():    
    """
    A class for generating training data for a neural network, using PyTorch for augmentations.
    """

    #===============================================================#
    #                                                               #
    #===============================================================#   
    def __init__(self):
        # The constructor doesn't do anything, so we just use the 'pass' keyword
        pass

    #===============================================================#
    #                                                               #
    #===============================================================#        
    def write_training_data(self,
                            x_dir: str,
                            y_dir: str,
                            generate_dir: str,
                            images_total_count: int,
                            image_height: int,
                            image_width: int, 
                            window_size: int,
                            windows_per_image_on_average: int, 
                            augmentations_count: int,
                            min_percentage_for_fei_window: float,
                            max_percentage_for_non_fei_window: float,
                            fei_window_percentage: int,
                            checkup_images_dir: str,                            
                            generated_images_to_save: int,
                            generated_windows_to_save: int,
                            combined_files_count: int,
                            percentage_train: int):
        """
        Writes training data (color images as numpy arrays) to a specified file location,
        using PyTorch-based augmentations for data augmentation.
    
        Args:
            x_dir (str): Directory containing the input training images.
            y_dir (str): Directory containing the output training images (ground truth).
            generate_dir (str): Directory where the generated data will be saved.
            images_total_count (int): Maximum number of training images to use. If -1, use all.
            image_height (int): Height of the training images.
            image_width (int): Width of the training images.
            window_size (int): Size of the windows to extract from the images.
            windows_per_image_on_average (int): Average number of windows to extract per image.
            augmentations_count (int): Number of augmentations to apply to each window.
            min_percentage_for_fei_window (float): Minimum percentage of foreground pixels for FEI.
            max_percentage_for_non_fei_window (float): Maximum percentage of FEI pixels for non-FEI.
            fei_window_percentage (int): Percentage of FEI windows per image.
            checkup_images_dir (str): Directory to save checkup images.
            generated_images_to_save (int): Number of generated full images to save as checkup. If -1, none.
            generated_windows_to_save (int): Number of generated windows to save as checkup. If -1, none.
            combined_files_count (int): How many .npz files to combine into one. If -1, no combination.
            percentage_train (int): Percentage of the data to be labeled "train" vs. "validation".
        """
        # Get a list of files in the directory and sort them
        x_files = sorted(listdir(x_dir))
        y_files = sorted(listdir(y_dir))
        
        # Ensure the same random order by using a consistent seed
        seed = 42
        random.seed(seed)
        random.shuffle(x_files)
        random.seed(seed)
        random.shuffle(y_files)  
        
        # Compare x_files and y_files and if different, raise an error
        if x_files != y_files:
            raise ValueError("training x_files and y_files are different")              
        
        # Determine the total number of images to process
        if images_total_count != -1:
            images_total_count = min(images_total_count, len(x_files))
        else:
            images_total_count = len(x_files)
    
        # Iterate through the images and process them
        counter = 0
        images_saved = 0
        windows_saved = 0
        for file in x_files:
            
            # Generate random windows from the images
            data_x_windowed, data_y_windowed = self.setup_random_windows(
                file, x_dir, y_dir,
                image_height, image_width, 
                window_size, windows_per_image_on_average, 
                min_percentage_for_fei_window, max_percentage_for_non_fei_window, fei_window_percentage)
            
            # Save the original images for checkup if required
            if generated_images_to_save != -1 and images_saved < generated_images_to_save:                
                file_no_extension = file.replace(".png", "")
                x_image_to_save = path.join(checkup_images_dir, "{}_{}_rgb.png".format(file_no_extension, "x"))
                y_image_to_save = path.join(checkup_images_dir, "{}_{}_rgb.png".format(file_no_extension, "y"))
                shutil.copy(path.join(x_dir, file), x_image_to_save)
                shutil.copy(path.join(y_dir, file), y_image_to_save)                
                images_saved += 1
            
            # Augment the generated windows using PyTorch transforms
            augmented_data_x_windowed, augmented_data_y_windowed = self.augment_data(
                data_x_windowed, data_y_windowed, augmentations_count)
            
            # Save the augmented windows for checkup if required
            if generated_windows_to_save != -1 and windows_saved < generated_windows_to_save:
                file_no_extension = file.replace(".png", "")
                for i in range(augmented_data_x_windowed.shape[0]):
                    x_image_to_save_path = path.join(
                        checkup_images_dir, f"{file_no_extension}_{i}_rgb_x.png"
                    )
                    y_image_to_save_path = path.join(
                        checkup_images_dir, f"{file_no_extension}_{i}_rgb_y.png"
                    )
                    # Convert from RGB to BGR before writing
                    x_image_bgr = cv2.cvtColor(augmented_data_x_windowed[i], cv2.COLOR_RGB2BGR)
                    cv2.imwrite(x_image_to_save_path, x_image_bgr)
                    cv2.imwrite(y_image_to_save_path, augmented_data_y_windowed[i])
                    
                    windows_saved += 1
                    if windows_saved >= generated_windows_to_save:
                        break
    
            # Save the generated RGB windows data
            x_short_file_name = file.replace(".png", "_rgb_x.npz")
            y_short_file_name = file.replace(".png", "_rgb_y.npz")
            x_long_data_file_name = path.join(generate_dir, x_short_file_name)
            y_long_data_file_name = path.join(generate_dir, y_short_file_name)
            
            print("Saving generated RGB windows x data to: " + x_long_data_file_name + " ...")
            np.savez_compressed(x_long_data_file_name, augmented_data_x_windowed)
            
            print("Saving generated RGB windows y data to: " + y_long_data_file_name + " ...")
            np.savez_compressed(y_long_data_file_name, augmented_data_y_windowed)
    
            # Save the training ready data (normalized X, one-hot Y)
            x_short_file_name = file.replace(".png", "_training_ready_x.npz")
            y_short_file_name = file.replace(".png", "_training_ready_y.npz")
            x_long_data_file_name = path.join(generate_dir, x_short_file_name)
            y_long_data_file_name = path.join(generate_dir, y_short_file_name)
            
            print("Saving training ready X data to: " + x_long_data_file_name + " ...")
            # Normalize X by dividing by 255
            augmented_data_x_windowed_norm = augmented_data_x_windowed / 255.0
            np.savez_compressed(x_long_data_file_name, augmented_data_x_windowed_norm)
    
            # Convert the augmented y data to one-hot encoding
            augmented_data_y_windowed_one_hot_encoding = np.zeros(
                (augmented_data_y_windowed.shape[0], window_size, window_size, 2),
                dtype=np.float32
            )
            for idx, window_y in enumerate(augmented_data_y_windowed):
                window_y_one_hot_encoding = three_channels_to_one_hot_encoding(window_y)
                augmented_data_y_windowed_one_hot_encoding[idx] = window_y_one_hot_encoding
    
            print("Saving training ready Y data to: " + y_long_data_file_name + " ...")
            np.savez_compressed(y_long_data_file_name, augmented_data_y_windowed_one_hot_encoding)
    
            if counter >= images_total_count:
                break  
            counter += 1
            
        # Handle combination into train/validation sets
        if combined_files_count == -1:
            combined_files_count = 1
    
        # Find all files in generate_dir that match the pattern *_training_ready_x.npz
        x_files_to_combine = sorted(glob.glob(os.path.join(generate_dir, '*_training_ready_x.npz')))
        y_files_to_combine = sorted(glob.glob(os.path.join(generate_dir, '*_training_ready_y.npz')))

        # Combine the files into batches of up to combined_files_count
        batch_size = combined_files_count
        for i in range(0, len(x_files_to_combine), batch_size):
            x_batch_files = x_files_to_combine[i:i + batch_size]
            y_batch_files = y_files_to_combine[i:i + batch_size]

            # Load and concatenate the data
            x_combined_data = np.concatenate([np.load(f)['arr_0'] for f in x_batch_files], axis=0)
            y_combined_data = np.concatenate([np.load(f)['arr_0'] for f in y_batch_files], axis=0)

            # Save the combined data
            combined_index = i // batch_size
            x_combined_file_name = os.path.join(generate_dir, f'training_ready_combined_{combined_index}_x.npz')
            y_combined_file_name = os.path.join(generate_dir, f'training_ready_combined_{combined_index}_y.npz')
            
            # Decide whether this batch is for training or validation
            random_number = random.randint(0, 100)
            if random_number >= percentage_train:
                x_combined_file_name = os.path.join(generate_dir, f'validation_ready_combined_{combined_index}_x.npz')
                y_combined_file_name = os.path.join(generate_dir, f'validation_ready_combined_{combined_index}_y.npz')
    
            print(f"Saving combined X data to: {x_combined_file_name} ...")
            np.savez_compressed(x_combined_file_name, x_combined_data)
            
            print(f"Saving combined Y data to: {y_combined_file_name} ...")
            np.savez_compressed(y_combined_file_name, y_combined_data)
            
        # Finally, remove the individual npz files to keep only the combined ones
        for file in x_files:
            # Remove *_rgb_x.npz and *_rgb_y.npz
            x_short_file_name = file.replace(".png", "_rgb_x.npz")
            y_short_file_name = file.replace(".png", "_rgb_y.npz")
            x_long_data_file_name = path.join(generate_dir, x_short_file_name)
            y_long_data_file_name = path.join(generate_dir, y_short_file_name)
            os.remove(x_long_data_file_name)
            os.remove(y_long_data_file_name)
    
            # Remove *_training_ready_x.npz and *_training_ready_y.npz
            x_short_file_name = file.replace(".png", "_training_ready_x.npz")
            y_short_file_name = file.replace(".png", "_training_ready_y.npz")
            x_long_data_file_name = path.join(generate_dir, x_short_file_name)
            y_long_data_file_name = path.join(generate_dir, y_short_file_name)
            os.remove(x_long_data_file_name)
            os.remove(y_long_data_file_name)

    #===============================================================#
    #                                                               #
    #===============================================================#   
    def setup_random_windows(
        self, 
        short_image_name: str,
        x_dir: str,
        y_dir: str,        
        image_height: int,
        image_width: int,
        window_size: int,
        windows_per_image_on_average: int,
        min_percentage_for_fei_window: float,
        max_percentage_for_non_fei_window: float,
        fei_window_percentage: int):
        """
        Extract random windows (patches) from the given image, balancing FEI vs. non-FEI 
        windows based on the specified parameters.
        """
        # Create empty arrays to store the data for the windows
        data_x_windowed = np.zeros((windows_per_image_on_average, window_size, window_size, 3), dtype=np.float32)
        data_y_windowed = np.zeros((windows_per_image_on_average, window_size, window_size, 3), dtype=np.float32)

        # Initialise the window index to 0
        window_index = 0
        
        print("Analysing training image {}".format(short_image_name))

        # Count the number of fei and non-fei windows needed
        fei_windows = 0
        non_fei_windows = 0
        
        x_original_image_path = path.join(x_dir, short_image_name)
        x_original_image = cv2.imread(x_original_image_path)
        
        # read the ground truth image
        y_original_image_path = path.join(y_dir, short_image_name)
        y_original_image = cv2.imread(y_original_image_path)

        # convert from BGR to RGB
        x_original_image_as_array = cv2.cvtColor(x_original_image, cv2.COLOR_BGR2RGB)
        y_original_image_as_array = cv2.cvtColor(y_original_image, cv2.COLOR_BGR2RGB)

        # Calculate the number of FEI and non-FEI windows needed for this image
        fei_windows_per_image = round(windows_per_image_on_average * fei_window_percentage / 100.0)
        non_fei_windows_per_image = windows_per_image_on_average - fei_windows_per_image

        # Set the attempts count to prevent infinite loops
        windows_attempts = 0
        while True:
            windows_attempts += 1

            # Calculate random window offsets
            x_offset = randint(0, image_width - window_size - 1)
            y_offset = randint(0, image_height - window_size - 1)

            # Extract the window from the image                
            window_x = x_original_image_as_array[y_offset : y_offset + window_size,
                                                 x_offset : x_offset + window_size, :]
            window_y = y_original_image_as_array[y_offset : y_offset + window_size,
                                                 x_offset : x_offset + window_size, : ]
            
            # Skip if window is outside mask area
            window_outside = is_window_outside_mask(window_x)
            if window_outside:
                continue

            # Check if window is FEI or non-FEI
            is_fei = self.is_window_a_fei(window_y, min_percentage_for_fei_window)
            is_non_fei = self.is_window_a_non_fei(window_y, max_percentage_for_non_fei_window)

            if is_fei:
                fei_windows += 1
            else:
                if is_non_fei:
                    non_fei_windows += 1
                else:
                    continue                
            
            # Check if we've got enough FEI and non-FEI
            if (fei_windows == fei_windows_per_image) and (non_fei_windows == non_fei_windows_per_image):
                break
            
            if windows_attempts > 1000 * windows_per_image_on_average:
                print(f"Could not find enough windows in image {short_image_name}.")
                print(f"FEI windows: {fei_windows}, non-FEI windows: {non_fei_windows}")
                break

            # Make sure we don't exceed maximum of each category
            if fei_windows > fei_windows_per_image:
                fei_windows = fei_windows_per_image
                continue

            if non_fei_windows > non_fei_windows_per_image:
                non_fei_windows = non_fei_windows_per_image
                continue

            if window_index < windows_per_image_on_average:
                data_x_windowed[window_index] = window_x
                data_y_windowed[window_index] = window_y
            else:
                break                   

            window_index += 1
        
        return data_x_windowed, data_y_windowed    
    
    #===============================================================#
    #                                                               #
    #===============================================================#   
    def augment_data(self, 
                     input_data_x_windowed: np.ndarray,
                     input_data_y_windowed: np.ndarray,
                     augmentations_count: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment the data using PyTorch's torchvision-style random transformations.

        For each window, we produce (augmentations_count + 1) total windows:
        - The original window
        - The window with randomly applied transformations, repeated augmentations_count times

        Args:
            input_data_x_windowed: Array of shape (N, H, W, 3) for input windows (RGB).
            input_data_y_windowed: Array of shape (N, H, W, 3) for mask windows (RGB or label-coded).
            augmentations_count: Number of random transformations to apply per window.

        Returns:
            A tuple (augmented_x, augmented_y), each of shape 
            (N * (augmentations_count + 1), H, W, 3).
        """

        # Set a fixed seed for reproducibility, if desired:
        random.seed(42)

        windows_count = input_data_x_windowed.shape[0]
        height = input_data_x_windowed.shape[1]
        width = input_data_x_windowed.shape[2]

        # Output arrays for the augmented data
        total_out = windows_count * (augmentations_count + 1)
        output_data_x_windowed = np.zeros((total_out, height, width, 3), dtype=np.float32)
        output_data_y_windowed = np.zeros((total_out, height, width, 3), dtype=np.float32)

        print("Augmenting data with PyTorch transforms ...")

        for i in range(windows_count):
            # The 0th slot for each block is the unmodified/original
            base_window_index = i * (augmentations_count + 1)

            original_x = input_data_x_windowed[i]
            original_y = input_data_y_windowed[i]

            # Store original
            output_data_x_windowed[base_window_index] = original_x
            output_data_y_windowed[base_window_index] = original_y

            # Convert original window to PIL Images for transforms
            pil_x_orig = Image.fromarray(original_x.astype(np.uint8))  # (H, W, 3)
            pil_y_orig = Image.fromarray(original_y.astype(np.uint8))  # (H, W, 3)

            for j in range(augmentations_count):
                # Apply the same random transformation to both x and y
                aug_x, aug_y = self._random_pytorch_transform(pil_x_orig, pil_y_orig)

                # Store in the output arrays
                index = base_window_index + j + 1
                output_data_x_windowed[index] = aug_x
                output_data_y_windowed[index] = aug_y

            print(f"Augmentation progress: {i + 1}/{windows_count} windows processed")

        return output_data_x_windowed, output_data_y_windowed

    #===============================================================#
    #                                                               #
    #===============================================================# 
    def _random_pytorch_transform(self, image_pil: Image.Image, mask_pil: Image.Image):
        """
        Apply a random combination of affine transformations (rotation, translation, shear, scale)
        as well as random horizontal/vertical flips to the image and mask in a consistent way.

        Returns:
            (augmented_image_np, augmented_mask_np)
        """

        # Convert PIL image to its width/height
        width, height = image_pil.size

        # Random rotation within [-10, 10] degrees
        angle = random.uniform(-10, 10)

        # Random translation up to 5% in each direction
        max_trans_x = 0.05 * width
        max_trans_y = 0.05 * height
        trans_x = random.uniform(-max_trans_x, max_trans_x)
        trans_y = random.uniform(-max_trans_y, max_trans_y)

        # Random scaling in [0.8, 1.0]
        scale = random.uniform(0.8, 1.0)

        # Shear range ~0.2 radians => ~11.46 degrees
        shear_degrees = random.uniform(-11.46, 11.46)

        # Apply affine to both image and mask
        aug_img = F.affine(
            image_pil,
            angle=angle,
            translate=(trans_x, trans_y),
            scale=scale,
            shear=shear_degrees,
            fill=0
        )

        aug_mask = F.affine(
            mask_pil,
            angle=angle,
            translate=(trans_x, trans_y),
            scale=scale,
            shear=shear_degrees,
            fill=0
        )

        # Random horizontal flip
        if random.random() < 0.5:
            aug_img = F.hflip(aug_img)
            aug_mask = F.hflip(aug_mask)

        # Random vertical flip
        if random.random() < 0.5:
            aug_img = F.vflip(aug_img)
            aug_mask = F.vflip(aug_mask)

        # Convert back to numpy arrays (H, W, 3), float32
        aug_img_np = np.array(aug_img, dtype=np.float32)
        aug_mask_np = np.array(aug_mask, dtype=np.float32)

        return aug_img_np, aug_mask_np

    #===============================================================#
    #                                                               #
    #===============================================================#           
    def is_window_a_fei(self, window: np.ndarray, min_percentage_for_fei_window: float) -> bool:
        """
        Determines whether a given window contains at least the minimum percentage of pixels classified as FEI.

        :param window: A 3D window to check for FEI pixels.
        :param min_percentage_for_fei_window: The minimum percentage of pixels in the window that must be FEI.

        :return: True if the window contains at least the minimum percentage of FEI pixels, False otherwise.
        """

        window_pixels_count = np.shape(window)[0] * np.shape(window)[1]
        fei_pixels = (window[:, :, :] > 0.0).sum() / 3
        percentage = (fei_pixels * 100.0) / window_pixels_count
        if percentage >= min_percentage_for_fei_window:
            return True
        else:
            return False

    #===============================================================#
    #                                                               #
    #===============================================================# 
    def is_window_a_non_fei(self, window: np.ndarray, max_percentage_for_non_fei_window: float) -> bool:
        """
        Check if a window contains no frontal eye images (FEI) based on the count of non-FEI pixels.

        Args:
        - window: a 3D array representing a window of an image, where each element represents a pixel.
        - param max_percentage_for_non_fei_window: The maximum percentage of pixels in the window that must be NON FEI.

        Returns:
        - a boolean value indicating whether the window contains no FEI or not.
        """

        window_pixels_count = np.shape(window)[0] * np.shape(window)[1]
        fei_pixels = (window[:, :, :] > 0.0).sum() / 3
        percentage = (fei_pixels * 100.0) / window_pixels_count
        if percentage <= max_percentage_for_non_fei_window:
            return True
        else:
            return False
