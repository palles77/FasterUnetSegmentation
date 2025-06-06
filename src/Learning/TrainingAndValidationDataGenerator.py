import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset

class TrainingAndValidationDataGenerator(Dataset):
    def __init__(self, cml_args, is_training: bool):
        """
        cml_args should have the following attributes:
        - generate_dir: directory with the npz files
        - windows_per_image_on_average
        - augmentations_count
        - generation_images_per_npz
        - batch_size
        """
        self.generate_dir = cml_args.generate_dir
        
        # Gather all files in the directory
        all_files = os.listdir(self.generate_dir)
        
        # Depending on is_training, pick the right files
        if is_training:
            self.x_files = [
                f for f in all_files 
                if os.path.isfile(os.path.join(self.generate_dir, f)) 
                and f.startswith("training_ready_combined_") 
                and f.endswith("_x.npz")
            ]
        else:
            self.x_files = [
                f for f in all_files 
                if os.path.isfile(os.path.join(self.generate_dir, f)) 
                and f.startswith("validation_ready_combined_") 
                and f.endswith("_x.npz")
            ]

        # Calculate how many windows are in each npz file after augmentation
        self.augmented_windows_per_npz_file = int(
            cml_args.windows_per_image_on_average * (cml_args.augmentations_count + 1)
        ) * cml_args.generation_images_per_npz
        
        # Number of batches per npz, given the batch_size
        self.batch_size = int(cml_args.batch_size)
        self.batches_per_npz_file = self.augmented_windows_per_npz_file // self.batch_size
        
        # Total number of batches across all npz files
        self.len = self.batches_per_npz_file * len(self.x_files)
        
        # Build file pairs (x_file, y_file)
        self.file_pairs = []
        for x_file_name_short in self.x_files:
            y_file_name_short = x_file_name_short.replace("_x", "_y")
            x_file_name_long = os.path.join(self.generate_dir, x_file_name_short)
            y_file_name_long = os.path.join(self.generate_dir, y_file_name_short)
            if os.path.exists(x_file_name_long) and os.path.exists(y_file_name_long):
                self.file_pairs.append((x_file_name_short, y_file_name_short))
        
        # Shuffle for the first epoch if desired
        self.on_epoch_end()

    def __len__(self):
        """
        Returns the total number of batches (not individual samples).
        """
        return self.len

    def __getitem__(self, index):
        # 1) Figure out which npz file, load X and Y from disk
        file_index = index // self.batches_per_npz_file
        file_x = os.path.join(self.generate_dir, self.file_pairs[file_index][0])
        file_y = os.path.join(self.generate_dir, self.file_pairs[file_index][1])
        
        data_x = np.load(file_x)["arr_0"]  # shape: (N, H, W, C)
        data_y = np.load(file_y)["arr_0"]  # shape: (N, H, W, <number_of_classes>)
        
        # 2) Slice out the batch
        batch_index_per_file = index % self.batches_per_npz_file
        start = batch_index_per_file * self.batch_size
        end = (batch_index_per_file + 1) * self.batch_size
        
        x_batch_np = data_x[start:end]  # still shape (batch_size, H, W, C)
        y_batch_np = data_y[start:end]  # shape (batch_size, H, W, num_classes)

        # 3) Convert to PyTorch tensors
        x_batch = torch.from_numpy(x_batch_np).float()
        y_batch = torch.from_numpy(y_batch_np).float()

        # 4) Permute the X batch from (N, H, W, C) -> (N, C, H, W)
        x_batch = x_batch.permute(0, 3, 1, 2)
        y_batch = y_batch.permute(0, 3, 1, 2)

        #   If y is one-hot, you might keep it as (N, H, W, 2) and permute to (N, 2, H, W),
        #   depending on how your loss function expects the shape. 
        #   For cross entropy in PyTorch, you usually want y of shape (N, H, W) 
        #   with integer class labels (0 or 1). 
        #   If you are doing a "2-channel" output + softmax, the typical approach is:
        #       y_batch = torch.argmax(y_batch, dim=-1)  # now shape (N, H, W), integer labels
        #
        #   Or if your code already uses a 'channels' dimension for y, do:
        #       y_batch = y_batch.permute(0, 3, 1, 2)

        # Return the batch
        return x_batch, y_batch

    def on_epoch_end(self):
        """
        Shuffle the file pairs at the end of each epoch, 
        mirroring the Keras `on_epoch_end`. 
        Note: In PyTorch, you typically rely on the DataLoader's shuffle argument 
        rather than shuffling inside the dataset itself. However, if you want 
        the exact Keras-style approach, you can call this manually at the end of each epoch.
        """
        np.random.shuffle(self.file_pairs)
