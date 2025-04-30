import os

# Directories (update these paths if needed)
good_dir_0 = r"datasets\sabari50312\fundus-pytorch\versions\1\val\0"
good_dir_1 = r"datasets\sabari50312\fundus-pytorch\versions\1\val\1"
degraded_dir = r"val_degraded_images"

# Get the list of files from the good and degraded directories
good_files_0 = {file.split('.')[0]: file for file in os.listdir(good_dir_0) if file.endswith('.png')}
good_files_1 = {file.split('.')[0]: file for file in os.listdir(good_dir_1) if file.endswith('.png')}
degraded_files = {file.split('_', 1)[-1]: file for file in os.listdir(degraded_dir) if file.startswith('0_') and file.endswith('.png')}

# Combined good files (0 and 1)
good_files = {**good_files_0, **good_files_1}

# Now, clean the directories based on the pairings
def clean_directory(dir_path, valid_filenames, prefix=''):
    for filename in os.listdir(dir_path):
        file_name_without_extension = filename.split('.')[0]
        
        # For degraded images, make sure to match by '0_' prefix
        if prefix:
            corresponding_degraded = f"0_{file_name_without_extension}.png"
            if corresponding_degraded not in degraded_files:
                # Delete non-matching degraded files
                file_path = os.path.join(dir_path, filename)
                print(f"Would delete {file_path}")  # For testing; replace with os.remove(file_path) to delete
        else:
            if file_name_without_extension not in valid_filenames:
                # Delete non-matching good files
                file_path = os.path.join(dir_path, filename)
                print(f"Would delete {file_path}")  # For testing; replace with os.remove(file_path) to delete

# Clean good directories (0 and 1)
clean_directory(good_dir_0, good_files, prefix='')
clean_directory(good_dir_1, good_files, prefix='')

# Clean the degraded directory (should match with '0_' prefix)
clean_directory(degraded_dir, good_files, prefix='0_')
