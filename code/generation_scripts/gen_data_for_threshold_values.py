import os
import random
import shutil
from sklearn.model_selection import train_test_split

# Define base directory and speaker names
base_base_dir = 'path/to/Thesis/activation_layer/data'
speakers = ['barack_obama']  # Add your speaker names here
threshold_values = [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240]

# Function to process each speaker and threshold
def process_speaker_threshold(speaker_name, threshold_value):
    speaker_dir = os.path.join(base_base_dir, f"{speaker_name}_threshold_images", f"thresh_{threshold_value}")
    deepfake_dir = os.path.join(speaker_dir, 'deepfake')
    original_dir = os.path.join(speaker_dir, 'original')

    # Train/test directories
    train_dir = os.path.join(speaker_dir, 'train')
    test_dir = os.path.join(speaker_dir, 'test')

    # Create train and test subfolders if not exist
    for folder in [train_dir, test_dir]:
        for category in ['deepfake', 'original']:
            os.makedirs(os.path.join(folder, category), exist_ok=True)

    # Helper function to copy files
    def copy_files(file_list, source_dir, target_dir):
        for file in file_list:
            shutil.copy(os.path.join(source_dir, file), os.path.join(target_dir, file))

    # Get list of all files in both classes
    deepfake_files = os.listdir(deepfake_dir)
    original_files = os.listdir(original_dir)

    # Balance the classes by limiting the larger class to the size of the smaller class
    min_class_size = min(len(deepfake_files), len(original_files))
    deepfake_files = random.sample(deepfake_files, min_class_size)
    original_files = random.sample(original_files, min_class_size)

    # Split the data into 80% train and 20% test
    deepfake_train, deepfake_test = train_test_split(deepfake_files, test_size=0.2, random_state=42)
    original_train, original_test = train_test_split(original_files, test_size=0.2, random_state=42)

    # Copy the images to the train and test directories
    copy_files(deepfake_train, deepfake_dir, os.path.join(train_dir, 'deepfake'))
    copy_files(deepfake_test, deepfake_dir, os.path.join(test_dir, 'deepfake'))
    copy_files(original_train, original_dir, os.path.join(train_dir, 'original'))
    copy_files(original_test, original_dir, os.path.join(test_dir, 'original'))

    print(f"Data split complete for {speaker_name} at {threshold_value}. Train and test sets created and balanced.")

# Process each speaker and threshold in the list
for speaker in speakers:
    for threshold in threshold_values:
        process_speaker_threshold(speaker, threshold)
