import os
import random
import shutil
from sklearn.model_selection import train_test_split

# Define base directory and speaker names
base_dir = 'path/to/Thesis/activation_layer/data'
speakers = ['Barack_Obama', 'Domenico_Grasso', 'Donald_Trump']  # Add your speaker names here

# Function to process each speaker
def process_speaker(speaker_name):
    speaker_dir = os.path.join(base_dir, speaker_name)
    deepfake_dir = os.path.join(speaker_dir, 'deepfake_img')
    original_dir = os.path.join(speaker_dir, 'original_img')

    # Train/test directories
    train_dir = os.path.join(speaker_dir, 'train')
    test_dir = os.path.join(speaker_dir, 'test')

    # Create train and test subfolders if not exist
    for folder in [train_dir, test_dir]:
        for category in ['deepfake_img', 'original_img']:
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
    copy_files(deepfake_train, deepfake_dir, os.path.join(train_dir, 'deepfake_img'))
    copy_files(deepfake_test, deepfake_dir, os.path.join(test_dir, 'deepfake_img'))
    copy_files(original_train, original_dir, os.path.join(train_dir, 'original_img'))
    copy_files(original_test, original_dir, os.path.join(test_dir, 'original_img'))

    print(f"Data split complete for {speaker_name}. Train and test sets created and balanced.")

# Process each speaker in the list
for speaker in speakers:
    process_speaker(speaker)
