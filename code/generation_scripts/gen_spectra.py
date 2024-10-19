import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Function to plot and save spectrogram
def plot_spectrogram(audio_path, save_path, dpi=300):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=None)

    # Compute the Short-Time Fourier Transform (STFT)
    D = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))

    # Convert to dB (log scale)
    S_db = librosa.amplitude_to_db(D, ref=np.max)

    # Plot spectrogram
    plt.figure(figsize=(10, 5), dpi=dpi)
    librosa.display.specshow(S_db, sr=sr, hop_length=512, x_axis='time', y_axis='log', cmap='magma')

    # Save as RGB image
    plt.axis('off')  # Remove axis for saving as an image
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, format='png')
    plt.close()

# Function to process all wav files in a folder
def process_wav_files(source_folder, destination_folder, dpi=300):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Iterate through all files in the source folder
    for file_name in os.listdir(source_folder):
        if file_name.endswith(".wav"):
            audio_path = os.path.join(source_folder, file_name)
            save_path = os.path.join(destination_folder, file_name.replace(".wav", ".png"))
            
            # Plot and save spectrogram for each wav file
            plot_spectrogram(audio_path, save_path, dpi=dpi)
            print(f"Saved spectrogram for {file_name} at {save_path}")

# Set source and destination folders
#source_folder = 'path/to/Data/Barack_Obama/Original'
#destination_folder = 'path/to/data/Barack_Obama/deepfake_img'
source_folder = 'path/to/Data/Barack_Obama/deepfake'
destination_folder = 'path/to/data/Barack_Obama/original_img'


#source_folder = 'path/to/Data/Domenico_Grasso/ElevenLabs'
#destination_folder = 'path/to/data/Domenico_Grasso/deepfake_img'
#source_folder = 'path/to/Data/Domenico_Grasso/Original'
#destination_folder = 'path/to/data/Domenico_Grasso/original_img'


#source_folder = 'path/to/data/Donald_Trump/ElevenLabs'
#destination_folder = 'path/to/data/Donald_Trump/deepfake_img'
#source_folder = 'path/to/data/Donald_Trump/Original'
#destination_folder = 'path/to/data/Donald_Trump/original_img'



#source_folder = 'path/to/data/Joe_Biden/ElevenLabs'
#destination_folder = 'path/to/data/Joe_Biden/deepfake_img'

#source_folder = 'path/to/data/Joe_Biden/Original'
#destination_folder = 'path/to/data/Joe_Biden/original_img'



# Process all WAV files in the source folder with the specified dpi
process_wav_files(source_folder, destination_folder, dpi=100)
