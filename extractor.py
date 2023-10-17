import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

def preprocess_and_save(dataset_path, save_path):
    """
    Go through the dataset of audio files, extract features, and save them as .npy files.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    labels = os.listdir(dataset_path)

    for label in labels:
        print(f"Processing label: {label}")
        song_dir = os.path.join(dataset_path, label)
        songs = os.listdir(song_dir)

        save_dir = os.path.join(save_path, label)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        visualization_dir = os.path.join(save_path, "visualizations", label)
        if not os.path.exists(visualization_dir):
            os.makedirs(visualization_dir)

        for song_name in songs:
            # Load the audio file
            file_path = os.path.join(song_dir, song_name)
            data, sr = librosa.load(file_path, sr=None)  # data is numpy array, sr is sample rate

            # Extract audio features (you can replace this with your feature extraction method)
            # Example: Mel-frequency cepstral coefficients (MFCCs)
            mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13)

            # Save the extracted features as a .npy file
            save_feature_path = os.path.join(save_dir, f"{os.path.splitext(song_name)[0]}.npy")
            np.save(save_feature_path, mfccs)

            # Create a visualization of the features and save it as an image
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(librosa.power_to_db(mfccs, ref=np.max), y_axis='mel', x_axis='time')
            plt.colorbar(format='%+2.0f dB')
            plt.title('MFCC')
            plt.savefig(os.path.join(visualization_dir, f"{os.path.splitext(song_name)[0]}.png"))
            plt.close()

        print(f"Label {label} processing completed!")

if __name__ == "__main__":
    dataset_path = "songs"  # path to the folder containing songs
    save_path = "extracted_features"  # path where the features and visualizations will be saved
    preprocess_and_save(dataset_path, save_path)
