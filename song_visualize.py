import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
def visualize_audio_files(dataset_path, save_path):
    """
    Go through the dataset of audio files and create visualizations.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # List all labels (subdirectories) in the dataset path
    labels = os.listdir(dataset_path)

    for label in labels:
        print(f"Processing label: {label}")
        song_dir = os.path.join(dataset_path, label)
        songs = os.listdir(song_dir)

        visualization_label_dir = os.path.join(save_path, label)
        if not os.path.exists(visualization_label_dir):
            os.makedirs(visualization_label_dir)

        for song_name in songs:
            file_path = os.path.join(song_dir, song_name)

            # Load the audio file
            data, sr = librosa.load(file_path, sr=None)  # data is numpy array, sr is sample rate

            # Create a mel-scaled spectrogram
            spectrogram = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=128, fmax=8000)

            # And plot it
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max),
                                     y_axis='mel', fmax=8000, x_axis='time')
            plt.colorbar(format='%+2.0f dB')
            plt.title(f'Mel Spectrogram - {song_name}')
            plt.tight_layout()
            plt.savefig(os.path.join(visualization_label_dir, f"{os.path.splitext(song_name)[0]}.png"))
            plt.close()

        print(f"Label {label} processing completed!")

if __name__ == "__main__":
    dataset_path = "songs"  # Replace with the path to your directory containing the songs
    save_path = "song_visualizations"  # Path where the visualizations will be saved
    visualize_audio_files(dataset_path, save_path)
