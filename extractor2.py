import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from spleeter.separator import Separator  # Importing Spleeter

def preprocess_and_save(dataset_path, save_path):
    """
    Go through the dataset of audio files, extract features, and save them as .npy files.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    visualizations_path = os.path.join(os.path.dirname(save_path), "visualizations")
    if not os.path.exists(visualizations_path):
        os.makedirs(visualizations_path)

    # Initialize the Spleeter separator with the desired configuration
    separator = Separator('spleeter:2stems')  # 2stems model separates vocals and accompaniment

    labels = os.listdir(dataset_path)

    for label in labels:
        print(f"Processing label: {label}")
        song_dir = os.path.join(dataset_path, label)
        songs = os.listdir(song_dir)

        save_dir = os.path.join(save_path, label)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        visualization_label_dir = os.path.join(visualizations_path, label)
        if not os.path.exists(visualization_label_dir):
            os.makedirs(visualization_label_dir)

        for song_name in songs:
            file_path = os.path.join(song_dir, song_name)

            # Use Spleeter to separate the vocals and accompaniment
            # This function creates a folder with the separated audio files
            separator.separate_to_file(file_path, os.path.join(save_dir, os.path.splitext(song_name)[0]))

            # Assuming you want to process the vocals, let's load the separated vocal track
            # Note: Spleeter saves the output as ogg files (or wav if specified during the separation)
            vocal_track_path = os.path.join(save_dir, os.path.splitext(song_name)[0], 'vocals.wav')  # or vocals.ogg
            data, sr = librosa.load(vocal_track_path, sr=None)

            # Now you can extract features from the vocal track
            mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13)

            # Save the extracted features
            save_feature_path = os.path.join(save_dir, f"{os.path.splitext(song_name)[0]}_vocals.npy")
            np.save(save_feature_path, mfccs)

            # Create and save the visualization
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(librosa.power_to_db(mfccs, ref=np.max), y_axis='mel', x_axis='time')
            plt.colorbar(format='%+2.0f dB')
            plt.title('MFCC')
            plt.tight_layout()
            plt.savefig(os.path.join(visualization_label_dir, f"{os.path.splitext(song_name)[0]}_vocals.png"))
            plt.close()

        print(f"Label {label} processing completed!")

if __name__ == "__main__":
    dataset_path = "songs"  # replace with your actual path
    save_path = "extracted_features2"  # replace with your actual path
    preprocess_and_save(dataset_path, save_path)
