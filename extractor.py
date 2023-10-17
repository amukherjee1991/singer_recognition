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

    # Create a separate directory for visualizations
    visualizations_path = os.path.join(os.path.dirname(save_path), "visualizations")
    if not os.path.exists(visualizations_path):
        os.makedirs(visualizations_path)

    labels = os.listdir(dataset_path)

    for label in labels:
        print(f"Processing label: {label}")
        song_dir = os.path.join(dataset_path, label)
        songs = os.listdir(song_dir)

        save_dir = os.path.join(save_path, label)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Adjusting the directory path for visualizations for each label
        visualization_label_dir = os.path.join(visualizations_path, label)
        if not os.path.exists(visualization_label_dir):
            os.makedirs(visualization_label_dir)

        for song_name in songs:
            # Load the audio file
            file_path = os.path.join(song_dir, song_name)
            data, sr = librosa.load(file_path, sr=None)

            # Extract MFCC features or any other features you prefer
            mfccs = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=13)

            # Save the extracted features
            save_feature_path = os.path.join(save_dir, f"{os.path.splitext(song_name)[0]}.npy")
            np.save(save_feature_path, mfccs)

            # Create and save the visualization
            plt.figure(figsize=(10, 4))
            librosa.display.specshow(librosa.power_to_db(mfccs, ref=np.max), y_axis='mel', x_axis='time')
            plt.colorbar(format='%+2.0f dB')
            plt.title('MFCC')
            plt.savefig(os.path.join(visualization_label_dir, f"{os.path.splitext(song_name)[0]}.png"))
            plt.close()

        print(f"Label {label} processing completed!")

if __name__ == "__main__":
    dataset_path = "songs"  # replace with your actual path
    save_path = "extracted_features"  # replace with your actual path
    preprocess_and_save(dataset_path, save_path)
