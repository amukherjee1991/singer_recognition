import os
import librosa
import numpy as np
import csv

def save_audio_as_csv(dataset_path, save_path):
    """
    Go through the dataset of audio files and save the raw data points into CSV files.
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    labels = os.listdir(dataset_path)

    for label in labels:
        print(f"Processing label: {label}")
        song_dir = os.path.join(dataset_path, label)
        songs = os.listdir(song_dir)

        save_label_dir = os.path.join(save_path, label)
        if not os.path.exists(save_label_dir):
            os.makedirs(save_label_dir)

        for song_name in songs:
            file_path = os.path.join(song_dir, song_name)

            # Load the audio file
            data, _ = librosa.load(file_path, sr=None)  # data is a numpy array

            # Define the name of the new CSV file
            csv_file_path = os.path.join(save_label_dir, f"{os.path.splitext(song_name)[0]}.csv")

            # Save the numpy array to CSV
            with open(csv_file_path, mode='w', newline='') as audio_file:
                audio_writer = csv.writer(audio_file)
                for item in data:
                    # The writer expects a list; here, each list has only one element.
                    audio_writer.writerow([item])

        print(f"Label {label} processing completed!")

if __name__ == "__main__":
    dataset_path = "songs"  # path to the folder containing songs
    save_path = "audio_as_csv"  # path where the CSV files will be saved
    save_audio_as_csv(dataset_path, save_path)
