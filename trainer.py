import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pickle

from model import SingerClassifier  # Import your actual model class from model.py

# Define your custom Dataset class.
class SongsDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = []
        self.labels = []

        # Walk through the root directory and list all the paths of '.pkl' files.
        for subdir, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(".pkl"):
                    self.files.append(os.path.join(subdir, file))
                    # The label is extracted as the folder name.
                    label = int(os.path.basename(subdir))
                    self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Load the .pkl file.
        pkl_file_path = self.files[idx]
        
        try:
            with open(pkl_file_path, 'rb') as file:
                data = pickle.load(file)
        except (pickle.UnpicklingError, FileNotFoundError) as e:
            print(f"Error loading {pkl_file_path}: {e}")
            return None, None

        # Get the label associated with the song.
        label = self.labels[idx]

        # Convert data and label to torch tensors.
        if data is not None:
            song_data = torch.from_numpy(data).float()  # Ensure data is a float tensor
            label = torch.tensor(label, dtype=torch.long)  # Ensure label is an integer.
            return song_data, label
        else:
            return None, None

def train_with_padding_and_trim(model, data_loader, criterion, optimizer, num_epochs):
    # Training loop
    for epoch in range(num_epochs):
        for batch_idx, (data, target) in enumerate(data_loader):
            if data is None:
                continue  # Skip batches with loading errors

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx+1}/{len(data_loader)}, Loss: {loss.item()}")

def main():
    # Preprocess the dataset
    audio_features_directory = 'audio_features'  # Replace with your actual directory

    # Set device for torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create an instance of your model
    model = SingerClassifier().to(device)

    # Create instance of the dataset
    dataset = SongsDataset(root_dir=audio_features_directory)

    # Remove any None entries from the dataset (due to loading errors)
    dataset = [data for data in dataset if data[0] is not None]

    # Create DataLoader
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())

    # Training loop
    num_epochs = 10
    train_with_padding_and_trim(model, data_loader, criterion, optimizer, num_epochs)

if __name__ == "__main__":
    main()
