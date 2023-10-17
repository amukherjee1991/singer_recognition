import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import model  # Assuming model.py is in the same directory

def find_min_length(directory):
    min_length = float('inf')
    for subdir, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.npy'):  # Making sure we only read '.npy' files
                file_path = os.path.join(subdir, file)
                data = np.load(file_path)
                min_length = min(min_length, data.shape[1])
    return min_length

class SongsDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.songs = []
        self.labels = []
        for label_dir in os.listdir(root_dir):
            song_files = os.listdir(os.path.join(root_dir, label_dir))
            for song_file in song_files:
                self.songs.append(os.path.join(root_dir, label_dir, song_file))
                self.labels.append(int(label_dir))

        self.min_length = find_min_length(root_dir)

    def __len__(self):
        return len(self.songs)

    def __getitem__(self, idx):
        song_path = self.songs[idx]
        label = self.labels[idx]
        song_data = np.load(song_path)

        # Truncate or pad the sequence based on the minimum length
        if song_data.shape[1] > self.min_length:
            song_data = song_data[:, :self.min_length]
        elif song_data.shape[1] < self.min_length:
            padding = np.zeros((song_data.shape[0], self.min_length - song_data.shape[1]))
            song_data = np.concatenate((song_data, padding), axis=1)

        song_data = torch.from_numpy(song_data).float()
        return song_data, label

def train_model(model, criterion, optimizer, train_loader, num_epochs=25):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Ensure our device is the desired one for training
    
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0

        # Training step
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)  # Move input and label tensors to the default device for training

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

    print('Finished Training')


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SongsDataset(root_dir='extracted_features')  # Specify your root directory
    train_loader = DataLoader(dataset, batch_size=5, shuffle=True)

    model_instance = model.SingerClassifier()
    model_instance.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model_instance.parameters(), lr=0.001)

    train_model(model_instance, criterion, optimizer, train_loader, num_epochs=25)

if __name__ == "__main__":
    main()