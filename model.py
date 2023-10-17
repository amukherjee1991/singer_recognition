# model.py
import torch.nn as nn
import torch.nn.functional as F

class SingerClassifier(nn.Module):
    def __init__(self):
        super(SingerClassifier, self).__init__()
        # Adjusting for 1D convolution here
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5)  # assuming your input data is 1-dimensional
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5)
        self.dropout = nn.Dropout(0.5)

        # You will need to adjust the input features for fc1 depending on your actual data's size after convolution and pooling layers
        self.fc1 = nn.Linear(64 * 5 * 5, 1000)  # This needs to be adjusted based on the output from the last convolution/pooling layer
        self.fc2 = nn.Linear(1000, 5)  # assuming 5 different singers

    def forward(self, x):
        # Note: If you're working with raw audio, the input x here should be a 1D tensor representing your audio data

        # Adjusting for 1D operations
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, 2)

        # Flattening the tensor
        x = x.view(x.size(0), -1)  # you might need to adjust the shape based on your actual data structure

        x = self.dropout(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

# This part is for testing purposes, to verify the model structure
if __name__ == "__main__":
    # Create an instance of the model and print it
    model = SingerClassifier()
    print(model)
