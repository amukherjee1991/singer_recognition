import torch
import torch.nn as nn
import torch.nn.functional as F

class SingerClassifier(nn.Module):
    def __init__(self):
        super(SingerClassifier, self).__init__()
        self.conv1 = nn.Conv1d(13, 32, kernel_size=5)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5)
        self.dropout = nn.Dropout(0.5)
        
        # Dummy forward pass to calculate the size
        x = torch.randn(1, 13, 222389)  # Random 'dummy' input
        x = self.conv1(x)
        x = F.max_pool1d(x, 2)
        x = self.conv2(x)
        x = F.max_pool1d(x, 2)
        self.flattened_size = x.numel()  # Number of elements in x

        # Now that we know the size, we can create the fully connected layers
        self.fc1 = nn.Linear(self.flattened_size, 1000)
        self.fc2 = nn.Linear(1000, 5)  # Adjust 5 to match the number of your target classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool1d(x, 2)
        x = x.view(-1, self.flattened_size)  # Flatten the output for the fully connected layers
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# This part is for testing purposes, to verify the model structure
if __name__ == "__main__":
    # Create an instance of the model and print it
    model = SingerClassifier()
    print(model)
