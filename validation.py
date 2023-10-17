# validation.py
import torch
from model import SingerClassifier
from data_loader import get_data_loaders

def validate_model():
    _, test_loader = get_data_loaders()  # get the test loader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SingerClassifier().to(device)
    model.load_state_dict(torch.load('singer_classifier.pth'))
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Accuracy: {(100 * correct / total)}%")

if __name__ == "__main__":
    validate_model()
