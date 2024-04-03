import torch
import statistics
import numpy as np
from torch import nn, optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from matplotlib import pyplot as plt

# Setting torch global manual seed for consistancy and replication as well GPU or CPU as per availability
torch.manual_seed(101)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

main_folder = "checkbox_state_v2/inference/"

# using torch transformations to transform with no augmentations and only normalizing
transform = transforms.Compose([
	transforms.Resize((299, 299)),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.6688, 0.6875, 0.7220], std=[0.3945, 0.3769, 0.3534])
])

# load the dataset using data loader
dataset = datasets.ImageFolder(main_folder, transform=transform)
full_data_loader = DataLoader(dataset, batch_size=64, shuffle=False)
num_classes = 3

# Load the efficient net baseline model with no weights and load the newly trained weights
efficientnet = models.efficientnet_b0(weights=None)
efficientnet.classifier[1] = nn.Linear(efficientnet.classifier[1].in_features, num_classes)

efficientnet.load_state_dict(torch.load("model_weights/main/efficientnet.pth"))
efficientnet.to(device)
efficientnet.eval()

# Loop to predict the training data with a small trick to make predictions 3 times and print the output of the voting.

num_predictions = 3
final_predictions = []

with torch.no_grad():
    for images, _ in full_data_loader:
        images = images.to(device)

        predictions = []
        for _ in range(num_predictions):
            outputs = efficientnet(images)
            _, predicted = torch.max(outputs, 1)
            predictions.append(predicted.cpu().numpy())

        predictions_array = np.array(predictions)
        voting_output = [statistics.mode(predictions) for predictions in zip(*predictions_array)]
        final_predictions.append(voting_output)

for idx, predicted_classes in enumerate(final_predictions[0]):
    if predicted_classes == 0:
        print(f"Image {idx + 1} Predicted Class: Checked")
    elif predicted_classes == 1:
        print(f"Image {idx + 1} Predicted Class: Other")
    elif predicted_classes == 2:
        print(f"Image {idx + 1} Predicted Class: UnChecked")
