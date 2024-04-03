import torch
from torch import nn, optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from matplotlib import pyplot as plt

# Setting torch global manual seed for consistancy and replication as well GPU or CPU as per availability
torch.manual_seed(101)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

main_folder = "checkbox_state_v2/data/"

# using torch transformations to transform with no augmentations and only normalizing
transform = transforms.Compose([
	transforms.Resize((299, 299)),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.6688, 0.6875, 0.7220], std=[0.3945, 0.3769, 0.3534])
])

# load the dataset using data loader
# randomply split and shuffle the dataset into train validation and test data with 80 10 10 ratio
dataset = datasets.ImageFolder(main_folder, transform=transform)
num_classes = len(dataset.classes)

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

train_data_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_data_loader = DataLoader(val_data, batch_size=64, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=64, shuffle=False)

# Load the efficient net baseline model with no weights and load the newly trained weights
efficientnet = models.efficientnet_b0(weights=None)
efficientnet.classifier[1] = nn.Linear(efficientnet.classifier[1].in_features, num_classes)

efficientnet.load_state_dict(torch.load("model_weights/main/efficientnet.pth"))
efficientnet.to(device)
efficientnet.eval()

correct_test = 0
total_test = 0

# Loop to predict and visualize the training data
with torch.no_grad():
    for images, labels in test_data_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = efficientnet(images)
        _, predicted = torch.max(outputs, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

        fig, axes = plt.subplots(13, 4, figsize=(14, 14))
        for i, ax in enumerate(axes.flat):
            image = images[i].permute(1, 2, 0).cpu().numpy()
            ax.imshow(image)
            if predicted[i] == labels[i]:
                title_color = "green"
            else:
                title_color = "red"
            ax.set_title(f"True: {labels[i]}, Predicted: {predicted[i]}", color=title_color)
            ax.axis("off")
        plt.tight_layout()
        plt.show()

test_accuracy = correct_test / total_test
print(f"Test Accuracy: {test_accuracy:.4f}")
