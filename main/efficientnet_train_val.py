import torch
from torch import nn, optim
from torch_optimizer import Lookahead
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split

# Setting torch global manual seed for consistancy and replication as well GPU or CPU as per availability
torch.manual_seed(101)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

main_folder = "checkbox_state_v2/data/"

# using torch transformations to transform and augment the dataset
# these are the final transforms used in the main model with best results
transform = transforms.Compose([
	transforms.Resize((299, 299)),
	transforms.AutoAugment(),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# these are the previous transforms used in the checkpoint models for experimentation
"""
transform = transforms.Compose([
	transforms.Resize((299, 299)),
	transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)), 
	transforms.RandomHorizontalFlip(p=0.5),
	transforms.RandomVerticalFlip(p=0.5),
	transforms.RandomAffine(degrees=(-15, 15), translate=(0.1, 0.1), scale=(0.9, 1.1), shear=(-15, 15)),
	transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
	transforms.ColorJitter(brightness=(0.9, 1.1), contrast=(0.5, 1.5), saturation=(0.5, 1.5), hue=(-0.05, 0.05)),
	transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.5),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.6688, 0.6875, 0.7220], std=[0.3945, 0.3769, 0.3534])
])
"""

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

# Load the efficient net baseline model with weights pre trained on imagenet data
efficientnet = models.efficientnet_b0(weights="DEFAULT")
efficientnet.classifier[1] = nn.Linear(efficientnet.classifier[1].in_features, num_classes)
efficientnet.to(device)

# Calculating weighted frequencies based on the differnet class samples for class imablance for loss funtion along with different hyper parameters for experimentaiton
"""
class_frequencies = [222, 138, 155]
class_weights = [sum(class_frequencies) / (len(class_frequencies) * freq) for freq in class_frequencies]
class_weights_tensor = torch.tensor(class_weights)

loss_func = nn.CrossEntropyLoss(weight=class_weights_tensor).to(device)
optimizer = Lookahead(optim.RMSprop(efficientnet.parameters(), lr=0.001, alpha=0.9, momentum=0.9, weight_decay=1e-5), alpha=0.5, k=5)
"""

# Initializing the optimizer and loss function
loss_func = nn.CrossEntropyLoss().to(device)
optimizer = Lookahead(optim.RMSprop(efficientnet.parameters(), lr=0.001))

best_val_loss = float("inf")
num_epochs = 100

# Main Training and Validation Loop, Saves the best model weights based on the lowest validation loss
for epoch in range(num_epochs):

	efficientnet.train()
	running_train_loss = 0.0
	correct_train = 0
	total_train = 0

	for images, labels in train_data_loader:
		images, labels = images.to(device), labels.to(device)
		optimizer.zero_grad()
		outputs = efficientnet(images)
		loss = loss_func(outputs, labels)
		loss.backward()
		optimizer.step()

		_, predicted = torch.max(outputs.data, 1)
		total_train += labels.size(0)
		correct_train += (predicted == labels).sum().item()

		running_train_loss += loss.item()

	train_accuracy = correct_train / total_train

	efficientnet.eval()
	running_val_loss = 0.0
	correct_val = 0
	total_val = 0

	with torch.no_grad():
		for images, labels in val_data_loader:
			images, labels = images.to(device), labels.to(device)
			outputs = efficientnet(images)
			loss = loss_func(outputs, labels)

			_, predicted = torch.max(outputs.data, 1)
			total_val += labels.size(0)
			correct_val += (predicted == labels).sum().item()

			running_val_loss += loss.item()

	val_accuracy = correct_val / total_val

	print(f"Epoch {epoch+1}/{num_epochs}, "
		  f"Train Loss: {running_train_loss:.4f}, "
		  f"Train Accuracy: {train_accuracy:.4f}, "
		  f"Val Loss: {running_val_loss:.4f}, "
		  f"Val Accuracy: {val_accuracy:.4f}")

	if running_val_loss < best_val_loss:
		best_val_loss = running_val_loss
		torch.save(efficientnet.state_dict(), "model_weights/main/efficientnet.pth")
		print("..!Saving Weights!..")

print("Finished Training")
