import torch
from torch import nn, optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split

torch.manual_seed(101)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

main_folder = "checkbox_state_v2/data/"

transform = transforms.Compose([
	transforms.Resize((224, 224)),
	transforms.ToTensor()
])

dataset = datasets.ImageFolder(main_folder, transform=transform)
num_classes = len(dataset.classes)

train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_data, val_data, test_data = random_split(dataset, [train_size, val_size, test_size])

train_data_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_data_loader = DataLoader(val_data, batch_size=64, shuffle=True)
test_data_loader = DataLoader(test_data, batch_size=64, shuffle=False)

densenet = models.densenet121(weights=None)
densenet.classifier = nn.Linear(densenet.classifier.in_features, num_classes)
densenet.to(device)

loss_func = nn.CrossEntropyLoss().to(device)
optimizer = optim.SGD(densenet.parameters())

best_val_loss = float("inf")
num_epochs = 200

for epoch in range(num_epochs):

	densenet.train()
	running_train_loss = 0.0
	correct_train = 0
	total_train = 0

	for images, labels in train_data_loader:
		images, labels = images.to(device), labels.to(device)
		optimizer.zero_grad()
		outputs = densenet(images)
		loss = loss_func(outputs, labels)
		loss.backward()
		optimizer.step()

		_, predicted = torch.max(outputs.data, 1)
		total_train += labels.size(0)
		correct_train += (predicted == labels).sum().item()

		running_train_loss += loss.item()

	train_accuracy = correct_train / total_train

	densenet.eval()
	running_val_loss = 0.0
	correct_val = 0
	total_val = 0

	with torch.no_grad():
		for images, labels in val_data_loader:
			images, labels = images.to(device), labels.to(device)
			outputs = densenet(images)
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
		torch.save(densenet.state_dict(), "model_weights/baselines/no_pretrain/densenet.pth")
		print("..!Saving Weights!..")

print("Finished Training")
