import torch
from torch import nn, optim
from torch_optimizer import Lookahead
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split

torch.manual_seed(101)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

main_folder = "checkbox_state_v2/data/"

transform = transforms.Compose([
	transforms.Resize((299, 299)),
	transforms.AutoAugment(),
	transforms.ToTensor(),
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

efficientnet = models.efficientnet_b0(weights="DEFAULT")
efficientnet.classifier[1] = nn.Linear(efficientnet.classifier[1].in_features, num_classes)
efficientnet.to(device)

loss_func = nn.CrossEntropyLoss().to(device)
optimizer = Lookahead(optim.RMSprop(efficientnet.parameters(), lr=0.001))

best_val_loss = float("inf")
num_epochs = 100

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
		torch.save(efficientnet.state_dict(), "model_weights/checkpoints/checkpoint_1/efficientnet.pth")
		print("..!Saving Weights!..")

print("Finished Training")
