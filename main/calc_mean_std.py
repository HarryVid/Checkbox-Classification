import torch
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split

torch.manual_seed(101)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

main_folder = "checkbox_state_v2/data/"

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder(main_folder, transform=transform)
full_data_loader = DataLoader(dataset, batch_size=64, shuffle=False)

sum_channel = torch.zeros(3)
sum_squared_channel = torch.zeros(3)
total_samples = 0

for images, _ in full_data_loader:
    sum_channel += torch.sum(images, dim=(0, 2, 3))
    sum_squared_channel += torch.sum(images ** 2, dim=(0, 2, 3))

    total_samples += images.size(0)

mean = sum_channel / (total_samples * images.size(2) * images.size(3))
variance = (sum_squared_channel / (total_samples * images.size(2) * images.size(3))) - (mean ** 2)
std = torch.sqrt(variance)

print("Mean:", mean)
print("Standard Deviation:", std)
