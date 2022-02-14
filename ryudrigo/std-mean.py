import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
import torch

nimages = 0
mean = 0.0
var = 0.0
dataset= FashionMNIST (root='data/fashionmnist', download=True, transform=T.Compose([T.Grayscale(3),T.ToTensor()]))
data_loader = DataLoader(dataset)
for i_batch, batch_target in enumerate(data_loader):
    batch = batch_target[0]
    # Rearrange batch to be the shape of [B, C, W * H]
    batch = batch.view(batch.size(0), batch.size(1), -1)
    # Update total number of images
    nimages += batch.size(0)
    # Compute mean and std here
    mean += batch.mean(2).sum(0) 
    var += batch.var(2).sum(0)

mean /= nimages
var /= nimages
std = torch.sqrt(var)

print(mean)
print(std)