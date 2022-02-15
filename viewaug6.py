import torch
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from src.utils import utils
from dotmap import DotMap
from src.datasets import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image
from ryudrigo.ranking_and_backbone import Backbone
from ryudrigo.custom_datasets import FFHQThumbDataset
from ryudrigo.ranking_and_backbone import Ranking

model = Ranking()

#dataset= FashionMNIST (root='data/fashionmnist', download=True, transform=T.Compose([T.Grayscale(3),T.ToTensor()]))
dataset = FFHQThumbDataset('data/ffhq-thumb')
data_loader = DataLoader(dataset)

transform = T.ToPILImage()

minibatch = []
for i, batch in enumerate(iter(data_loader)):
    minibatch.append(batch[0])
    if i>3:
        break
        
minibatch = torch.cat(minibatch)
x, y, imgs1, imgs2 = model.generate_backbone_activations(minibatch)
print ('x-shape')
print (x.shape)

print ('y-shape')
print (y.shape)

print('y-hat')
print (model(x))
print ('y-true')
print (y)

for i, (img1, img2) in enumerate(zip (imgs1, imgs2)):
    img1 = transform(img1)
    img2 = transform(img2)
    img1.save('ryudrigo/temp2-ffhq/'+ str(i)+'-img1'+ '.jpg')
    img2.save('ryudrigo/temp2-ffhq/'+ str(i)+'-img2'+ '.jpg')
