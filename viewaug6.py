from src.systems.image_systems import PretrainViewMakerSystem
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

for i, batch in enumerate(iter(data_loader)):
    if i>2:
        break
    img = batch[0]
    img1, img2, vis1, vis2 = model(img)
    
    img1 = transform(img1[0])
    img2 = transform(img2[0])
    
    img1.save('ryudrigo/temp2-ffhq/'+ str(i)+'-img1'+ '.jpg')
    img2.save('ryudrigo/temp2-ffhq/'+ str(i)+'-img2'+ '.jpg')
   
    for p, vis_images in enumerate(vis1[:-2]):
            for k, vis_img in enumerate(vis_images[0]):
                vis_img = transform(vis_img)
                vis_img.save('ryudrigo/temp2-ffhq/{}-img1-vis1-{}-{}.jpg'.format(i, p, k))
