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

config_json = utils.load_json("ryudrigo/config.json")
config = DotMap(config_json)
checkpoint = torch.load("lightning_logs/version_58/checkpoints/epoch=199.ckpt")
model = Backbone(config)
model.load_state_dict(checkpoint['state_dict'], strict=False)

#dataset= FashionMNIST (root='data/fashionmnist', download=True, transform=T.Compose([T.Grayscale(3),T.ToTensor()]))
dataset = FFHQThumbDataset('data/ffhq-thumb')
data_loader = DataLoader(dataset)

transform = T.ToPILImage()

for i, batch in enumerate(iter(data_loader)):
    img = batch[0]
    y_hat = model.forward(img)
    pred = torch.topk(y_hat, 1)
    pred=pred.indices[0]
    if i>99:
        break
    img = transform(img[0])
    img.save('ryudrigo/examples-ffhq/'+ str(i)+'-'+str(pred)+ '.jpg')
