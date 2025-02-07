from src.systems.image_systems import PretrainViewMakerSystem
import torch
from src.utils import utils
from dotmap import DotMap
from src.datasets import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image

config_json = utils.load_json("pretrain_viewmaker_cifar_simclr_resnet18/config.json")
config = DotMap(config_json)
checkpoint = torch.load("pretrain_viewmaker_cifar_simclr_resnet18/checkpoints/epoch=50.ckpt")

model = PretrainViewMakerSystem(config)
model.load_state_dict(checkpoint['state_dict'], strict=False)

_, dataset = datasets.get_image_datasets( 'meta_fashionmnist',False)
data_loader = DataLoader(dataset)

transform = T.ToPILImage()

for i, batch in enumerate(iter(data_loader)):
    img = batch[1]
    img = model.viewmaker_example(img)
    if i>999:
        break
    img = transform(img[0])
    img.save('examples-viewmaker-fashion/'+ str(i)+'.jpg')
