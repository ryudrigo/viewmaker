from src.systems.image_systems import TransferExpertSystem
import torch
from src.utils import utils
from dotmap import DotMap
from src.datasets import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as T
from PIL import Image
'''
config_json = utils.load_json("experiments/experiments/transfer_expert_cifar_simclr_resnet18_meta_fashionmnist/config.json")
config = DotMap(config_json)
checkpoint = torch.load("experiments/experiments/transfer_expert_cifar_simclr_resnet18_meta_fashionmnist/checkpoints/epoch=45.ckpt")

model = TransferExpertSystem(config)
model.load_state_dict(checkpoint['state_dict'], strict=False)
'''
_, dataset = datasets.get_image_datasets( 'meta_dtd',default_augmentations='all')
data_loader = DataLoader(dataset)

transform = T.ToPILImage()

for i, batch in enumerate(iter(data_loader)):
    img = transform(batch[1][0])

    img.save('examples-expert-flowers/'+ str(i)+'.jpg')
