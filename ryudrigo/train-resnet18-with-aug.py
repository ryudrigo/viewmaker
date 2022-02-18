import os
import torchvision.transforms as T
import numpy as np
import torch
import torch.nn as nn
import torchmetrics
import torchvision
from kornia import image_to_tensor, tensor_to_image
from kornia.augmentation import ColorJitter, RandomChannelShuffle, RandomHorizontalFlip, RandomThinPlateSpline, Resize,Normalize
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import CSVLogger
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.datasets import STL10
from torch.utils.tensorboard import SummaryWriter

class DataAugmentation(nn.Module):

    def __init__(self, apply_color_jitter: bool = False) -> None:
        super().__init__()
        self._apply_color_jitter = apply_color_jitter

        self.transforms = nn.Sequential(
            Normalize((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
            RandomHorizontalFlip(p=0.75),
            RandomChannelShuffle(p=0.75),
            RandomThinPlateSpline(p=0.75),
        )

        self.jitter = ColorJitter(1, 1, 1, 1)

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x: Tensor) -> Tensor:
        x_out = self.transforms(x)  # BxCxHxW
        if self._apply_color_jitter:
            x_out = self.jitter(x_out)
        return x_out

class Preprocess(nn.Module):

    @torch.no_grad()  # disable gradients for effiency
    def forward(self, x) -> Tensor:
        x_tmp: np.ndarray = np.array(x)  # HxWxC
        x_out: Tensor = image_to_tensor(x_tmp, keepdim=True)  # CxHxW
        return x_out.float() / 255.0


class CoolSystem(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = torchvision.models.resnet18(pretrained=False)

        self.preprocess = Preprocess()  # per sample transforms

        self.transform = DataAugmentation(apply_color_jitter=True)  # per batch augmentation_kornia

        self.accuracy = torchmetrics.Accuracy()
        
        self.pil_transform = T.ToPILImage()
        
        self.writer = SummaryWriter()
        
        self.batch_size=64

    def forward(self, x):
        return F.softmax(self.model(x))

    def compute_loss(self, y_hat, y):
        return F.cross_entropy(y_hat, y)

    def show_batch(self):
        
        imgs, labels = next(iter(self.train_dataloader()))
        imgs_aug = self.transform(imgs)  # apply transforms
        
        for i, img in enumerate(imgs):
            img = self.pil_transform(img)
            img.save('temp3/{}.jpg'.format(i))
        

    def training_step(self, batch, batch_idx):
        x, y = batch
        x_aug = self.transform(x)  # => we perform GPU/Batched data augmentation
        y_hat = self(x_aug)
        loss = self.compute_loss(y_hat, y)
        self.writer.add_scalar("Train/loss", loss, self.current_epoch*self.batch_size+batch_idx)
        self.writer.add_scalar("Train/acc", self.accuracy(y_hat, y), self.current_epoch*self.batch_size+batch_idx)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.compute_loss(y_hat, y)
        self.writer.add_scalar("Val/loss", loss, self.current_epoch*self.batch_size+batch_idx)
        self.writer.add_scalar("Val/acc", self.accuracy(y_hat, y), self.current_epoch*self.batch_size+batch_idx)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, self.trainer.max_epochs, 0)
        #return [optimizer], [scheduler]
        return optimizer

    def prepare_data(self):
        #CIFAR10(os.getcwd(), train=True, download=True,  transform=self.preprocess)
        STL10(os.getcwd(), split='train', download=True,  transform=self.preprocess)
        #CIFAR10(os.getcwd(), train=False, download=True, transform=self.preprocess)
        STL10(os.getcwd(), split='test', download=True,  transform=self.preprocess)
        

    def train_dataloader(self):
        #dataset = CIFAR10(os.getcwd(), train=True, download=True, transform=self.preprocess)
        dataset = STL10(os.getcwd(), split='train', download=True, transform=self.preprocess)
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=6)
        return loader

    def val_dataloader(self):
        #dataset = CIFAR10(os.getcwd(), train=False, download=True, transform=self.preprocess)
        dataset = STL10(os.getcwd(), split='test', download=True, transform=self.preprocess)
        loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=6)
        return loader
        
if __name__ == "__main__":
    model = CoolSystem()
    model.show_batch()
    trainer = Trainer(gpus='0',max_epochs=1000)    
    trainer.fit(model)
    