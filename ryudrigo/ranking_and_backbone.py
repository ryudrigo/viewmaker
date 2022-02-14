import pytorch_lightning as pl
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from src.utils import utils
import os
from dotmap import DotMap
from src.systems.image_systems import PretrainViewMakerSystem
import torch
import torch.nn as nn
from src.datasets import datasets
from torch.utils.data import DataLoader
from torchvision.datasets import FashionMNIST
import torch.nn.functional as F
import torch.utils.data as torch_data_utils
from torch.utils.tensorboard import SummaryWriter
from ryudrigo.custom_datasets import FFHQThumbDataset
import torchvision.transforms.functional as TF
import random

def my_collate(batch):
    # item: a tuple of (img, label)
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    data = torch.stack(data)
    target = torch.LongTensor(target)
    return [data, target]

class Backbone(pl.LightningModule):
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_size = config.optim_params.batch_size
        self.viewmaker, self.pretrain_config = self.load_pretrained_model()
        
        self.train_dataset, self.val_dataset = datasets.get_image_datasets(
            config.data_params.dataset,
            default_augmentations='all'
        )
        
        self.viewmaker = self.viewmaker.eval()
        utils.frozen_params(self.viewmaker)

        self.backbone_feature_extractor, self.backbone_classifier = self.create_backbone()
        self.writer = SummaryWriter()

        self.transform = T.ToPILImage()

    def load_pretrained_model(self):
        pretrained_dir = self.config.pretrain_model.exp_dir
        checkpoint_name = self.config.pretrain_model.checkpoint_name

        config_path = os.path.join(pretrained_dir, 'config.json')
        config_json = utils.load_json(config_path)
        config = DotMap(config_json)
       
        SystemClass = globals()[config.system]
        system = SystemClass(config)
        checkpoint_file = os.path.join(pretrained_dir, 'checkpoints', checkpoint_name)
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        system.load_state_dict(checkpoint['state_dict'], strict=False)
        
        viewmaker = system.viewmaker.eval()

        return viewmaker, system.config

    def create_backbone(self):
        num_classes = self.train_dataset.NUM_CLASSES
        backbone = models.resnet18(pretrained=True)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        feature_extractor = nn.Sequential (*layers)
        classifier = nn.Linear (num_filters,num_classes) 
        return feature_extractor, classifier
        
    def forward(self, img, valid=False):
        self.backbone_feature_extractor.eval()
        with torch.no_grad():
            representations = self.backbone_feature_extractor(img).flatten(1)
        x = self.backbone_classifier(representations)
        return x
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        #this line is responsible for viewmaker augmentation
        #x = self.viewmaker(x)
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.writer.add_scalar('Loss/train', loss, self.current_epoch*self.batch_size+batch_idx)
        return loss
    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optim
    
    def train_dataloader(self):
        #dataset= FashionMNIST (root='data/fashionmnist', download=True,train=True, transform=T.Compose([T.Grayscale(3),T.ToTensor(), T.Normalize(0.286, 0.330)]))
        dataset = FFHQThumbDataset('data/ffhq-thumb')
        '''
        indices = torch.arange(20000)
        dataset_20k = torch_data_utils.Subset(dataset, indices)
        '''
        return DataLoader(dataset, num_workers=6, batch_size=self.batch_size, collate_fn=my_collate)
   
    def val_dataloader(self):
        #dataset= FashionMNIST (root='data/fashionmnist', download=True,train=False, transform=T.Compose([T.Grayscale(3),T.ToTensor()]))
        dataset = FFHQThumbDataset('data/ffhq-thumb', train=False)
        return DataLoader(dataset, num_workers=6, batch_size=self.batch_size, collate_fn=my_collate)
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.writer.add_scalar('Loss/val', loss, self.current_epoch*self.batch_size+batch_idx)
        
        pred= torch.argmax(y_hat, dim=1)
        acc = torch.sum(y == pred).item() / (len(y) * 1.0)
        self.writer.add_scalar('Val/acc', acc, self.current_epoch*self.batch_size+batch_idx)
        return {"loss": loss}

activations={}
def save_activation(name):
    def activations_hook(module, act_in, act_out):
        activations[name]=act_out.detach()
    return activations_hook
    
class Ranking (pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.batch_size=128
        self.backbone = self.load_backbone()
        self.hooks = {}
        for name, module in self.backbone.named_modules():
            self.hooks[name] = module.register_forward_hook(save_activation(name))
        self.linear = torch.nn.Linear(122*64, 1)
        self.flatten = torch.nn.Flatten()
        self.writer = SummaryWriter()
        
    
    def load_backbone(self):
        config_json = utils.load_json("ryudrigo/config.json")
        config = DotMap(config_json)
        checkpoint = torch.load("lightning_logs/version_58/checkpoints/epoch=199.ckpt")
        backbone_model = Backbone(config)
        backbone_model.load_state_dict(checkpoint['state_dict'], strict=False)
        utils.frozen_params(backbone_model)
        return backbone_model
        
    def get_activations(self):
        list_of_names = ['backbone_feature_extractor.0']
        list_of_acts = [activations[list_of_names[0]].mean(dim=(-2, -1))]
        for ra, name in enumerate(list(activations.keys())):
            if 'conv' in name:
                num_of_feat_maps = activations[name].shape[-3]
                act = activations[name].mean(dim=(-2, -1))
                act=torch.chunk(act, num_of_feat_maps//64, dim=-1)
                list_of_acts+=act
                for _ in range (num_of_feat_maps//64):
                    list_of_names.append(name)
                
        return torch.cat(list_of_acts), list_of_names
        
    def generate_backbone_activations(self, imgs):
        x=[]
        y=[]
        imgs1=[]
        imgs2=[]
        for img in imgs:
            one_img_x = []
            one_img_y = []            
            angle1=0
            angle2=0
            while (abs(angle1-angle2)<5):
                angle1 = random.randint(-40, 40)
                angle2 = random.randint(-40, 40)
            img1 = TF.rotate(img, angle1, expand=True)
            img2 = TF.rotate(img, angle2, expand=True)
            imgs1.append(img1)
            imgs2.append(img2)
            self.backbone(torch.unsqueeze(img1, 0))
            act1, _ = self.get_activations()
            self.backbone(torch.unsqueeze(img2, 0))
            act2, which_layer = self.get_activations() #which layer can come from act1 or act2, they're equal
            act = torch.cat((act1, act2))
            which_layer+=which_layer #duplicate the entries so which_layer[i] corresponds to act[i]
            if angle1>angle2: 
                one_img_y.append(1.)
            else: #angles will never be equal becasue they need a difference larger than 5
                one_img_y.append(0.)
            one_img_x.append(act)
            one_img_x=torch.cat(one_img_x)
            one_img_y=torch.as_tensor(one_img_y,device=torch.device('cuda'))
            x.append(torch.unsqueeze(one_img_x, 0))
            y.append(torch.unsqueeze(one_img_y,0))
        x = torch.cat(x)
        y = torch.cat(y)
        return x, y, which_layer, imgs1, imgs2 #returns images for debugging purposes
        
    def forward(self, backbone_activations):
        backbone_activations = self.flatten(backbone_activations)
        y_hat = torch.sigmoid(self.linear(backbone_activations))
        return y_hat
        
    def training_step(self, batch, batch_idx):
        x, _ = batch
        x, y, _, _, _ = self.generate_backbone_activations(x)
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.writer.add_scalar('Loss/train', loss, self.current_epoch*self.batch_size+batch_idx)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x, y, _, _, _ = self.generate_backbone_activations(x)
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.writer.add_scalar('Loss/val', loss, self.current_epoch*self.batch_size+batch_idx)
        
        pred= torch.argmax(y_hat, dim=1)
        acc = torch.sum(y == pred).item() / (len(y) * 1.0)
        self.writer.add_scalar('Val/acc', acc, self.current_epoch*self.batch_size+batch_idx)
        return loss
    
    def train_dataloader(self):
        dataset = FFHQThumbDataset('data/ffhq-thumb')
        return DataLoader(dataset, num_workers=6, batch_size=self.batch_size, collate_fn=my_collate)
   
    def val_dataloader(self):
        dataset = FFHQThumbDataset('data/ffhq-thumb', train=False)
        return DataLoader(dataset, num_workers=6, batch_size=self.batch_size, collate_fn=my_collate)
    
    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optim
    
if __name__ == "__main__":
    trainer = pl.Trainer(gpus='0', max_epochs=50)
    model = Ranking()
    trainer.fit (model)