import torch
from torchvision import transforms, datasets
import os
import json
from PIL import Image
from src.utils import utils
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader
from dotmap import DotMap
import torchvision.transforms as T

class FFHQThumbDataset(Dataset):
    # order of emotions(may be used as labels): ['anger', 'contempt', 'disgust', 'fear', 'happiness', 'neutral', 'sadness', 'surprise']
    def __init__(self, root_dir, train=True):
        self.root_dir=root_dir  
        self.labels_dir = os.path.join (root_dir, 'json')
        self.train=train
        self.num_validation=10050
        
    def __len__(self):
        if (self.train):
            return len([name for name in os.listdir(self.labels_dir)]) - self.num_validation
        else:
            return self.num_validation
            
    def __getitem__(self, idx):
        if not self.train:
            idx+= len([name for name in os.listdir(self.labels_dir)])- self.num_validation - 2
            
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if len(str(idx))<4:
            folder_name = '0000'
        else:
            folder_name = str(idx//1000) + '000'
        if len(folder_name) == 4: #e.g. from 2000 to 02000
            folder_name='0'+folder_name

        if len(str(idx)) == 1:
            file_name = '0000'+str(idx)
        elif len(str(idx)) == 2:
            file_name = '000'+str(idx)
        elif len(str(idx)) == 3:
            file_name = '00'+str(idx)
        elif len(str(idx)) == 4:
            file_name = '0'+str(idx)
        else:
            file_name = str(idx)
        img_path = os.path.join (self.root_dir, folder_name)
        img_path = os.path.join (img_path, file_name+'.png')
        labels_json = utils.load_json(os.path.join(self.labels_dir, file_name+'.json'))
        if len(labels_json)==0:
            return torch.zeros((3,128, 128)), 5
        labels = DotMap(labels_json[0])
        emotion_list =[b for a, b in vars(labels.faceAttributes.emotion).items()]
        emotion_values = [b for a, b in emotion_list[0].items()]
        emotion_tensor = torch.Tensor(emotion_values)
        emotion_label = torch.argmax(emotion_tensor, dim=0)
        if emotion_label!=4 and emotion_label!=5:
            return torch.zeros((3,128, 128)), 5
        try:
            img = Image.open(img_path)
        except:
            return torch.zeros((3,128, 128)), 5
        img = TF.to_tensor(img)
        return img, emotion_label
        

if __name__ == "__main__":
    dataset = FFHQThumbDataset('data/ffhq-thumb')
    data_loader = DataLoader(dataset)
    pil_transform = T.ToPILImage()
    for i, batch in enumerate(iter(data_loader)):
        img = batch[0]
        pred = batch[1]
        if i>99:
            break
        img = pil_transform(img[0])
        img.save('ryudrigo/examples-ffhq/'+ str(i)+'-'+str(pred)+ '.jpg')
