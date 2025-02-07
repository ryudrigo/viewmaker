import torch
from ryudrigo.ranking_and_backbone import Ranking
from ryudrigo.custom_datasets import FFHQThumbDataset
from torch.utils.data import DataLoader

def my_collate(batch):
    # item: a tuple of (img, label)
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    data = torch.stack(data)
    target = torch.LongTensor(target)
    return [data, target]

#contrast checkpoint = torch.load('lightning_logs/version_54/checkpoints/epoch=1.ckpt')
#rotation checkpoint = torch.load('lightning_logs/version_55/checkpoints/epoch=4.ckpt')
#saturation checkpoint = torch.load('lightning_logs/version_57/checkpoints/epoch=1.ckpt')
checkpoint = torch.load('lightning_logs/version_58/checkpoints/epoch=0.ckpt')
ranking = Ranking()
ranking.load_state_dict(checkpoint['state_dict'], strict=False)
dataset = FFHQThumbDataset('data/ffhq-thumb')
data_loader = DataLoader(dataset, batch_size=128,collate_fn=my_collate)

all_acts = {}
sample_size=32
for i, batch in enumerate(iter(data_loader)):
    if i>=sample_size:
        break
    ranking.backbone(batch[0])
    act, names = ranking.get_activations()
    for a, n in zip (act, names):
        if n not in all_acts.keys():
            all_acts[n]=[a]
        else:
            all_acts[n].append(a)

layers_multiplicity=[]
layers_std=[]
for layer in all_acts.values():
    layer = torch.stack(layer)
    layer = layer.permute((1,0,2))
    std=torch.std(layer)
    '''
    print ('this is shape:')
    print (layer.shape)
    print ('this is std:')
    print (std)
    print('\n')     
    '''    
    layers_std.append(std)
    layers_multiplicity.append(layer.shape[1]//sample_size)
i=0
j=0
layer_mean_impact=[]
layer_max_impact=[]
while (i<61):
    w = ranking.linear.weight[0][i*64*2:(i+layers_multiplicity[j])*64*2]        
    '''
    print (i)
    print (j)
    print (layers_multiplicity[j])
    print (i*64*2)
    print ((i+layers_multiplicity[j])*64*2)
    print (len(w)//(64*2))
    print ('\n')
    '''
    i=i+layers_multiplicity[j]
    j+=1
    mean_w = sum(w)/len(w)
    max_w = max(w)
    layer_mean_impact.append(mean_w)
    layer_max_impact.append(max_w)
layer_max_impact = torch.as_tensor(layer_max_impact)/sum(layer_max_impact)    
layer_mean_impact = torch.as_tensor(layer_mean_impact)/sum(layer_mean_impact)    
print ('layer_mean_impact:')
for impact in layer_mean_impact:
    print (float(impact))
print ('layer_max_impact:')
for impact in layer_max_impact:
    print (float(impact))
for name in all_acts.keys():
    print (name)