import torch
from ryudrigo.ranking_and_backbone import Ranking

checkpoint = torch.load('lightning_logs/version_53/checkpoints/epoch=2.ckpt')
ranking = Ranking()
ranking.load_state_dict(checkpoint['state_dict'], strict=False)
dataset = FFHQThumbDataset('data/ffhq-thumb')
data_loader = DataLoader(dataset, batch_size=128)

acts = {}
#obs. not done in current commit. Work in progress!
for i, batch in enumerate(iter(data_loader)):
    if i>6:
        break
    ranking.backbone(batch)
    acts, names = ranking.get_activations()
    for a, n in zip (acts, names):
        if n not in acts.keys():
            acts[n]=[a]
        else
            acts[n].append(a)
print (acts.keys())