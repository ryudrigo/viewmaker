import pytorch_lightning as pl
from ryudrigo.ranking_and_backbone import Backbone
from src.utils import utils
from dotmap import DotMap

trainer = pl.Trainer(gpus='0', max_epochs=50)
config_json = utils.load_json("ryudrigo/config.json")
config = DotMap(config_json)
model = Backbone(config)
trainer.fit (model)