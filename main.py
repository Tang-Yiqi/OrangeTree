import torch
import pandas as pd
import numpy as np

from modules.LoadModel import LoadNewModel
from datetime import datetime

import os
from omegaconf import OmegaConf
import hydra
import json
import math
from time import sleep
import random

import torch._dynamo

torch._dynamo.config.cache_size_limit = 32


configs = {}
@hydra.main(version_base=None, config_path="config", config_name="general")
def LoadConfigs(cfg):
    global configs
    configs = OmegaConf.to_container(cfg, resolve=True)
    configs['device'] = torch.device(configs['device'])
    configs['StartTime'] = datetime.now().strftime("%Y_%m_%d_%H.%M.%S")

    return configs
LoadConfigs()

with open('models/' + configs['model']['name'] + configs['StartTime'] + '.json', 'w') as file:
    json.dump(str(configs), file)


os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

RandomSeed = 42
torch.manual_seed(RandomSeed)
torch.cuda.manual_seed_all(RandomSeed)
torch.manual_seed(RandomSeed)
np.random.seed(RandomSeed)
random.seed(RandomSeed)

torch.use_deterministic_algorithms(True, warn_only=False)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

torch.set_float32_matmul_precision("highest")

model = LoadNewModel(configs)


if configs['model']['name'] == 'OrangeTreeFast':
    from modules.TrainAndTestFast import TrainModel
    model = TrainModel(model, configs)
elif configs['model']['name'] == 'OrangeTree':
    from modules.TrainAndTest import TrainModel
    model = TrainModel(model, configs)
