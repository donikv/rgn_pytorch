#%%
import os
import numpy as np
from io import StringIO

from torch.utils.data import DataLoader
from tqdm import tqdm
# import bcolz
from data_utlis import ProteinNetDataset
from pathlib import Path
import torch
from model import RGN
import sys
#from gpu_profile import gpu_profile


from config_builder import build_configs

mc, tc = build_configs("./gru.config")

model = RGN(mc)
model.cuda(0)
model.train(tc)
model.test(pn_test)
f = open("model.pickle", "wb")
torch.save(model, f)
exit()

