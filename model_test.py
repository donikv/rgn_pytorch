#%%
import os
import numpy as np
from io import StringIO

from torch.utils.data import DataLoader
# from tqdm import tqdm
# import bcolz
# from data_utlis import ProteinNetDataset
from pathlib import Path

from model import RGN
import sys
from gpu_profile import gpu_profile
import torch
sys.settrace(gpu_profile)
home = str(Path.home())
# pn_path = home + '/casp7/training_30'
pn_path = home + '\\Downloads\\casp7\\casp7\\testing'
pn_test = os.curdir + '/../rgn_pytorch/data/text_sample'
model = RGN(42, window_size=9)
model.cuda(0)
model.train(pn_test, log_interval=1, optimiz='Adam', epochs=1, profile_gpu=True)
model.test(pn_test)
f = open("model.pickle", "w+")
torch.save(model, f)
exit()
