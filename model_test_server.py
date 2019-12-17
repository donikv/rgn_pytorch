#%%
import os
import numpy as np
from io import StringIO

from torch.utils.data import DataLoader
# from tqdm import tqdm
# import bcolz
from data_utlis import ProteinNetDataset
from pathlib import Path

from model import RGN
import sys
#from gpu_profile import gpu_profile

#sys.settrace(gpu_profile)
home = str(Path.home())
pn_path = home + '/casp7/training_30'
pn_test = home + '/casp7/testing'
model = RGN(42)
model.cuda(0)
model.train(pn_path, log_interval=1, epochs=5)
model.test(pn_test)
exit()
