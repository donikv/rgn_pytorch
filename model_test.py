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

home = str(Path.home())
pn_path = home + '/casp7/training_30'
#pn_path = os.curdir + '/../rgn_pytorch/data/text_sample'
# dataset = ProteinNetDataset(pn_path)
# trn_data = DataLoader(dataset, batch_size=32, shuffle=True)
model = RGN(42)
model.cuda(0)
# for b_id, data in enumerate(trn_data):
#     sequences = data['sequence']
#     out = model(sequences)
#     print(out.shape)
#     print(data['coords'].transpose(0, 1).shape)
model.train(pn_path, log_interval=1)
