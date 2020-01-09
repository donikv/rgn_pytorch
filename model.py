import sys
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# import torch.optim.adagrad as Adagrad
import numpy as np
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader

from data_utlis import ProteinNetDataset
from data_utlis import ProteinNetWindowedDataset
from geometric_ops import *
from simple_profile import dump_tensors
import pickle
from config_builder import *

class Angularization(nn.Module):
    def __init__(self, d_in=800, dih_out=3, linear_out=20, alphabet_size=20):
        super(Angularization, self).__init__()
        self.linear_layer = nn.Linear(2 * d_in, linear_out, bias=True)
        self.alphabet = torch.FloatTensor(alphabet_size, dih_out).uniform_(-np.pi, np.pi).move_to_gpu().requires_grad_(True)

        self.model = nn.Sequential(OrderedDict([
            ('LINEAR', nn.Linear(2 * d_in, linear_out, bias=True)),
            ('SOFTMAX', nn.Softmax(dim=1))
        ]))

    def forward(self, x):
        out = self.model(x)
        ang_out = calculate_dihedrals(out, self.alphabet)

        return ang_out

    def parameters(self, recurse):
        yield self.alphabet
        yield from self.model.parameters(recurse=recurse)


class dRMSD(nn.Module):
    def __init__(self):
        super(dRMSD, self).__init__()

    def forward(self, predicted, actual, mask=None):
        return drmsd(predicted, actual, mask=mask)


class AngularLoss(nn.Module):
    def __init__(self):
        super(AngularLoss, self).__init__()

    def forward(self, predicted, actual, mask=None):
        return calc_angular_difference(predicted, actual, mask=mask)


class RGN(nn.Module):
    def __init__(self, config, window_size=None):
        super(RGN, self).__init__()
        self.num_layers = config.num_layers
        self.hidden_size = config.hidden_size
        self.window_size = config.window_size

        self.angularization_layer = Angularization(d_in=config.hidden_size, dih_out=3, alphabet_size=config.alphabet_size)

        self.lstm = config.get_cell()

    def forward(self, x):
        lens = list(map(len, x))
        batch_sz = len(lens)
        x = x.float().transpose(0, 1).contiguous()
        h0 = torch.zeros((self.num_layers * 2, batch_sz, self.hidden_size)).move_to_gpu().requires_grad_(True)
        c0 = torch.zeros((self.num_layers * 2, batch_sz, self.hidden_size)).move_to_gpu().requires_grad_(True)

        lstm_out, _ = self.lstm(x, (h0, c0))

        ang_out = self.angularization_layer(lstm_out)
        return calculate_coordinates(ang_out)

    def parameters(self, recurse=True):
        yield from self.lstm.parameters(recurse=recurse)
        yield from self.angularization_layer.parameters(recurse=recurse)

    def train(self, config):
        optimizer = config.get_optimizer(self.parameters())
        criterion = config.get_loss()
        torch.autograd.set_detect_anomaly = True

        train_loader = config.loaders['train_loader']
        for epoch in range(config.epochs):
            for batch_idx, pn_data in enumerate(train_loader):
                data, target, mask = pn_data['sequence'], pn_data['coords'], pn_data['mask'].transpose(0, 1).move_to_gpu()
                data, target = data.move_to_gpu(), target.transpose(0, 1).move_to_gpu()
                net_out = self(data)
                optimizer.zero_grad()
                loss = criterion(net_out, target, mask)
                l = loss.mean()
                l.backward()
                optimizer.step()
                if config.verbose:
                    # dump_tensors()
                    print(list(map(lambda x: (x.grad, len(x.grad)), self.parameters())))
                if batch_idx % config.log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), l.item()))
                torch.cuda.empty_cache()

    def test(self, config):
        test_loader = config.loaders['test_loader']
        test_loss = 0
        predictions = []
        with torch.no_grad():
            for pn_data in test_loader:
                name = pn_data['name']
                data, target, mask = pn_data['sequence'], pn_data['coords'], pn_data['mask'].transpose(0, 1).move_to_gpu()
                data, target = data.move_to_gpu(), target.transpose(0, 1).move_to_gpu()
                output = self(data)
                curr_loss = self.error(output, target, mask).item()  # sum up batch loss
                predictions.append((name, output.detach(), target.detach()))
                print('Error on {}: {}'.format(name, curr_loss))
                test_loss += curr_loss
                torch.cuda.empty_cache()
            print('Total loss: {}'.format(test_loss/len(test_loader)))
            self._save_prediction_to_file(predictions, 'predictions.pickle')

    def _save_prediction_to_file(self, predictions, out):
        outf = open(out, 'wb+')
        pickle.dump(predictions, outf)
        outf.close()
