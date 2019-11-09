import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.adagrad as Adagrad
import numpy as np
from torch.utils.data import DataLoader

from rgn_pytorch.data_utlis import ProteinNetDataset
from rgn_pytorch.geometric_ops import *


class Angularization(nn.Module):
    def __init__(self, d_in=800, dih_out=3, linear_out=800, alphabet_size=20):
        super(Angularization, self).__init__()
        self.linear_layer = nn.Linear(2 * d_in, linear_out, bias=True)
        self.softmax_layer = nn.Softmax()
        self.alphabet = torch.torch.FloatTensor(alphabet_size, dih_out).uniform_(-np.pi, np.pi)

    def forward(self, x):
        lin_out = self.linear_layer(x)
        softmax_out = self.softmax_layer(lin_out)
        ang_out = calculate_dihedrals(softmax_out, self.alphabet)

        return ang_out

    def parameters(self, recurse):
        params = []
        for param in self.linear_layer.parameters(recurse):
            params.append(param)
        params.append(self.alphabet)
        return params


class dRMSD(nn.Module):
    def __init__(self, weights=None):
        super(dRMSD, self).__init__()
        self.weights = weights

    def forward(self, predicted, actual):
        return drmsd(predicted, actual, weights=self.weights)


class RGN(nn.Module):
    def __init__(self, d_in, linear_out=20, h=800, num_layers=2, alphabet_size=20):
        super(RGN, self).__init__()
        self.lstm_layers = []
        self.num_layers = num_layers
        i = 0
        while i < num_layers:
            if i == 0:
                self.lstm_layers.append(nn.LSTM(d_in, hidden_size=h, bidirectional=True))
            else:
                self.lstm_layers.append(nn.LSTM(2 * h, hidden_size=h, bidirectional=True))
        self.angularization_layer = Angularization(alphabet_size=alphabet_size)

        self.error = dRMSD()

    def forward(self, x):
        lstm_out = x
        for lstm in self.lstm_layers:
            lstm_out = lstm(lstm_out)

        ang_out = self.angularization_layer(lstm_out)
        return calculate_coordinates(ang_out)

    def parameters(self, recurse):
        #Type: (T, bool) -> Iterator[Parameter]:
        lstm_params = [lstm.parameters(recurse) for lstm in self.lstm_layers]
        params = []
        for lstm_param in lstm_params:
            for param in lstm_param:
                params.append(param)
        for param in self.angularization_layer.parameters(recurse):
            params.append(param)
        return params

    def train(self, pn_path, epochs=30, log_interval=10):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        criterion = self.error

        train_loader = DataLoader(ProteinNetDataset(pn_path), batch_size=32, shuffle=True)

        for epoch in range(epochs):
            for batch_idx, pn_data in enumerate(train_loader):
                data, target = pn_data['sequence'], pn_data['coords']
                data, target = nn.Variable(data), nn.Variable(target)
                optimizer.zero_grad()
                net_out = self(data)
                loss = criterion(net_out, target)
                loss.backward()
                optimizer.step()
                if batch_idx % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), loss.data[0]))

    def _transform_for_lstm(self, data):
        return data