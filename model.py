import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

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
