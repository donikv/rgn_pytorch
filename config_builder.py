import torch.nn as nn

class ModelConfig():

    def __init__(self, in_dim, linear_out=20, cell='LSTM', 
                 num_layers=2, 
                 alphabet_size=20, 
                 hidden_size=800, 
                 bidirectional=True,
                 dropout=0.4):

        self.in_dim = in_dim
        self.linear_out = linear_out
        self.num_layers = num_layers
        self.alphabet_size = alphabet_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.cell = cell
        self.bidirectional=bidirectional

    def get_cell(self):
        if self.cell == 'LSTM':
            cell = nn.LSTM(self.in_dim, self.hidden_size, self.num_layers, bidirectional=self.bidirectional, dropout=self.dropout)
            return cell
        
        return nn.GRU(self.in_dim, self.hidden_size, self.num_layers, bidirectional=self.bidirectional, dropout=self.dropout)


class TrainingConfig():

    def __init__(self, train_loader, test_loader, epochs=30, log_interval=10, batch_size=32, optimizer='SGD', verbose=False, profile_gpu=False, loss='dRMSD'):
        self.loaders = {'train': train_loader, 'test': test_loader}
        self.epochs = epochs
        self.log_interval = log_interval 
        self.batch_size = batch_size
        self.loss = loss
        self.optimizer = optimizer
        self.verbose = verbose
        self.profile_gpu = profile_gpu


