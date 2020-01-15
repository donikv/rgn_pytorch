import torch.nn as nn
import torch.optim as optim
from losses import *
from data_utlis import ProteinNetDataset
from torch.utils.data import DataLoader

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

    def __init__(self, pn_train, pn_valid, pn_test, epochs=30, log_interval=10, batch_size=32, optimizer='SGD', verbose=False, profile_gpu=False, loss='dRMSD', lr=1e-4):
        
        train_dataset = ProteinNetDataset(pn_train) #if self.window_size is None else ProteinNetWindowedDataset(pn_path)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        valid_dataset = ProteinNetDataset(pn_valid) #if self.window_size is None else ProteinNetWindowedDataset(pn_path)
        valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
        test_dataset = ProteinNetDataset(pn_test) #if self.window_size is None else ProteinNetWindowedDataset(pn_path)
        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, pin_memory=True)

        self.loaders = {'train': train_loader, 'test': test_loader, 'valid': valid_loader}
        self.epochs = epochs
        self.log_interval = log_interval 
        self.batch_size = batch_size
        self.loss = loss
        self.optimizer = optimizer
        self.verbose = verbose
        self.profile_gpu = profile_gpu
        self.lr = lr
        if profile_gpu:
            from gpu_profile import gpu_profile
            import sys
            gpu_profile(frame=sys._getframe(), event='line', arg=None)

    def get_optimizer(self, parameters):
        if self.optimizer == 'Adam':
            return optim.Adam(parameters, lr=self.lr)

        return optim.SGD(parameters, lr=self.lr)
    
    def get_loss(self):
        if self.loss == 'Angular':
            return AngularLoss()
        return dRMSD()

def build_configs(f):
    import configparser
    config = configparser.ConfigParser()
    config.read(f)
    model_params = config['MODEL']
    train_params = config['TRAINING']
    model_config = ModelConfig(int(model_params['in']), int(model_params['linear_out']), model_params['cell'], int(model_params['num_layers']), int(model_params['alphabet_size']), int(model_params['hidden_size']), model_params.getboolean('bidirectional'), model_params.getfloat('dropout'))
    train_config = TrainingConfig(train_params['train_path'], train_params['valid_path'],train_params['test_path'], int(train_params['epochs']), int(train_params['log_interval']), int(train_params['batch_size']), train_params['optimizer'], loss=train_params['loss'], lr=train_params.getfloat('lr'))
    return (model_config, train_config)
