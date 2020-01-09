from geometric_ops import *

class dRMSD(nn.Module):
    def __init__(self):
        super(dRMSD, self).__init__()

    def forward(self, predicted, actual, mask=None):
        return drmsd(predicted, actual, mask=mask)


class AngularLoss(nn.Module):
    def __init__(self):
        super(AngularLoss, self).__init__()

    def forward(self, predicted, actual, mask=None):
        return calc_angular_difference(predicted, actual)

