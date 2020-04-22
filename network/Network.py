import torch
import numpy as np
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, lr, epochcs, num_of_classes):
        super(Net, self).__init__()
        # initializing variables
        self.lr = lr
        self.epochs = epochcs
        self.num_of_classes = num_of_classes
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        print(self.device)
