import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


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
        # Creating layers
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 6, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(6)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(6, 8, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(8)
        self.conv4 = nn.Conv2d(8, 10, padding=1)
        self.bn4 = nn.BatchNorm2d(10)
        self.maxpool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(self.calculate_input_dim(), self.num_of_classes)
        self.optimizer = optim.SGD(self.parameters(), lr=self.lr)
        self.loss = nn.CrossEntropyLoss()
        self.to(self.device)

    def forward(self, batch_data):
        batch_data = torch.tensor(batch_data).to(self.device)

        batch_data = self.conv1(batch_data)
        batch_data = self.bn1(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv2(batch_data)
        batch_data = self.bn2(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.maxpool1(batch_data)

        batch_data = self.conv3(batch_data)
        batch_data = self.bn3(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv4(batch_data)
        batch_data = self.bn4(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.maxpool2(batch_data)

        batch_data = batch_data.view(batch_data.size()[0], -1)
        result = self.fc1(batch_data)
        return result

    def calculate_input_dim(self):
        batch_data = torch.zeros((1, 3, 256, 256))
        batch_data = self.conv1(batch_data)
        batch_data = self.conv2(batch_data)
        batch_data = self.maxpool1(batch_data)
        batch_data = self.conv3(batch_data)
        batch_data = self.conv4(batch_data)
        batch_data = self.maxpool2(batch_data)
        return int(np.prod(batch_data.size()))
