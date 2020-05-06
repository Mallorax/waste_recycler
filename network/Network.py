import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, lr, epochs, classes, training_data_loader, test_data_loader):
        super(Net, self).__init__()
        # initializing variables
        self.lr = lr
        self.epochs = epochs
        self.classes = classes
        self.training_data_loader = training_data_loader
        self.test_data_loader = test_data_loader
        self.loss_history = []
        self.acc_history = []
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        # Creating layers
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(4)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(4, 6, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(6)
        self.conv3 = nn.Conv2d(6, 8, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(8)
        self.maxpool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(self.calculate_input_dim(), len(self.classes))
        self.droput = nn.Dropout(0.3)
        self.optimizer = optim.SGD(self.parameters(), lr=self.lr)
        self.loss = nn.CrossEntropyLoss()
        self.to(self.device)

    def forward(self, batch_data):
        batch_data = torch.tensor(batch_data).to(self.device)

        batch_data = self.conv1(batch_data)
        batch_data = self.bn1(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.maxpool1(batch_data)

        batch_data = self.conv2(batch_data)
        batch_data = self.bn2(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv3(batch_data)
        batch_data = self.bn3(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.maxpool2(batch_data)

        batch_data = batch_data.view(batch_data.size()[0], -1)
        result = self.droput(self.fc1(batch_data))
        return result

    def _train(self):
        self.train()
        for i in range(self.epochs):
            ep_loss = []
            ep_acc = []
            for j, data in enumerate(self.training_data_loader):
                self.optimizer.zero_grad()
                input, label = data[0].to(self.device), data[1].to(self.device)
                prediction = self.forward(input)
                prediction = F.softmax(prediction, dim=1)
                classes = torch.argmax(prediction, dim=1)
                wrong = torch.where(classes != label,
                                    torch.tensor([1.]).to(self.device),
                                    torch.tensor([0.]).to(self.device))
                acc = 1 - torch.sum(wrong) / self.training_data_loader.batch_size
                loss = self.loss(prediction, label)
                loss.backward()
                self.optimizer.step()
                ep_loss.append(loss.item())
                ep_acc.append(acc.item())
                if j % 100 == 99:
                    print('Epoch %d, step %d, average loss %.3f accuracy %.3f' %
                          (i + 1, j + 1, np.mean(ep_loss), np.mean(ep_acc)))
            self.loss_history.append(np.mean(ep_loss))
            self.acc_history.append(np.mean(ep_acc))

    def _test(self):
        self.eval()
        class_correct = list(0. for i in range(4))
        class_total = list(0. for i in range(4))
        correct = 0
        total = 0
        for j, data in enumerate(self.test_data_loader):
            input, labels = data[0].to(self.device), data[1].to(self.device)
            prediction = self.forward(input)
            _, predicted = torch.max(prediction, 1)
            c = (predicted == labels).squeeze()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i in range(6):
                print(labels.size())
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

        for i in range(4):
            print('Accuracy of %5s : %d %%' % (
                self.classes[i], 100 * class_correct[i] / class_total[i]))
        print('Overall accuracy of the network: %d %%' % (100 * correct / total))

    def calculate_input_dim(self):
        batch_data = torch.zeros((1, 3, 256, 256))
        batch_data = self.conv1(batch_data)
        batch_data = self.maxpool1(batch_data)
        batch_data = self.conv2(batch_data)
        batch_data = self.conv3(batch_data)
        batch_data = self.maxpool2(batch_data)
        return int(np.prod(batch_data.size()))
