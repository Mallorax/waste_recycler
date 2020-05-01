import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from network import Network
import os.path as path

training_root = "D:\Data\Trainset\\"
test_root = "D:\Data\Testset\\"
classes = ['Inne', 'Makulatura', 'Plastik', 'Szklo']
model_path = "model.pt"
train_set = torchvision.datasets.ImageFolder(root=training_root,
                                             transform=transforms.ToTensor())
training_loader = torch.utils.data.DataLoader(train_set, batch_size=8,
                                              shuffle=True, num_workers=4)
test_set = torchvision.datasets.ImageFolder(root=test_root,
                                            transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_set, batch_size=8,
                                          shuffle=True, num_workers=4)


def main():
    # dataiter = iter(trainingloader)
    # images, labels = dataiter.next()
    # print(' '.join('%5s' % classes[labels[j]] for j in range(8)))
    # imshow(torchvision.utils.make_grid(images))
    net = None
    if path.exists(model_path):
        print("Model detected, loading...")
        net = torch.load(model_path)
    else:
        net = Network.Net(lr=0.001, epochs=15, classes=classes,
                          training_data_loader=training_loader,
                          test_data_loader=test_loader)
        net.to(net.device)
        net._train()
        plt.plot(net.loss_history)
        plt.savefig('loss_history.png')
        plt.show()
        plt.plot(net.acc_history)
        plt.savefig('acc_history.png')
        plt.show()
        net.eval()
        torch.save(net, model_path)
    print("Testing...")
    net._test()


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    main()
