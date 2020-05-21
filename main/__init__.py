import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from network import Network
import os.path as path
import itertools

training_root = "D:\Data\\train\\"
test_root = "D:\Data\\t\\"
classes = ['Inne', 'Makulatura', 'Plastik', 'Szklo']
model_path = "model.pt"
train_set = torchvision.datasets.ImageFolder(root=training_root,
                                             transform=transforms.ToTensor())
training_data_loader = torch.utils.data.DataLoader(train_set, batch_size=6,
                                                   shuffle=True, num_workers=4)

test_set = torchvision.datasets.ImageFolder(root=test_root,
                                            transform=transforms.ToTensor())
test_data_loader = torch.utils.data.DataLoader(test_set, batch_size=6,
                                               shuffle=True,
                                               num_workers=4)


def main():
    net = None
    if path.exists(model_path):
        print("Model detected, loading...")
        net = torch.load(model_path)
    else:
        net = Network.Net(lr=0.001, epochs=25, classes=classes,
                          training_data_loader=training_data_loader,
                          test_data_loader=test_data_loader)
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
    plt.figure(figsize=(4, 4))
    c_matrix = net._test()
    plot_confusion_matrix(c_matrix, net.classes)
    plot_confusion_matrix(c_matrix, net.classes, normalize=True)
    print(len(test_set.targets))


def plot_confusion_matrix(matrix, classes, normalize=False):
    fmt = None
    title = None
    if normalize:
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
        title = "Normalized confusion matrix"
    else:
        fmt = 'd'
        title = "Confusion matrix"

    print(matrix)
    plt.imshow(matrix, interpolation='nearest', cmap=plt.get_cmap('BuGn'))
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    threshold = matrix.max() / 2
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, format(matrix[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if matrix[i, j] > threshold else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(title + ".png")
    plt.show()


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    main()
