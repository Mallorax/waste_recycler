import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

trening_root = "D:\Data\Trainset\\"
test_root = "D:\Data\Testset\\"
classes = ('Inne', 'Makulatura', 'Plastik', 'Szklo')
trainset = torchvision.datasets.ImageFolder(root=trening_root,
                                            transform=transforms.ToTensor())
trainingloader = torch.utils.data.DataLoader(trainset, batch_size=8,
                                             shuffle=True, num_workers=8)
testset = torchvision.datasets.ImageFolder(root=test_root,
                                           transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=8,
                                         shuffle=True, num_workers=8)


def main():
    dataiter = iter(trainingloader)
    images, labels = dataiter.next()
    print(' '.join('%5s' % classes[labels[j]] for j in range(8)))
    imshow(torchvision.utils.make_grid(images))


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == '__main__':
    main()
