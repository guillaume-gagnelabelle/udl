import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def show(trainloader):
    # get some random training images
    dataiter_train = iter(trainloader)
    images, labels = next(dataiter_train)
    print(' '.join('%2s' % labels[j].item() for j in range(32)))
    imshow(torchvision.utils.make_grid(images[:32]))

def show_images(images, labels):
    # get some random training images
    print(' '.join('%2s' % labels[j].item() for j in range(32)))
    imshow(torchvision.utils.make_grid(images[:32]))
