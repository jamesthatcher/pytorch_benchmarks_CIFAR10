# coding=utf-8

import time

import pandas as pd
import torch
import torchvision
from torch import nn, optim
from torchvision import models, transforms

from model_trainer import train

n_epochs = 10
n_classes = 10
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.001
use_cuda = True

random_seed = 1
torch.manual_seed(random_seed)

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('~/.pytorch/CIFAR10/', train=True, download=True,
                                 transform=torchvision.transforms.Compose([
                                     torchvision.transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CIFAR10('~/.pytorch/CIFAR10/', train=False, download=True,
                                 transform=torchvision.transforms.Compose([
                                     torchvision.transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
    batch_size=batch_size_test, shuffle=True)


train_times = pd.DataFrame()
for m, n in zip([models.squeezenet1_1(), models.vgg19_bn(), models.densenet121()], ['Squeeze Net', 'VGG19',
                                                                                 'DenseNet201']):

    optimizer = optim.Adam(m.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    if use_cuda:
        start = time.time()
        train(n_classes=n_classes, n_epochs=n_epochs, train_loader=train_loader, test_loader=test_loader,
          model=m.cuda(),
          optimizer=optimizer,
          criterion=criterion,
          use_cuda=True, save_path=f'{n}_gpu.pt')
        end = time.time()
        d = {'Model': [n], 'Train time (s)': [end - start], 'GPU': [True]}
        train_times = train_times.append(pd.DataFrame(d), ignore_index=True)
        print(f'GPU trained in {end - start} seconds...')

    else:
        start = time.time()
        train(n_classes=n_classes, n_epochs=n_epochs, train_loader=train_loader, test_loader=test_loader,
          model=m,
          optimizer=optimizer,
          criterion=criterion,
          use_cuda=False, save_path=f'{n}_cpu.pt')
        end = time.time()

        d = {'Model': [n], 'Train time (s)': [end - start], 'CPU': [False]}
        train_times = train_times.append(pd.DataFrame(d), ignore_index=True)
        print(f'CPU trained in {end - start} seconds...')

if use_cuda:
    train_times.to_csv('gpu_train_times.csv')
else:
    train_times.to_csv('cpu_train_times.csv')
