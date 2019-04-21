# coding=utf-8

import time

import pandas as pd
import torch
import torchvision
from torch import nn, optim
from torchvision import models, transforms

from model_trainer import train

n_epochs = 1
n_classes = 10
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.001

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

transfer_model = models.squeezenet1_1()
optimizer = optim.Adam(transfer_model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

train_times = pd.DataFrame()
for m, n in zip([models.squeezenet1_1(), models.vgg19_bn(), models.densenet201()], ['Squeeze Net', 'VGG19',
                                                                                 'DenseNet201']):
    start = time.time()
    train(n_classes=n_classes, n_epochs=n_epochs, train_loader=train_loader, test_loader=test_loader,
          model=transfer_model,
          optimizer=optimizer,
          criterion=criterion,
          use_cuda=True, save_path=f'{n}_gpu.pt')
    end = time.time()
    d = {'Model': [n], 'Train time (s)': [start-end], 'GPU': [True]}
    train_times = train_times.append(pd.DataFrame(d), ignore_index=True)

    start = time.time()
    train(n_classes=n_classes, n_epochs=n_epochs, train_loader=train_loader, test_loader=test_loader,
          model=transfer_model,
          optimizer=optimizer,
          criterion=criterion,
          use_cuda=True, save_path=f'{n}_cpu.pt')
    end = time.time()

    d = {'Model': [n], 'Train time (s)': [start - end], 'GPU': [False]}
    train_times = train_times.append(pd.DataFrame(d), ignore_index=True)
    print(f'Completed training model: {n}')
