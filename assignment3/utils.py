import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

def train(model, x, y, learning_rate, epochs):


    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    for epoch in range(epochs):
        for batch in range(len(x)):
            model.loss(x[batch], y[batch]).backward()
            optimizer.step()
            optimizer.zero_grad()
    return model

def get_data_mnist():
    mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True)
    x_train = mnist_train.data.reshape(-1, 1, 28,
                                       28).float()  # torch.functional.nn.conv2d argument must include channels (1)
    y_train = torch.zeros((mnist_train.targets.shape[0], 10))  # Create output tensor
    y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1  # Populate output

    mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
    x_test = mnist_test.data.reshape(-1, 1, 28,
                                     28).float()  # torch.functional.nn.conv2d argument must include channels (1)
    y_test = torch.zeros((mnist_test.targets.shape[0], 10))  # Create output tensor
    y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1  # Populate output

    # Normalization of inputs
    mean = x_train.mean()
    std = x_train.std()
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    # Divide training data into batches to speed up optimization
    batches = 600
    x_train_batches = torch.split(x_train.cuda(), batches)
    y_train_batches = torch.split(y_train.cuda(), batches)
    return x_train_batches, y_train_batches, x_test, y_test


def passion_for_fashion():
    mnist_train = torchvision.datasets.FashionMNIST("./data", train=True, download=True)
    x_train = mnist_train.data.reshape(-1, 1, 28,
                                       28).float()  # torch.functional.nn.conv2d argument must include channels (1)
    y_train = torch.zeros((mnist_train.targets.shape[0], 10))  # Create output tensor
    y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1  # Populate output

    mnist_test = torchvision.datasets.FashionMNIST('./data', train=False, download=True)
    x_test = mnist_test.data.reshape(-1, 1, 28,
                                     28).float()  # torch.functional.nn.conv2d argument must include channels (1)
    y_test = torch.zeros((mnist_test.targets.shape[0], 10))  # Create output tensor
    y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1  # Populate output

    # Normalization of inputs
    mean = x_train.mean()
    std = x_train.std()
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std

    # Divide training data into batches to speed up optimization
    batches = 600
    x_train_batches = torch.split(x_train.cuda(), batches)
    y_train_batches = torch.split(y_train.cuda(), batches)
    return x_train_batches, y_train_batches, x_test, y_test

