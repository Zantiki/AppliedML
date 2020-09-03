import torch
import torch.nn as nn
import assignment3.utils as util


class ConvoA(nn.Module):
    def __init__(self):
        super(ConvoA, self).__init__()

        # Model layers (includes initialized model variables):
        self.conv_first = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.pool_first = nn.MaxPool2d(kernel_size=2)

        self.conv_second = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool_second = nn.MaxPool2d(kernel_size=2)

        self.dense = nn.Linear(64 * 7 * 7, 10)

    def logits(self, x):
        x = self.conv_first(x)
        x = self.pool_first(x)
        x = self.conv_second(x)
        x = self.pool_second(x)
        return self.dense(x.reshape(-1, 64 * 7 * 7))

    # Predictor
    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    # Cross Entropy loss
    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


class ConvoB(nn.Module):

    def __init__(self):
        super(ConvoB, self).__init__()

        # Model layers (includes initialized model variables):
        self.conv_first = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.pool_first = nn.MaxPool2d(kernel_size=2)

        self.conv_second = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool_second = nn.MaxPool2d(kernel_size=2)
        self.dense = nn.Linear(64 * 7 * 7, 1024)
        self.dense_two = nn.Linear(1024, 10)

    def logits(self, x):
        x = self.conv_first(x)
        x = self.pool_first(x)
        x = self.conv_second(x)
        x = self.pool_second(x)

        x = self.dense(x.reshape(-1, 64 * 7 * 7))

        return self.dense_two(x)

    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    # Cross Entropy loss
    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


class ConvoC(nn.Module):

    def __init__(self):
        super(ConvoC, self).__init__()

        # Model layers (includes initialized model variables):
        self.conv_first = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.pool_first = nn.MaxPool2d(kernel_size=2)

        self.conv_second = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool_second = nn.MaxPool2d(kernel_size=2)
        self.dense = nn.Linear(64 * 7 * 7, 1024)
        self.dense_two = nn.Linear(1024, 10)

        self.relu = nn.ReLU()

    def logits(self, x):
        #x = self.relu(x)
        x = self.conv_first(x)
        x = self.pool_first(x)
        x = self.conv_second(x)
        x = self.pool_second(x)

        x = self.dense(x.reshape(-1, 64 * 7 * 7))
        return self.relu(self.dense_two(x))

    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    # Cross Entropy loss
    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    # Accuracy
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())

class convD(nn.Module):
    pass