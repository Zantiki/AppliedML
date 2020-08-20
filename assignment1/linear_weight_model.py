
import torch
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d, art3d

import urllib3
import numpy as np


class LinearRegressionModel:
    def __init__(self):

        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return x @ self.W + self.b

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))


class LinearRegression3D:

    def __init__(self):
        self.W = torch.tensor([[0.0], [0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def f(self, x):
        return x @ self.W + self.b

    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))


class NonLinearRegressionModel:
    def __init__(self):
        # requires_grad enables calculation of gradients
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return 20*torch.sigmoid(x @ self.W + self.b) + 31

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))


def get_data(three_dimensions=False):
    http = urllib3.PoolManager()
    x_list = []
    y_list = []
    z_list = []

    if three_dimensions:
        file_url = "https://gitlab.com/ntnu-tdat3025/regression/childgrowth-datasets/raw/master/day_length_weight.csv"
        data_csv = http.request("GET", file_url).data.decode("UTF-8")
        lines = data_csv.split("\n")
        for line in lines:
            if line != "" and line != "# day,length,weight":
                data_line = line.split(",")
                day, length, weight = data_line[0], data_line[1], data_line[2]
                x_list.append(float(length))
                y_list.append(float(weight))
                z_list.append(float(day))
    else:
        file_url = "https://gitlab.com/ntnu-tdat3025/regression/childgrowth-datasets/raw/master/length_weight.csv"
        data_csv = http.request("GET", file_url).data.decode("UTF-8")
        lines = data_csv.split("\n")
        for line in lines:
            if line != "" and  "# length,weight" not in line:
                data_line = line.split(",")
                length, weight = data_line[0], data_line[1]
                x_list.append(float(length))
                y_list.append(float(weight))

    y_train = torch.tensor(y_list).reshape(-1, 1)

    if len(z_list) > 0:

        x_packed = []

        """while len(x_list) > 0:
            x_packed.append([x_list.pop(), y_list.pop()])"""
        x_train = torch.tensor([[length, weight] for length, weight in zip(x_list, y_list)])
        z_train = torch.tensor(z_list)
        return x_train, z_train
    else:
        x_train = torch.tensor(x_list).reshape(-1, 1)
        return x_train, y_train


def train(model, x, y):
    learning_rate = 0.00001
    optimizer = torch.optim.SGD([model.b, model.W], learning_rate)
    for epoch in range(120000):
        model.loss(x, y).backward()
        optimizer.step()
        optimizer.zero_grad()
    return model


def plot2D(model, x_train, y_train):
    plt.plot(x_train, y_train, 'o', label='$(\\hat x^{(i)},\\hat y^{(i)})$')
    plt.xlabel('x')
    plt.ylabel('y')
    x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]])  # x = [[1], [6]]]
    plt.plot(x, model.f(x).detach(), label='$y = f(x) = xW+b$')
    plt.legend()
    print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))
    plt.show()


def plot_3D_alt(model, x_train, y_train):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_info = fig.text(0.01, 0.02, '')
    y_list = []
    x_list = []

    for i in x_train:
        x_list.append(i[0].detach())
        y_list.append(i[1].detach())

    z_list = model.f(x_train).detach()
    ax.scatter(x_list, y_list, z_list, label="Trained")
    ax.scatter(x_list, y_list, y_train, label="Actual")
    ax.set_xlabel("     Length as cm")
    ax.set_ylabel("     Weight as kg")
    ax.set_zlabel("     Age as days")
    ax.legend()
    plot_info.set_text(
        '$W=\\left[\\stackrel{%.2f}{%.2f}\\right]$\n$b=[%.2f]$\n$loss = %.2f$' %
        (model.W[0, 0], model.W[1, 0], model.b[0, 0], model.loss(x_train, y_train)))
    print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))
    plt.show()


if __name__ == "__main__":

    # A: 2D
    """model = LinearRegressionModel()
    x, y = get_data()
    train(model, x, y)
    plot2D(model, x, y)
    input("Move to assignment B?")"""

    # B: 3D
    x, y = get_data(three_dimensions=True)
    model = LinearRegression3D()
    train(model, x, y)
    plot_3D_alt(model,  x, y)
