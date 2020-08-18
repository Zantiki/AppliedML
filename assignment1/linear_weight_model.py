
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
    def loss(self, x, y, z):
        return torch.mean(torch.square(self.f(x) - y))


class LinearRegression3D:

    def __init__(self):
        self.W = np.array([[0.0], [0.0]])
        self.b = np.array([[0.0]])

    def f(self, x):
        return x @ self.W + self.b

    def loss(self, x, y, z):
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

        while len(x_list) > 0:
            x_packed.append([x_list.pop(), z_list.pop()])
        x_train = torch.tensor(x_packed).reshape(-1, 2)
        print("x_train: ", x_train)
        return x_train, y_train
    else:
        x_train = torch.tensor(x_list).reshape(-1, 1)
        return x_train, y_train


def train(model, x, y, z):

    learning_rate = 0.0001
    optimizer = torch.optim.SGD([model.b, model.W], learning_rate)
    for epoch in range(12000):
        model.loss(x, y, z).backward()
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
    print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train, None)))
    plt.show()


def plot3D(model, x, y):
    fig = plt.figure('Linear regression: 3D')
    x_train = x.numpy()
    y_train = y.numpy()
    plot1 = fig.add_subplot(111, projection='3d')
    plot1.plot(x_train[:, 0].squeeze(),
               x_train[:, 1].squeeze(),
               y_train[:, 0].squeeze(),
               'o',
               label='$(\\hat x_1^{(i)}, \\hat x_2^{(i)},\\hat y^{(i)})$',
               color='blue')

    plot1_f = plot1.plot_wireframe(np.array([[]]), np.array([[]]), np.array([[]]), color='green',
                                  label='$y = f(x) = xW+b$')

    plot1_info = fig.text(0.01, 0.02, '')

    plot1_loss = []
    for i in range(0, x_train.shape[0]):
        line, = plot1.plot([0, 0], [0, 0], [0, 0], color='red')
        plot1_loss.append(line)
        if i == 0:
            line.set_label('$|f(\\hat x^{(i)})-\\hat y^{(i)}|$')

    plot1.set_xlabel('$x_1$')
    plot1.set_ylabel('$x_2$')
    plot1.set_zlabel('$y$')
    plot1.legend(loc='upper left')
    plot1.set_xticks([])
    plot1.set_yticks([])
    plot1.set_zticks([])
    plot1.w_xaxis.line.set_lw(0)
    plot1.w_yaxis.line.set_lw(0)
    plot1.w_zaxis.line.set_lw(0)
    plot1.quiver([0], [0], [0], [np.max(x_train[:, 0] + 1)], [0], [0], arrow_length_ratio=0.05, color='black')
    plot1.quiver([0], [0], [0], [0], [np.max(x_train[:, 1] + 1)], [0], arrow_length_ratio=0.05, color='black')
    plot1.quiver([0], [0], [0], [0], [0], [np.max(y_train[:, 0] + 1)], arrow_length_ratio=0.05, color='black')
    update_figure(x_train, y_train, plot1, plot1_loss, plot1_info, fig)
    plt.show()


def update_figure(x_train, y_train, plot1, plot1_loss, plot1_info, fig, event=None):

    if event is not None:
        if event.key == 'W':
            model.W[0, 0] += 0.01
        elif event.key == 'w':
            model.W[0, 0] -= 0.01
        elif event.key == 'E':
            model.W[1, 0] += 0.01
        elif event.key == 'e':
            model.W[1, 0] -= 0.01

        elif event.key == 'B':
            model.b[0, 0] += 0.05
        elif event.key == 'b':
            model.b[0, 0] -= 0.05

    global plot1_f
    x1_grid, x2_grid = np.meshgrid(np.linspace(1, 6, 10), np.linspace(1, 4.5, 10))
    y_grid = np.empty([10, 10])
    for i in range(0, x1_grid.shape[0]):
        for j in range(0, x1_grid.shape[1]):
            y_grid[i, j] = model.f([[x1_grid[i, j], x2_grid[i, j]]])
    plot1_f = plot1.plot_wireframe(x1_grid, x2_grid, y_grid, color='green')

    for i in range(0, x_train.shape[0]):
        plot1_loss[i].set_data(np.array([x_train[i, 0], x_train[i, 0]]), np.array([x_train[i, 1], x_train[i, 1]]))
        plot1_loss[i].set_3d_properties(np.array([y_train[i, 0], model.f(x_train[i, :])], dtype=object))

    fig.canvas.draw()



if __name__ == "__main__":

    # 3D
    x, y = get_data(three_dimensions=True)
    print("X: ", x)
    model = LinearRegression3D()
    plot3D(model, x, y)
    exit()
    # 2D
    get_data()
    model = LinearRegressionModel()
    x, y = get_data()
    model = train(model, x, y, None)
    plot2D(model, x, y)

    # 3D
   # data = get_data()
    # model = train(data)
    # plot(model)