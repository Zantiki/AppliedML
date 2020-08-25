import torch
import numpy as np
import matplotlib.pyplot as plt


class NotModel:

    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    def logits(self, x):
        return x @ self.W + self.b

    # Predictor
    def f(self, x):
        return torch.sigmoid(self.logits(x))

    # Cross Entropy loss
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.logits(x), y)

class NandModel:

    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0], [0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return torch.sigmoid(x @ self.W + self.b)

    # Cross Entropy loss
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy_with_logits(self.f(x), y)


class XorModel:
    def __init__(self):

        self.W1 = torch.tensor([[10.0, -10.0], [10.0, -10.0]], requires_grad=True)
        self.b1 = torch.tensor([[-5.0, 15.0]], requires_grad=True)
        self.W2 = torch.tensor([[10.0], [10.0]], requires_grad=True)
        self.b2 = torch.tensor([[-15.0]], requires_grad=True)

    # First layer function
    def f1(self, x):
        return torch.sigmoid(x @ self.W1 + self.b1)

    # Second layer function
    def f2(self, h):
        return torch.sigmoid(h @ self.W2 + self.b2)

    # Predictor
    def f(self, x):
        return self.f2(self.f1(x))

    # Uses Cross Entropy
    def loss(self, x, y):
        return torch.nn.functional.binary_cross_entropy(self.f(x), y)


def make_test_data():

    x_train_pairs = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    x_train_singles = torch.tensor([[0.0], [1.0]])
    not_y_train = torch.tensor([[1.0], [0.0]])
    xor_y_train = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    nand_y_train = torch.tensor([[1.0], [1.0], [1.0], [0.0]])

    return x_train_pairs, x_train_singles, not_y_train, xor_y_train, nand_y_train


def plot_2d(model, x_train, y_train):
    plt.xlabel('x')
    plt.ylabel('y')

    x = torch.tensor([[x * 0.1 for x in range(11)]]).reshape(-1, 1)
    # x = [[1], [6]]]
    plt.scatter(x_train, y_train, label='Actual')
    plt.plot(x, model.f(x).detach(), label='TrainedLine', c="red")
    plt.scatter(x_train, model.f(x_train).detach(), label='TrainedPoints', c='green')
    plt.legend()
    print("loss = %s" % (model.loss(x_train, y_train)))
    plt.show()


def plot_3d(model, x_train, y_train):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_info = fig.text(0.01, 0.02, '')
    y_list = []
    x_list = []

    for i in x_train:
        x_list.append(i[0].detach())
        y_list.append(i[1].detach())
    print(torch.min(x_train))
    z_list = model.f(x_train).detach()

    sigmoid_x = torch.tensor([[x * 0.001 for x in range(1100)]]).detach()
    sigmoid_y = torch.tensor([[x * 0.001 for x in range(1100)]]).detach()

    y_sigmoid_list = [x * 0.001 for x in range(1100)]
    y_sigmoid_list.sort(reverse=True)
    sigmoid_y_2 = torch.tensor([y_sigmoid_list]).detach()

    sigmoid_x_y = torch.tensor([[x * 0.001, x * 0.001] for x in range(1100)])
    sigmoid_x_y_2 = torch.tensor([[x, y] for y, x in zip(y_sigmoid_list, [x * 0.001 for x in range(1100)])])

    sigmoid_z = model.f(sigmoid_x_y).detach()
    sigmoid_z_2 = model.f(sigmoid_x_y_2).detach()

    ax.scatter(sigmoid_x, sigmoid_y, sigmoid_z, c='red')
    ax.scatter(sigmoid_x, sigmoid_y_2, sigmoid_z_2, c='red')

    ax.scatter(x_list, y_list, z_list, label="Trained", c="green")
    ax.scatter(x_list, y_list, y_train, label="Actual",  c="blue")
    ax.set_xlabel("     X")
    ax.set_ylabel("     Y")
    ax.set_zlabel("     Z")
    ax.legend()
    plot_info.set_text("loss = %s" % (model.loss(x_train, y_train).detach()))
    plt.show()


def train(model, x, y, learning_rate, epochs):
    optimizer_params = []
    model_params = vars(model)

    for param in model_params:
        optimizer_params.append(model_params[param])

    optimizer = torch.optim.SGD(optimizer_params, learning_rate)
    for epoch in range(epochs):
        model.loss(x, y).backward()
        optimizer.step()
        optimizer.zero_grad()
    return model


if __name__ == "__main__":

    config = {
        "xor": {
            "epochs": 100,
            "lr": 0.001
        },
        "not": {
            "epochs": 10000,
            "lr": 1
        },
        "nand": {
            "epochs": 10000,
            "lr": 0.001
        }

    }


    x_train_pairs, x_train_singles, not_y_train, xor_y_train, nand_y_train = make_test_data()

    # NOT
    model = NotModel()
    train(model, x_train_singles, not_y_train, config["not"]["lr"], config["not"]["epochs"])
    plot_2d(model, x_train_singles, not_y_train)

    # NAND
    model = NandModel()
    train(model, x_train_pairs, nand_y_train, config["not"]["lr"], config["not"]["epochs"])
    plot_3d(model, x_train_pairs, nand_y_train)

    # XOR
    model = XorModel()
    train(model, x_train_pairs, xor_y_train, config["xor"]["lr"], config["xor"]["epochs"])
    plot_3d(model, x_train_pairs, xor_y_train)

