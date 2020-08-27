import torch
import torchvision
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

    filename = "nand.png"

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
    filename = "xor.png"

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


class ImgModel:

    def __init__(self):
        self.W = torch.ones((784, 10), requires_grad=True)
        self.b = torch.ones((1, 10), requires_grad=True)

    def f(self, x):
        soft_max = torch.nn.Softmax(dim=1)
        return soft_max(x @ self.W + self.b)

    def loss(self, x, y):
        return torch.nn.functional.cross_entropy(self.f(x), y.argmax(1))



def get_mnist_data():

    # Load observations from the mnist dataset. The observations are divided into a training set and a test set
    mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True)
    x_train = mnist_train.data.reshape(-1, 784).float()  # Reshape input
    y_train = torch.zeros((mnist_train.targets.shape[0], 10))  # Create output tensor
    y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1  # Populate output

    mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
    x_test = mnist_test.data.reshape(-1, 784).float()  # Reshape input
    y_test = torch.zeros((mnist_test.targets.shape[0], 10))  # Create output tensor
    y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1  # Populate output

    return x_train, y_train, x_test, y_test


def make_test_data():

    x_train_pairs = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    x_train_singles = torch.tensor([[0.0], [1.0]])
    not_y_train = torch.tensor([[1.0], [0.0]])
    xor_y_train = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
    nand_y_train = torch.tensor([[1.0], [1.0], [1.0], [0.0]])

    return x_train_pairs, x_train_singles, not_y_train, xor_y_train, nand_y_train


def calc_accuracy(model, x, y):
    return torch.mean(torch.eq(model.f(x).argmax(1), y.argmax(1)).float())


def plot_img(model, x_test, y_test):
    plt.rc('font', size=15)
    fig = plt.figure(figsize=(15, 15))
    plot_info = fig.text(0.01, 0.02, '')
    x = 0
    for i in range(1, 11):
        img = model.W.detach().numpy()[:, x].reshape(28, 28)
        x += 1
        fig.add_subplot(2, 5, i)
        plt.imshow(img)

    plot_info.set_text("loss: {}, accuracy: {}%".format(np.round(model.loss(x_test, y_test).detach().numpy(), 4),
                                        np.round(calc_accuracy(model, x_test, y_test).detach().numpy()*100, 4)))
    plt.savefig("plots/imgs.png")
    plt.show()


def plot_2d(model, x_train, y_train):
    plt.xlabel('x')
    plt.ylabel('y')

    x = torch.tensor([[x * 0.1 for x in range(11)]]).reshape(-1, 1)
    # x = [[1], [6]]]
    plt.scatter(x_train, y_train, label='Actual')
    plt.plot(x, model.f(x).detach(), label='TrainedLine', c="green")
    # plt.scatter(x_train, model.f(x_train).detach(), label='TrainedPoints', c='green')
    plt.legend()
    print("loss = %s" % (model.loss(x_train, y_train).detach().numpy()))
    plt.savefig("plots/not.png")
    plt.show()


def grid(model, lower=0, upper=1, steps=20):
    gx, gy = [], []
    mod = (upper - lower) / steps

    for i in [(x * mod) + lower for x in range(0, steps+1)]:
        for j in [(x * mod) + lower for x in range(0, steps+1)]:

            gx.append(i)
            gy.append(j)

    gz = model.f(torch.FloatTensor([[i, j] for i, j in zip(gx, gy)])).detach()
    return [gx, gy, gz]


def plot_3d(model, x_train, y_train):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot_info = fig.text(0.01, 0.02, '')
    y_list = []
    x_list = []

    for i in x_train:
        x_list.append(i[0].detach())
        y_list.append(i[1].detach())

    x_grid, y_grid, z_grid = grid(model=model, steps=75)
    ax.scatter(x_grid, y_grid, z_grid, label="Trained", c="green")
    ax.scatter(x_list, y_list, y_train, label="Actual",  c="blue")
    ax.set_xlabel("     X")
    ax.set_ylabel("     Y")
    ax.set_zlabel("     Z")
    ax.legend()
    plot_info.set_text("loss = %s" % (model.loss(x_train, y_train).detach().numpy()))
    plt.savefig("plots/{}".format(model.filename))
    plt.show()


def train(model, x, y, learning_rate, epochs):
    optimizer_params = []
    model_params = vars(model)

    for param in model_params:
        if param == "filename":
            continue
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
        },
        "img":{
            "epochs": 500,
            "lr": 0.1
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

    # XOR-bad
    model = XorModel()
    model.filename = "xor_bad.png"
    model.W1 = torch.tensor([[-20.0, -40.0], [30.0, -20.0]], requires_grad=True)
    model.b1 = torch.tensor([[-5.0, 1.0]], requires_grad=True)
    model.W2 = torch.tensor([[1.0], [1.0]], requires_grad=True)
    model.b2 = torch.tensor([[-5.0]], requires_grad=True)

    train(model, x_train_pairs, xor_y_train, config["xor"]["lr"], config["xor"]["epochs"])
    plot_3d(model, x_train_pairs, xor_y_train)

    # MNIST
    x_train, y_train, x_test, y_test = get_mnist_data()
    model = ImgModel()
    train(model, x_train, y_train, config["img"]["lr"], config["img"]["epochs"])
    plot_img(model, x_test, y_test)

