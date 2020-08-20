
import torch
import matplotlib.pyplot as plt
import urllib3


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


def get_data(three_dimensions=False, non_linear=False):
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
    elif non_linear:
        file_url = "https://gitlab.com/ntnu-tdat3025/regression/childgrowth-datasets/raw/master/day_head_circumference.csv"
        data_csv = http.request("GET", file_url).data.decode("UTF-8")
        lines = data_csv.split("\n")
        for line in lines:
            if line != "" and "# day,head circumference" not in line:
                data_line = line.split(",")
                day, circumference = data_line[0], data_line[1]
                x_list.append(float(day))
                y_list.append(float(circumference))

    else:
        file_url = "https://gitlab.com/ntnu-tdat3025/regression/childgrowth-datasets/raw/master/length_weight.csv"
        data_csv = http.request("GET", file_url).data.decode("UTF-8")
        lines = data_csv.split("\n")
        for line in lines:
            if line != "" and  line != "# length,weight":
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
        z_train = torch.tensor(z_list).reshape(-1, 1)
        return x_train, z_train
    else:
        x_train = torch.tensor(x_list).reshape(-1, 1)
        return x_train, y_train


def train(model, x, y, learning_rate, epochs ):
    optimizer = torch.optim.SGD([model.b, model.W], learning_rate)
    for epoch in range(epochs):
        model.loss(x, y).backward()
        optimizer.step()
        optimizer.zero_grad()
    return model


def plot_2D(model, x_train, y_train):
    plt.xlabel('x')
    plt.ylabel('y')
    x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]])  # x = [[1], [6]]]
    plt.scatter(x_train, y_train, label='Actual')
    plt.plot(x, model.f(x).detach(), label='TrainedLine', c="red")
    plt.scatter(x_train, model.f(x_train).detach(), label='TrainedPoints', c='green')
    plt.legend()
    print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))
    plt.show()


def plot_3D(model, x_train, y_train):
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
    ax.scatter(x_list, y_list, z_list, label="Trained", c="green")
    ax.scatter(x_list, y_list, y_train, label="Actual",  c="blue")
    ax.set_xlabel("     Length as cm")
    ax.set_ylabel("     Weight as kg")
    ax.set_zlabel("     Age as days")
    ax.legend()
    plot_info.set_text(
        '$W=\\left[{%.2f},{%.2f}\\right]$\n$b=[%.2f]$\n$loss = %.2f$' %
        (model.W[0, 0], model.W[1, 0], model.b[0, 0], model.loss(x_train, y_train)))
    print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))
    plt.show()


if __name__ == "__main__":

    config = {
        "Linear_2d": (0.00001, 120000),
        "Linear_3d": (0.00001, 50000),
        "Non_linear": (0.000001, 80000)
    }

    # A: Simple linear 2D
    model = LinearRegressionModel()
    x, y = get_data()
    train(model, x, y, config["Linear_2d"][0], config["Linear_2d"][1])
    plot_2D(model, x, y)

    # B: 3D
    x, y = get_data(three_dimensions=True)
    model = LinearRegression3D()
    train(model, x, y, config["Linear_3d"][0], config["Linear_3d"][1])
    plot_3D(model,  x, y)

    # C: Non linear 2D
    model = NonLinearRegressionModel()
    x, y = get_data(non_linear=True)
    train(model, x, y, config["Non_linear"][0], config["Non_linear"][1])
    plot_2D(model, x, y)