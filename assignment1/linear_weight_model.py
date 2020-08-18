
import torch
import matplotlib
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


def get_data(three_dimensions=False):
    http = urllib3.PoolManager()
    fetched_data = []
    if three_dimensions:
        file_url = "https://gitlab.com/ntnu-tdat3025/regression/childgrowth-datasets/raw/master/day_length_weight.csv"
        data_csv = http.request("GET", file_url).data.decode("UTF-8")
        lines = data_csv.split("\n")
        for line in lines:
            if line != "":
                data_line = line.split(",")
                day, length, weight = data_line[0], data_line[1], data_line[2]
                fetched_data.append([day, length, weight])
    else:
        file_url = "https://gitlab.com/ntnu-tdat3025/regression/childgrowth-datasets/raw/master/length_weight.csv"
        data_csv = http.request("GET", file_url).data.decode("UTF-8")
        lines = data_csv.split("\n")
        for line in lines:
            if line != "":
                data_line = line.split(",")
                length, weight = data_line[0], data_line[1]
                fetched_data.append([length, weight])

    return np.array(fetched_data)

def get_loss(params, models, ):
    pass


def train(model, training_params):

    optimizer = torch.optim.SGD([model.b, model.W], 0.01)
    for epoch in range(10000):
        model.loss(training_params).backward()
        optimizer.step()
        optimizer.zero_grad()
    return model


def plot(data_dict):
    pass


if __name__ == "__main__":

    get_data(True)
    exit()
    # 2D
    data = get_data()
    model = train(data)
    plot(model)


    # 3D
    data = get_data()
    model = train(data)
    calc_loss()
    plot(model)