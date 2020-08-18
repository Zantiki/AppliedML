
import torch
import matplotlib
import numpy as np


def get_data(three_dimensions=False):
    fetched_data = []
    if three_dimensions:
        file_url = "https://gitlab.com/ntnu-tdat3025/regression/childgrowth-datasets/raw/master/day_length_weight.csv"
        with open(file_url, "r") as data_csv:
            line = data_csv.readline()
            while line:
                line = data_csv.readline()
                data_line = line.split(",")
                day, length, weight = data_line[0], data_line[1], data_line[2]
                fetched_data.append([day, length, weight])
    else:
        file_url = "https://gitlab.com/ntnu-tdat3025/regression/childgrowth-datasets/raw/master/length_weight.csv"
        with open(file_url, "r") as data_csv:
            line = data_csv.readline()
            while line:
                line = data_csv.readline()
                data_line = line.split(",")
                length, weight = data_line[0], data_line[1]
                fetched_data.append([length, weight])

    return np.array(fetched_data)



def train():
    pass


def plot(data_dict):
    pass

def calc_loss(model):
    pass


if __name__ == "__main__":

    get_data()
    exit()
    # 2D
    data = get_data()
    model = train(data)
    calc_loss()
    plot(model)


    # 3D
    data = get_data()
    model = train(data)
    calc_loss()
    plot(model)