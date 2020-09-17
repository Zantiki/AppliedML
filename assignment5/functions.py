import pandas
from pandas import DataFrame as df
import pandas
import matplotlib.pyplot as plt
import itertools

def build_feature_dict(filename):
    feature_map = {
        # index: {
        #   name: feature_name,
        #   class_map: [
        #           {c: class}
        #       ]
        #   }

    }
    with open(filename, "r") as file:
        file_data = file.readlines()

        for line in file_data:
            if "\n" in line:
                line = line.replace("\n", "")
            values = line.split(". ")
            index = int(values[0])

            values = values[1]
            values = values.split(": ")

            feature_name = values[0]
            values = values[1].split(",")
            class_map = {}

            for value in values:
                class_relation = value.split("=")
                class_map[class_relation[1]] = class_relation[0]

            feature_map[index] = {
                "name": feature_name,
                "class_map": class_map
            }

        return feature_map


def build_pandas(feature_dict, data_file):
    with open(data_file, "r") as file:
        line = file.readline()
        data_dict = {}
        dict_index = 0
        while line:
            if "\n" in line:
                line = line.replace("\n", "")

            feature_values = line.split(",")
            frame_dict = {}
            for index, classification in zip(feature_dict, feature_values):
                feature = feature_dict[index]
                feature_name = feature["name"]
                frame_dict[feature_name] = feature["class_map"][classification]
            data_dict[dict_index] = frame_dict
            line = file.readline()
            dict_index += 1
        main_table = df.from_dict(data_dict).T
    return main_table

def get_overlap_amount(data_frame, label1, label2):
    unique_classes1 = set(data_frame[label1])
    unique_classes2 = set(data_frame[label2])
    combos = []
    amounts = []
    x = []
    y = []
    for classification1 in unique_classes1:
        for classification2 in unique_classes2:
            x.append(classification1)
            y.append(classification2)
            combos.append((classification1, classification2))
    for combo in combos:
        amount = len(data_frame.loc[(data_frame[label1] == combo[0]) & (data_frame[label2] == combo[1])])
        amounts.append(amount)
    return x, y, amounts

if __name__ == "__main__":

        label1 = "population"
        label2 = "habitat"
        feature_dict = build_feature_dict("data/cleaned_features.txt")
        data_frames = build_pandas(feature_dict, "data/agaricus-lepiota.data")
        x, y, amounts = get_overlap_amount(data_frames, label1, label2)
        # plt.scatter(x=x, y=y, s=amounts)
        # plt.show()
        dummies = pandas.get_dummies(data_frames)
        print(dummies.describe())

        plt.spy(dummies.T, markersize=0.02, c="red")
        fig = plt.gcf()
        fig.set_size_inches(100, 3)
        plt.plot()
        plt.savefig("space.png")
