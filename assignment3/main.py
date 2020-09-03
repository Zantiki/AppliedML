import assignment3.utils as util
import torch
import time
from assignment3.models import ConvoA, ConvoB, ConvoC


if __name__ == "__main__":
    config = {
        "convo_a": {
            "epochs": 10,
            "lr": 0.001
        }
    }
    x_train_b, y_train_b, x_test, y_test= util.get_data_mnist()
    convo_a = ConvoA()
    convo_a.cuda()
    start_time = time.time()
    util.train(convo_a, x_train_b, y_train_b, config["convo_a"]["lr"], config["convo_a"]["epochs"])
    training_time = time.time() - start_time
    device = torch.device("cpu")
    convo_a.to(device)
    print("Training time A: {}".format(training_time))
    print("Accuracy A:  {}%".format(convo_a.accuracy(x_test, y_test).numpy()))

    convo_b = ConvoB()

    start_time = time.time()
    convo_b.cuda()
    model = util.train(convo_b, x_train_b, y_train_b, config["convo_a"]["lr"], config["convo_a"]["epochs"])
    training_time = time.time() - start_time
    device = torch.device("cpu")
    convo_b.to(device)
    print("Training time B: {}".format(training_time))
    print("Accuracy B:  {}%".format(convo_b.accuracy(x_test, y_test).numpy()))


    convo_c = ConvoC()

    start_time = time.time()
    convo_c.cuda()
    util.train(convo_c, x_train_b, y_train_b, config["convo_a"]["lr"], config["convo_a"]["epochs"])
    training_time = time.time() - start_time

    device = torch.device("cpu")
    convo_c.to(device)
    print("Training time C: {}".format(training_time))
    print("Accuracy c:  {}%".format(convo_c.accuracy(x_test, y_test).numpy()))

    x_train_b, y_train_b, x_test, y_test = util.passion_for_fashion()
    convo_d = ConvoC()

    start_time = time.time()
    convo_d.cuda()
    util.train(convo_d, x_train_b, y_train_b, config["convo_a"]["lr"], config["convo_a"]["epochs"])
    training_time = time.time() - start_time

    device = torch.device("cpu")
    convo_d.to(device)
    print("Training time d: {}".format(training_time))
    print("Accuracy d:  {}%".format(convo_d.accuracy(x_test, y_test).numpy()))
