import assignment3.utils as util
import torch
import time
from assignment3.models import ConvoA, ConvoB, ConvoC, ConvoD


if __name__ == "__main__":
    config = {
        "convo_a": {
            "epochs": 3,
            "lr": 0.001
        }
    }
    x_train_b, y_train_b, x_test, y_test= util.get_data_mnist()
    convo_a = ConvoA()

    start_time = time.time()
    # model = util.train(convo_a, x_train_b, y_train_b, config["convo_a"]["lr"], config["convo_a"]["epochs"])
    training_time = time.time() - start_time
    print("Training time A: {}".format(training_time))
    print("Accuracy A:  {}%".format(convo_a.accuracy(x_test, y_test).numpy()))

    convo_b = ConvoB()

    start_time = time.time()
    # model = util.train(convo_b, x_train_b, y_train_b, config["convo_a"]["lr"], config["convo_a"]["epochs"])
    training_time = time.time() - start_time
    print("Training time B: {}".format(training_time))
    print("Accuracy B:  {}%".format(convo_b.accuracy(x_test, y_test).numpy()))


    # Todo: hei Jakob, denne linjen bruker cuda.
    convo_c = ConvoC()

    start_time = time.time()
    convo_c.cuda()
    util.train(convo_c, x_train_b, y_train_b, config["convo_a"]["lr"], config["convo_a"]["epochs"])
    training_time = time.time() - start_time

    device = torch.device("cpu")
    convo_c.to(device)
    print("Training time C: {}".format(training_time))
    print("Accuracy c:  {}%".format(convo_c.accuracy(x_test, y_test).numpy()))