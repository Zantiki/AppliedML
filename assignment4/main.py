import assignment3.utils as util
import torch
import time
from assignment4.models import mmLTSM, moLSTSM
from assignment4.utils import gen_mm_data, train


if __name__ == "__main__":
    x_train, y_train, char_encodings, chars = gen_mm_data()
    model1 = mmLTSM(len(char_encodings))
    train(model1, x_train, y_train, char_encodings, chars, 0.001, 500)