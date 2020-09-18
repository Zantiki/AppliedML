import assignment3.utils as util
import torch
import time
from assignment4.models import mmLTSM, moLSTSM
from assignment4.utils import gen_mm_data, gen_mo_data, train1, train2


if __name__ == "__main__":
    x_train, y_train, char_encodings, chars = gen_mm_data()
    print(x_train.shape)
    print(y_train.shape)
    model1 = mmLTSM(len(char_encodings))
    train1(model1, x_train, y_train, char_encodings, chars, 0.001, 500)

    x_train, y_train, char_encodings, chars, emojis = gen_mo_data()
    # x_train = x_train.reshape(4, 7, 14)
    model1 = moLSTSM()
    x_train = torch.transpose(x_train, 0, 1)
    train2(model1, x_train, y_train, char_encodings, chars, emojis, 0.001, 500)