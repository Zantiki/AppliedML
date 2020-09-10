import assignment3.utils as util
import torch
import time
from assignment4.models import mmLTSM, moLSTSM
from assignment4.utils import gen_mm_data, gen_mo_data, train1, train2


if __name__ == "__main__":
    """x_train, y_train, char_encodings, chars = gen_mm_data()
    model1 = mmLTSM(len(char_encodings))
    train1(model1, x_train, y_train, char_encodings, chars, 0.001, 500)"""

    x_train, y_train, char_encodings, chars, emojis = gen_mo_data()
    print(chars)
    model1 = moLSTSM()
    train2(model1, x_train, y_train, char_encodings, chars, emojis, 0.001, 300)