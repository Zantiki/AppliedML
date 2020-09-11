import torch
import torch.nn as nn


class mmLTSM(nn.Module):

    def __init__(self, encoding_size):
        super(mmLTSM, self).__init__()

        self.lstm = nn.LSTM(encoding_size, 128)  # 128 is the state size
        self.dense = nn.Linear(128, encoding_size)  # 128 is the state size

    def reset(self):  # Reset states prior to new input sequence
        zero_state = torch.zeros(1, 1, 128)  # Shape: (number of layers, batch size, state size)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        return self.dense(out.reshape(-1, 128))

    def f(self, x):  # x shape: (sequence length, batch size, encoding size)
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x,
             y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
        x = self.logits(x)
        print("x_after_logits", x.shape)
        print(x)
        print("y_in_loss", y.shape)
        return nn.functional.cross_entropy(x, y.argmax(1))


class moLSTSM(nn.Module):

    # batch_size = 4
    state_size = 128
    state_size2 = 14
    encoding_text = 14
    encoding_emoji = 7

    def __init__(self):
        super(moLSTSM, self).__init__()
        self.lstm = nn.LSTM(self.encoding_text, self.state_size)  # 128 is the state size
        self.dense = nn.Linear(self.state_size, self.encoding_emoji)
        # self.dense2 = nn.Linear(14, self.encoding_emoji)

    def reset(self, batch_size):  # Reset states prior to new input sequence
        zero_state = torch.zeros(1, batch_size, self.state_size)  # Shape: (number of layers, batch size, state size)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        # print("out len logits: ", len(out), len(out[0]))
        # out.view(seq_len, batch, num_directions, hidden_size)
        # return self.dense(out[:, -1, :].reshape(-1, 128))
        # print(out[:, -1, :])5
        # print(out.detach().numpy().shape)
        # print("x_before_dense: ", x.shape)
        # print("x_after_ltsm", out.shape)
        # x = self.dense(out).reshape(len(out), 28)#.reshape(7, 28)
        # x = x.reshape(7, 28*2)
        x = self.dense(out[-1].reshape(-1, 128))
        # x = x.reshape()
        return x
        # return out

    def f(self, x):  # x shape: (sequence length, batch size, encoding size)
        # print(x)
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
        # print(self.logits(x))
        # print("x_loss: ", len(self.logits(x)), len(self.logits(x)[0]))
        # print("y_loss: ", len(y))
        # print(y.argmax(1), y[0])
        x = self.logits(x)
        # x = torch.transpose(x, 1, 0)
        # print("x_after_logits", x.shape)
        # print("y_in_loss", y)
        # exit(0)
        return nn.functional.cross_entropy(x, y.argmax(1))
