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

    def reset(self, batch_size):  # Reset states prior to new input sequence
        zero_state = torch.zeros(1, batch_size, self.state_size)  # Shape: (number of layers, batch size, state size)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        x = self.dense(out[-1].reshape(-1, 128))
        return x

    def f(self, x):  # x shape: (sequence length, batch size, encoding size)
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
        x = self.logits(x)
        return nn.functional.cross_entropy(x, y.argmax(1))
