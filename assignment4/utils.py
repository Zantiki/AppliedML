import torch


def train(model, x_train, y_train, char_encodings, index_to_char, learning_rate, epochs):
    optimizer = torch.optim.RMSprop(model.parameters(), learning_rate)
    for epoch in range(epochs):
        model.reset()
        model.loss(x_train, y_train).backward()
        optimizer.step()
        optimizer.zero_grad()

        if epoch % 10 == 9:
            # Generate characters from the initial characters ' h'
            model.reset()
            text = ' h'
            model.f(torch.tensor([[char_encodings[0]]]))
            y = model.f(torch.tensor([[char_encodings[1]]]))
            text += index_to_char[y.argmax(1)]
            for c in range(50):
                y = model.f(torch.tensor([[char_encodings[y.argmax(1)]]]))
                text += index_to_char[y.argmax(1)]
            print(text)


def gen_mm_data():
    hello_x = list(" hello world")
    hello_y = list("hello world ")
    chars = [' ', 'h', 'e', 'l', 'o', 'w', 'r', 'd']
    char_encodings = []
    for i in range(len(chars)):
        char_encodings.append([0.0 for y in range(len(chars))])
        char_encodings[i][i] = 1.0

    y_train_array = []
    x_train_array = []

    for char in hello_x:
        index_of_char = chars.index(char)
        x_train_array.append([char_encodings[index_of_char]])

    for char in hello_y:
        index_of_char = chars.index(char)
        y_train_array.append(char_encodings[index_of_char])

    print(x_train_array)
    print(y_train_array)

    x_train = torch.tensor(x_train_array)
    y_train = torch.tensor(y_train_array)

    return x_train, y_train, char_encodings, chars


def gen_mo_data():
    pass