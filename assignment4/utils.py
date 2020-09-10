import torch


def train1(model, x_train, y_train, char_encodings, index_to_char, learning_rate, epochs):
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

def train2(model, x_train, y_train, char_encodings, index_to_char, emojis, learning_rate, epochs):
    optimizer = torch.optim.RMSprop(model.parameters(), learning_rate)
    print("x_train len: ", len(x_train[0][0]))
    # print(x_train)
    for epoch in range(epochs):
        model.reset()
        model.loss(x_train, y_train).backward()
        optimizer.step()
        optimizer.zero_grad()
        if epoch % 10 == 0:
            model.reset()
            print("EPOCH:", epoch)
            tests = ['hat ', 'matt', "son ", "rat "]
            tests2 = ['matt', 'rat ', 'cap ', 'son ', 'hat ', 'cat ', 'flat']

            for test in tests:
                test_encoding = encode_word(test, char_encodings, index_to_char)
                # print(test_encoding)
                test_tensor = torch.tensor([test_encoding])
                y = model.f(test_tensor)
                index = y.argmax(1).numpy()[0]

                print(test, emojis[index])

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

def encode_word(word, char_encodings, chars):

    word_matrix = []
    for char in word:
        char_index = chars.index(char)
        word_matrix.append(char_encodings[char_index])
    return word_matrix


def gen_mo_data():
    emojies = ['🤓', '🐀️', '🧢️', '👶️', '🎩️' , '🐈️', '🏢️' ]
    # , '🐀️', '🧢️', '👶️', '🎩️' , '🐈️', '🏢️', '🤬'
    # , 'rat ', 'cap ', 'son ', 'hat ', 'cat ', 'flat', 'brat'
    words = ['matt', 'rat ', 'cap ', 'sony', 'hat ', 'cat ', 'flat']
    chars = [c for c in ''.join(set(''.join(words)))]

    print(chars)
    char_encodings = []
    emoji_encodings = []
    for i in range(len(chars)):
        char_encodings.append([0.0 for y in range(len(chars))])
        char_encodings[i][i] = 1.0

    char_index = 0
    for i in range(len(emojies)):
        emoji_encodings.append([0.0 for y in range(len(emojies))])
        emoji_encodings[i][i] = 1.0
        char_index += 1

    word_encodings = []

    for word in words:
        word_encodings.append(encode_word(word, char_encodings, chars))

    y_train_array = emoji_encodings
    x_train_array = word_encodings

    # for word in word_encodings:
        # index_of_char = chars.index(word)
        # x_train_array.append([char_encodings[index_of_char]])

    # for char in emojies:
       # index_of_char = chars.index(char)
       # y_train_array.append(char_encodings[index_of_char])


    y_train = torch.tensor(y_train_array)
    x_train = torch.tensor(x_train_array)
    return x_train, y_train, char_encodings, chars, emojies