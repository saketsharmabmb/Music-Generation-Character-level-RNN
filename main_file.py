import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation, SimpleRNN, LSTM
from keras.callbacks import History
from keras import backend as K


def sample(preds, temperature):
    # Helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


data_str = (open(r"data/input.txt")).read()
print(len(data_str))
data = [i for i in data_str]
data_set = sorted(set(data_str))
print("Length of vocabulary = ", len(data_set))
char_2_idx = {ch: i for i, ch in enumerate(data_set)}
idx_2_char = {i: ch for i, ch in enumerate(data_set)}

data = np.array([char_2_idx[i] for i in data_str])
train_data = data[0:int(0.8 * len(data))]
y_train = data[1:int(0.8 * len(data)) + 1]

train_data = to_categorical(train_data, len(data_set))
val_data = data[int(0.8 * len(data)):-1]
y_val = data[int(0.8 * len(data)) + 1:]

# Preparing data batches
batch_size = 25
length_to_keep = int(len(train_data) / batch_size) * len(data_set)*batch_size
train_data = train_data[0:int(length_to_keep / len(data_set)), :].copy()
train_data = np.reshape(train_data, (int(len(train_data) / batch_size), batch_size, len(data_set)))

X = train_data[:, :-1, :]
y = train_data[:, 1:, :]
X_val = to_categorical(val_data, len(data_set))
length_to_keep = int(len(X_val) / batch_size) * len(data_set)* batch_size
X_val = X_val[0:int(length_to_keep / len(data_set)), :].copy()
X_val = np.reshape(X_val, (int(len(X_val) / batch_size), batch_size, len(data_set)))
y_val = X_val[:, 1:, :]
X_val = X_val[:, :-1, :]

# RNN Model
epochs = 100
vocab_size = len(data_set)
input_dim = vocab_size
output_dim = vocab_size
hidden_dim = 100

rnn_model = Sequential()
rnn_model.add(LSTM(hidden_dim, input_shape=(None, vocab_size)))
# rnn_model.add(SimpleRNN(hidden_dim, activation='tanh', return_sequences=True, input_shape=(None, vocab_size)))
rnn_model.add(Dense(output_dim))
rnn_model.add(Activation('softmax'))
rnn_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
rnn_model.summary()
print('Training')
modelhistory = History()
history = rnn_model.fit(X, y[:, -1, :], batch_size=128, nb_epoch=epochs, validation_data=(X_val, y_val[:, -1, :]))

# Function to get rnn layer output
get_rnn_layer_output = K.function([rnn_model.layers[0].input], [rnn_model.layers[0].output])
prime_len = 25
gen_len = 900
start_index = 0
d = 0
rnn_activations = []
# T is Temperature parameter for Softmax layer.
for T in [1.0]:
    d += 1
    generated = ''
    sentence = data_str[start_index: start_index + prime_len]
    generated += sentence
    print('Generating with seed: "' + sentence + '"')

    for i in range(gen_len):
        x = np.zeros((1, prime_len, len(data_set)))
        for t, char in enumerate(sentence):
            x[0, t, char_2_idx[char]] = 1.

        preds = rnn_model.predict(x, verbose=0)[0]
        layer_output = get_rnn_layer_output([x])[0]
        rnn_activations.append(layer_output[0][-1])
        next_index = sample(preds, T)
        next_char = idx_2_char[next_index]

        generated += next_char
        sentence = sentence[1:] + next_char

    f = open('pred_feature' + '_' + str(T) + '_' + str(d) + '.txt', 'w')
    f.write(generated)
    f.close()
    rnn_activations = np.array(rnn_activations)
    print(rnn_activations.shape)
    np.savetxt('rnn_activations_pred', rnn_activations, delimiter=',')
