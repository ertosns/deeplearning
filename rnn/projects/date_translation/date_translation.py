from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np
import random
from tqdm import tqdm
from babel.dates import format_date
from nmt_utils import *
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import os
import time

m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)

print('dataset size: {}'.format(len(dataset)))
print('dataset 0: {}'.format(dataset[0]))
#human date vocabulary
print('human vocab size: {}'.format(len(human_vocab)))
print('human vocab 0: {}'.format(human_vocab))
#dictionary map from machine chars to index
print('machine vocab size: {}'.format(len(machine_vocab)))
print('machine vocab 0: {}'.format(machine_vocab))
#dictionary map from index to corresponding machine vocab char
print('inv machine vocab size: {}'.format(len(inv_machine_vocab)))
print('inv machine vocab 0: {}'.format(inv_machine_vocab))


print('')
Tx = 30
Ty = 10

# X is the human readable dates with length Tx converted to vector of chars poistions in the vocab
# Y is the machine dates with legnth Ty converted to vector of chars positions in the vocab
# Xoh is one-hot vector of X
# Yoh is one-hot vector of Y
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor1 = Dense(10, activation = "tanh")
densor2 = Dense(1, activation = "relu")
activator = Activation(softmax, name='attention_weights')
dotor = Dot(axes = 1)

n_a = 32 # number of units for the pre-attention, bi-directional LSTM's hidden state 'a'
n_s = 64 # number of units for the post-attention LSTM's hidden state "s"
post_activation_LSTM_cell = LSTM(n_s, return_state = True) # post-attention LSTM
output_layer = Dense(len(machine_vocab), activation=softmax)


def one_step_attention(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.

    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)

    Returns:
    context -- context vector, input of the next (post-attention) LSTM cell
    """
    # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a"
    s_prev = repeator(s_prev)
    # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)
    # For grading purposes, please list 'a' first and 's_prev' second, in this order.
    concat = concatenator([a, s_prev])
    # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e.
    e = densor1(concat)
    # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies.
    energies = densor2(e)
    # Use "activator" on "energies" to compute the attention weights "alphas"
    alphas = activator(energies)
    # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell
    context = dotor([alphas, a])

    return context


def model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"

    Returns:
    model -- Keras model instance
    """
    print("our mode is of shape: {}X{}".format( Tx, human_vocab_size))
    X = Input(shape=(Tx, human_vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0
    outputs = []
    a = Bidirectional(LSTM(units=n_a, return_sequences=True))(X)
    for t in range(Ty):
        context = one_step_attention(a, s)
        s, _, c = post_activation_LSTM_cell(inputs=context, initial_state=[s, c])
        out = output_layer(inputs=s)
        outputs.append(out)

    model = Model(inputs=[X, s0, c0], outputs=outputs)
    return model

model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
optimizer = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Yoh.swapaxes(0,1))


parser = ArgumentParser()
parser.add_argument('-t', '--train', type=bool, default=False)
parser.add_argument('-i', '--input', type=str, default='input.txt')

args = parser.parse_args()
train=args.train
input=args.input
WEIGHTS='weights.h5'

print('Xoh shape: {}'.format(Xoh.shape))
if train:
    model.fit([Xoh, s0, c0], outputs, epochs=500, batch_size=100)
    model.save_weights(WEIGHTS)
else:
    if not os.path.exists(WEIGHTS):
        print("weight doesn't exist!")
        exit()
    model.load_weights(WEIGHTS)
    with open('input.txt') as f:
        for line in [dt[:-1] for dt in f.readlines()]: # trim \n
            print(f'line: {line}')
            # padded source
            source = string_to_int(line, Tx, human_vocab)
            print(f'padded source: {source}')
            # convert each index to one-hot-vector
            source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source))).swapaxes(0,1)
            print(f'source to categorical: {source}')
            #TODO (res)
            rr=list(source.shape)
            rr.reverse()
            source=source.reshape([1]+rr)
            m = 1
            s0 = np.zeros((m, n_s))
            c0 = np.zeros((m, n_s))
            prediction = model.predict([source, s0, c0])
            prediction = np.argmax(prediction, axis = -1)
            output = [inv_machine_vocab[int(i)] for i in prediction]
            print('original {} translation {}'.format(''.join(line), ''.join(output)))
            #attention_map = plot_attention_map(model, human_vocab, inv_machine_vocab, "Tuesday 09 Oct 1993", num = 7, n_s = 64);
            #time.sleep(10)
