# https://www.geeksforgeeks.org/ml-text-generation-using-gated-recurrent-unit-networks/

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import GRU

from keras.optimizers import RMSprop

from keras.callbacks import LambdaCallback
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
import random
import sys

# 2 : loading data
with open('poems.txt', 'r') as file:
    text = file.read()

# print(text)

# 3 : Creating a mapping from each unique character in the text to a unique number
vocabulary = sorted(list(set(text)))

char_to_indices = dict((c, i) for i, c in enumerate(vocabulary))
indices_to_char = dict((i, c) for i, c in enumerate(vocabulary))

print(vocabulary)

# 4 : pre-processing the data
max_length = 100
steps = 5
sentences = []
next_chars = []
for i in range(0, len(text) - max_length, steps):
    sentences.append(text[i: i + max_length])
    next_chars.append(text[i + max_length])

# hot encoding each caracter into a boolean vector

# initializing a matrix of boolean vectors with each column representing the hot
# encoded representation of the caracter
X = np.zeros((len(sentences), max_length, len(vocabulary)), dtype = np.bool)
y = np.zeros((len(sentences), len(vocabulary)), dtype = np.bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_to_indices[char]] = 1
    y[i, char_to_indices[next_chars[i]]] = 1

# building the GRU network
model = Sequential()
model.add(GRU(128, input_shape = (max_length, len(vocabulary))))
model.add(Dense(len(vocabulary)))
model.add(Activation('softmax'))
optimizer = RMSprop(lr = 0.01)
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer)

#6 : helper functions
# Helper function to sample an index from a probability array
def sample_index(preds, temperature = 1.0):
# temperature determines the freedom the function has when generating text
  
    # Converting the predictions vector into a numpy array
    preds = np.asarray(preds).astype('float64')
  
    # Normalizing the predicitons array
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
  
    # The main sampling step. Creates an array of probablities signifying
    # the probability of each character to be the next character in the 
    # generated text
    probas = np.random.multinomial(1, preds, 1)
  
    # Returning the character with maximum probability to be the next character
    # in the generated text
    return np.argmax(probas)


# Helper function to generate text after the end of each epoch
def on_epoch_end(epoch, logs):
    print()
    print('----- Generating text after Epoch: % d' % epoch)

    if epoch < 60:
        return
    
    # Choosing a random starting index for the text generation
    start_index = random.randint(0, len(text) - max_length - 1)
  
    # Sampling for different values of diversity
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)
  
        generated = ''
  
        # Seed sentence
        sentence = text[start_index: start_index + max_length]
  
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)
  
        for i in range(400):
            # Initializing the predictions vector
            x_pred = np.zeros((1, max_length, len(vocabulary)))
  
            for t, char in enumerate(sentence):
                x_pred[0, t, char_to_indices[char]] = 1.
  
            # Making the predictions for the next character
            preds = model.predict(x_pred, verbose = 0)[0]
  
            # Getting the index of the most probable next character
            next_index = sample_index(preds, diversity)
  
            # Getting the most probable next character using the mapping built
            next_char = indices_to_char[next_index]
  
            # Building the generated text
            generated += next_char
            sentence = sentence[1:] + next_char
  
            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()


# Defining a custom callback function to 
# describe the internal states of the network
print_callback = LambdaCallback(on_epoch_end = on_epoch_end)

# Defining a helper function to save the model after each epoch
# in which the loss decreases
filepath = "weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor ='loss',
                             verbose = 1, save_best_only = True,
                             mode ='min')

# Defining a helper function to reduce the learning rate each time
# the learning plateaus
reduce_alpha = ReduceLROnPlateau(monitor ='loss', factor = 0.2,
                              patience = 1, min_lr = 0.001)
callbacks = [print_callback, checkpoint, reduce_alpha]

#7 : training the GRU model
# Training the GRU model
model.fit(X, y, batch_size = 128, epochs = 100, callbacks = callbacks)

def generate_text(length, diversity):
    # Get random starting text
    start_index = random.randint(0, len(text) - max_length - 1)
  
    # Defining the generated text
    generated = ''
    sentence = text[start_index: start_index + max_length]
    generated += sentence
  
    # Generating new text of given length
    for i in range(length):
  
            # Initializing the predicition vector
            x_pred = np.zeros((1, max_length, len(vocabulary)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_to_indices[char]] = 1.
  
            # Making the predicitons
            preds = model.predict(x_pred, verbose = 0)[0]
  
            # Getting the index of the next most probable index
            next_index = sample_index(preds, diversity)
  
            # Getting the most probable next character using the mapping built
            next_char = indices_to_char[next_index]
  
            # Generating new text
            generated += next_char
            sentence = sentence[1:] + next_char
    return generated
  
print(generate_text(500, 0.2))
