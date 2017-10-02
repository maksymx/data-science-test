import string

import numpy as np
import pandas as pd
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

punctuations = string.punctuation

# set parameters:
max_features = 5000
maxlen = 50
batch_size = 128
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 2
MAX_NB_WORDS = 20000


def prepare_labels():
    emoji = pd.read_csv('emoji.txt', sep="\n", header=None)
    emoji.columns = ["icons"]
    emoji.index += 1
    return emoji


def transform_labels(emoji):
    for i, name in ((0, 'heart_eyes'), (1, 'yum'), (2, 'sob'), (3, 'blush'), (4, 'weary'),
                    (5, 'smirk'), (6, 'grin'), (7, 'flushed'), (8, 'relaxed'), (9, 'wink')):
        emoji['icons'].replace(name, i, inplace=True)
    emoji.reindex()
    return emoji


def prepare_features():
    with open('tweets.txt') as f:
        tweets = f.readlines()
    return tweets


stoplist = set('for a of the and to in'.split())

emoji = prepare_labels()
print(emoji.icons.unique())
emoji = transform_labels(emoji)

tweets = prepare_features()

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(tweets)
sequences = tokenizer.texts_to_sequences(tweets)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

sequences_array = np.array(list(map(lambda x: np.array(x), sequences)))

max_len = max(len(a) for a in sequences_array)

data = pad_sequences(sequences, maxlen=maxlen)

inputs_train, inputs_test, expected_output_train, expected_output_test = train_test_split(data,
                                                                                          emoji.as_matrix())  # matched OK

model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
model.add(Embedding(max_features,
                    embedding_dims,
                    input_length=maxlen))
model.add(Dropout(0.2))

# we add a Convolution1D, which will learn filters
# word group filters of size filter_length:
model.add(Conv1D(filters,
                 kernel_size,
                 padding='valid',
                 activation='relu',
                 strides=1))
# we use max pooling:
model.add(GlobalMaxPooling1D())

# We add a vanilla hidden layer:
model.add(Dense(hidden_dims))
model.add(Dropout(0.2))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(inputs_train, expected_output_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(inputs_test, expected_output_test))
