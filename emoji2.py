import re

import numpy as np
import pandas as pd
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.layers import Dense, Input, Flatten
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords

from helper_functions import get_embeddings_index

BASE_DIR = '.'
LINKS_RE = re.compile(r'https?:\/\/.*[\r\n]*', flags=re.MULTILINE)
TEXT_DATA_DIR = BASE_DIR + '/20_newsgroup/'
MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2


def prepare_labels():
    emoji = pd.read_csv('emoji.txt', sep="\n", header=None)
    emoji.columns = ["icons"]
    emoji.index += 1
    return emoji


def transform_labels(emoji):
    for i, name in ((0, 'heart_eyes'), (1, 'yum'), (2, 'sob'), (3, 'blush'), (4, 'weary'),
                    (5, 'smirk'), (6, 'grin'), (7, 'flushed'), (8, 'relaxed'), (9, 'wink')):
        emoji['icons'].replace(name, i, inplace=True)
    return emoji


def prepare_features():
    lemmatizer = WordNetLemmatizer()
    tweets = []
    with open('tweets.txt') as f:
        for line in f.readlines():
            line2 = LINKS_RE.sub('', line)
            line3 = line2.strip()
            line4 = [lemmatizer.lemmatize(t.lower()) for t in line3.split() if t.lower() not in stopwords]
            tweets.append(' '.join(line4))
    return tweets


if __name__ == '__main__':
    emoji = prepare_labels()
    unique_emojis = emoji.icons.unique()
    print(unique_emojis)
    emoji = transform_labels(emoji)
    tweets = prepare_features()

    embeddings_index = get_embeddings_index()

    # second, prepare text samples and their labels
    print('Processing text dataset')

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(tweets)
    sequences = tokenizer.texts_to_sequences(tweets)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    # TODO: remove stopwords
    # TODO: try tf-idf
    # TODO: https://stackoverflow.com/questions/33536182/testing-the-keras-sentiment-classification-with-model-predict
    # TODO:

    labels = to_categorical(np.asarray(emoji))
    print('Shape of data tensor:', data.shape)
    print('Shape of label tensor:', labels.shape)

    # split the data into a training set and a validation set
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_val = data[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]

    print('Preparing embedding matrix.')

    # prepare embedding matrix
    num_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            if len(embedding_matrix[i]) == len(embedding_vector):
                embedding_matrix[i] = embedding_vector
            else:
                print(len(embedding_matrix[i]), ">>>>>>>", len(embedding_vector))
                embedding_matrix[i] = np.resize(embedding_vector, (1, len(embedding_matrix[i])))

    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                # trainable=False
                                )

    print('Training model.')
    # import pdb; pdb.set_trace()
    # train a 1D convnet with global maxpooling
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = Conv1D(128, 5, activation='relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(5)(x)
    x = Conv1D(128, 5, activation='relu')(x)
    x = MaxPooling1D(35)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    preds = Dense(len(unique_emojis), activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    model.fit(x_train, y_train,
              # batch_size=256,
              batch_size=128,
              epochs=5,
              validation_data=(x_val, y_val))

    model.save('my_model.h5')
