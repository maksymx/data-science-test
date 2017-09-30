import string
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords
from sklearn.model_selection import train_test_split
# from spacy.en import English
# from gensim.models import Word2Vec

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.externals import joblib
# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics import accuracy_score
# from sklearn.pipeline import Pipeline
# from sklearn.svm import SVC, LinearSVC

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

# parser = English()
punctuations = string.punctuation

# set parameters:
max_features = 5000
# maxlen = 150
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


def my_tokenizer(sentence):
    tokens = [word for word in sentence.lower().split() if word not in stoplist]
    # remove words that appear only once
    frequency = defaultdict(int)
    for token in tokens:
        frequency[token] += 1
    return [token for token in tokens if frequency[token] > 1]


# Create spacy tokenizer that parses a sentence and generates tokens
# these can also be replaced by word vectors

# def spacy_tokenizer(sentence):
#     tokens = parser(sentence)
#     tokens = [tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_ for tok in tokens]
#     tokens = [tok for tok in tokens if (tok not in stopwords and tok not in punctuations)]
#     return tokens


# Basic utility function to clean the text
def clean_text(text):
    return text.strip().lower()


# Custom transformer using spaCy
class predictors(TransformerMixin):
    def transform(self, X, **transform_params):
        return [clean_text(text) for text in X]

    def fit(self, X, y=None, **fit_params):
        return self

    def get_params(self, deep=True):
        return {}


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

maxlen = 50

data = pad_sequences(sequences, maxlen=maxlen)

inputs_train, inputs_test, expected_output_train, expected_output_test = train_test_split(data, emoji.as_matrix())  # matched OK

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
##########################

# vectorizer = CountVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1, 3))

# classifier = LinearSVC()
# classifier2 = SVC()
# classifier3 = RandomForestClassifier()
#
# pipe = Pipeline([('cleaner', predictors()),
#                  ('vectorizer', vectorizer),
#                  ('classifier', classifier)])
#
# # Create model and measure accuracy
# pipe.fit(inputs_train, expected_output_train)
#
# # now we can save it to a file
# joblib.dump(pipe, 'model.pkl')
#
# pred_data = pipe.predict(inputs_test)
#
# for (sample, pred) in zip(inputs_test, pred_data):
#     print(sample, ">>>>>", pred)
#
# print("Accuracy:", accuracy_score(expected_output_test, pred_data))

