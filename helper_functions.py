import os
import pickle
import re
from urllib.request import urlretrieve

import numpy as np
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords

BASE_DIR = '.'
GLOVE_DIR = BASE_DIR + '/glove.6B/'

LINKS_RE = re.compile(r'https?:\/\/.*[\r\n]*', flags=re.MULTILINE)
RETWEET_RE = re.compile(r'RT', flags=re.MULTILINE)
PUNCTUATION_RE = re.compile(r'[#?@\._:!,=\-]+', flags=re.MULTILINE)

lemmatizer = WordNetLemmatizer()


def get_embeddings_index():
    pickle_file_name = 'glove.pickle'
    glove_file_name = 'glove.6B.100d.txt'
    pickle_file_path = os.path.join(GLOVE_DIR, pickle_file_name)
    glove_file_path = os.path.join(GLOVE_DIR, glove_file_name)
    glove_blob = 'https://worksheets.codalab.org/rest/bundles/0x15a09c8f74f94a20bec0b68a2e6703b3/contents/blob/'

    # first, build index mapping words in the embeddings set
    # to their embedding vector
    if os.path.exists(pickle_file_path):
        print('Loading word vectors.')
        with open(pickle_file_path, 'rb') as f2:
            embeddings_index = pickle.load(f2)
    else:
        if not os.path.exists(glove_file_path):
            print('Downloading word vectors text file.')
            urlretrieve(glove_blob, glove_file_path)

        print('Indexing word vectors.')
        embeddings_index = {}
        with open(glove_file_path) as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs

        print('Saving word vectors.')
        with open(pickle_file_path, 'wb') as f2:
            pickle.dump(embeddings_index, f2)

    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index


def transform_tweet(tweet):
    tweet = LINKS_RE.sub('', tweet)
    tweet = RETWEET_RE.sub('', tweet)
    tweet = PUNCTUATION_RE.sub('', tweet)
    tweet = tweet.strip()
    tweet = tweet.lower()
    tweet = ' '.join([t for t in tweet.split() if t not in stopwords])
    return tweet
