import string
from collections import defaultdict

import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from spacy.en import English

# from gensim.models import Word2Vec

parser = English()
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

def spacy_tokenizer(sentence):
    tokens = parser(sentence)
    tokens = [tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_ for tok in tokens]
    tokens = [tok for tok in tokens if (tok not in stopwords and tok not in punctuations)]
    return tokens


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

inputs_train, inputs_test, expected_output_train, expected_output_test = train_test_split(tweets,
                                                                                          emoji.as_matrix())  # matched OK

vectorizer = CountVectorizer(tokenizer=spacy_tokenizer, ngram_range=(1, 3))

classifier = LinearSVC()
classifier2 = SVC()
classifier3 = RandomForestClassifier()

pipe = Pipeline([('cleaner', predictors()),
                 ('vectorizer', vectorizer),
                 ('classifier', classifier)])

# Create model and measure accuracy
pipe.fit(inputs_train, expected_output_train)

# now we can save it to a file
joblib.dump(pipe, 'model.pkl')

pred_data = pipe.predict(inputs_test)

for (sample, pred) in zip(inputs_test, pred_data):
    print(sample, ">>>>>", pred)

print("Accuracy:", accuracy_score(expected_output_test, pred_data))
