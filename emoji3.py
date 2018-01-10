import string
from collections import defaultdict
import itertools
import numpy as np
import pandas as pd
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation
from nltk.stem.snowball import SnowballStemmer
from nltk import ngrams
from sklearn.base import TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS as stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from nltk.classify.naivebayes import NaiveBayesClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from spacy.en import English
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt

from helper_functions import transform_tweet

parser = English()
punctuations = string.punctuation


class StemmedCountVectorizer(CountVectorizer):
    stemmer = SnowballStemmer("english", ignore_stopwords=True)

    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()

        def func(doc):
            for w in analyzer(doc):
                stemmed = self.stemmer.stem(w)
                yield stemmed

        return func


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
    tweets = []
    with open('tweets.txt') as f:
        for line in f.readlines():
            tweet = transform_tweet(line)
            tweets.append(tweet)
    return tweets


def classifier(X, y, model_class, shuffle=True, n_folds=10, **kwargs):
    stratified_k_fold = cross_validation.StratifiedShuffleSplit(y, test_size=0.2, random_state=30)
    y_pred = y.copy()
    for ii, jj in stratified_k_fold:
        X_train, X_test = X[ii], X[jj]
        y_train = y[ii]
        #         print y_train.count()
        #         print (y[y == 1].count())
        #         print (y[y == 0].count())
        clf = model_class(**kwargs)
        clf.fit(X_train, y_train)
        y_pred[jj] = clf.predict(X_test)
    return y_pred


def n_gramize_features_labels(features, labels):
    inputs, outputs = [], []
    for (inp, outp) in zip(features, labels):
        tri_grams = ngrams(inp.split(), 4)
        for gram in tri_grams:
            inputs.append(gram)
            outputs.append(outp)
    return inputs, outputs


def my_tokenizer(sentence):
    stoplist = set('for a of the and to in'.split())
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


if __name__ == '__main__':
    emoji = prepare_labels()
    print(emoji.icons.unique())
    emoji = transform_labels(emoji)
    print(emoji.icons.value_counts())

    tweets = prepare_features()

    inputs_train, inputs_test, \
    expected_output_train, expected_output_test = train_test_split(tweets, emoji.as_matrix())

    # count_vect = CountVectorizer()
    # X_train_counts = count_vect.fit_transform(inputs_train)
    # print(X_train_counts.shape)
    #
    # tfidf_transformer = TfidfTransformer()
    # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    # print(X_train_tfidf.shape)
    #
    # clf = MultinomialNB().fit(X_train_tfidf, expected_output_train)

    # text_clf = Pipeline([
    #     ('vect', CountVectorizer()),
    #     ('tfidf', TfidfTransformer()),
    #     # ('clf', MultinomialNB()),
    #     ('clf-svm', SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)),
    # ])

    # vectorizer = CountVectorizer(tokenizer=spacy_tokenizer, ngram_range=(3, 3))
    #
    # classifier = LinearSVC()
    # classifier2 = SVC()
    # classifier3 = RandomForestClassifier()
    # classifier4 = MultinomialNB()
    #
    # pipe = Pipeline([('cleaner', predictors()),
    #                  ('vectorizer', vectorizer),
    #                  ('classifier', classifier4)])
    #
    # # Create model and measure accuracy
    # pipe.fit(inputs_train, expected_output_train)

    # # now we can save it to a file
    # joblib.dump(pipe, 'model.pkl')
    #
    # pred_data = pipe.predict(inputs_test)
    #
    # for (sample, pred) in zip(inputs_test, pred_data):
    #     print(sample, ">>>>>", pred)
    #
    # print("Accuracy:", accuracy_score(expected_output_test, pred_data))

    # svc = confusion_matrix(expected_output_train, classifier(inputs_train, expected_output_train, SVC))
    # print(svc)

    stemmed_count_vect = StemmedCountVectorizer(stop_words='english')
    text_clf = Pipeline([('vect', stemmed_count_vect),
                         ('tfidf', TfidfTransformer()),
                         # ('mnb', MultinomialNB(fit_prior=False)),
                         # ('classifier', RandomForestClassifier()),
                         ('classifier', LinearSVC()),
                         ])

    text_clf.fit(inputs_train, expected_output_train)

    # now we can save it to a file
    joblib.dump(text_clf, 'model.pkl')

    pred_data = text_clf.predict(inputs_test)

    for (sample, pred) in zip(inputs_test, pred_data):
        print(sample, ">>>>>", pred)

    print("Accuracy:", accuracy_score(expected_output_test, pred_data))
