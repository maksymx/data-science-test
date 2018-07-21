# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import nltk
import csv
from helper_functions import LINKS_RE, RETWEET_RE, ONLY_WORDS_RE

# Importing the dataset
tweets = pd.read_csv('tweets.txt', delimiter='\n', header=None, quoting=csv.QUOTE_NONE)
emoji = pd.read_csv('emoji.txt', delimiter='\n', header=None, quoting=csv.QUOTE_NONE)

# Cleaning the texts
corpus = []
stopword_set = set(stopwords.words('english'))
ps = PorterStemmer()

for i in range(0, len(tweets[0])):
    tweet = LINKS_RE.sub(' ', tweets[0][i])
    tweet = RETWEET_RE.sub(' ', tweet)
    tweet = ONLY_WORDS_RE.sub(' ', tweet)
    tweet = tweet.lower()
    tweet = tweet.split()
    tweet = [ps.stem(word) for word in tweet if not word in stopword_set]
    tweet = ' '.join(tweet)
    corpus.append(tweet)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()

le = preprocessing.LabelEncoder()
y = le.fit_transform(emoji)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
# msk = np.random.rand(len(X)) < 0.8
# X_train = X[msk]
# X_test = X[~msk]
# y_train = y[msk]
# y_test = y[~msk]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred, labels=['lol', 'lal'])