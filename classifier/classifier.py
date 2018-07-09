from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import nltk
from nltk.corpus import stopwords
from NL import text_helpers
from sklearn.neural_network import MLPClassifier
import csv
import os
from zipfile import ZipFile
import requests
import io

X = []
y = []

# Data Read
dirname = '...'
filenames = os.listdir(dirname)
filenames.sort()

for filename in filenames:
    full_filename = os.path.join(dirname, filename)
    data = open(full_filename, 'r', errors='ignore').readlines()

    for data_ in data:
        X.append(data_)
        y.append(filename)


# USE tf-idf
max_features = 1000
def tokenizer(text):
    words = nltk.tokenize(text)
    return words

tfidf = TfidfVectorizer(tokenizer=tokenizer, stop_words = 'english', max_features = max_features)
text_helpers.normalize_text(X, stops=stops) # Spam = 1 , not spam = 0
X = text_helpers.normalize_text()

# how much you split
SPLIT_PERC = 0.7
train_indices = np.sort(np.random.choice(len(y), round(SPLI)))

