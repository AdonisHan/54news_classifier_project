from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import nltk
from mlxtend.plotting import plot_decision_regions
from nltk.corpus import stopwords

def tokenizer(text):
    words = nltk.word_tokenize(text)
    return words

stops = list(set(stopwords.words('english')))
self_stop_words = ['PL', 'non_PL',
                   'huhuh', 'um', 'ahm', 'maam', 'ok', 'hello', 'yeah', 'im', 'right', 'huh', 'yeap', 'dot', 'uh'
                   'ahi', 'thank', 'hi', 'yes',
                   'zero','one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
                   'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'
                   ]
stop_words = stops + self_stop_words

max_features = 200

clf_1 = Pipeline([('vect', CountVectorizer()), ('clf', MultinomialNB())])
clf_2 = Pipeline([('vect', HashingVectorizer(non_negative=True)), ('clf', MultinomialNB())])
clf_3 = Pipeline([('vect', TfidfVectorizer()), ('clf', MultinomialNB())])
clf_4 = Pipeline([('vect', TfidfVectorizer()), ('clf', SVC(gamma=0.15))])
clf_5 = Pipeline([('vect', CountVectorizer()), ('clf', SVC(gamma=0.15))])
clf_6 = Pipeline([('vect', TfidfVectorizer()), ('clf', SVC(kernel='rbf', gamma=50, C=1.0))])
clf_7 = Pipeline([('vect', TfidfVectorizer(stop_words=stop_words)), ('clf', RandomForestClassifier(max_depth=50, n_estimators=300))])
clf_8 = Pipeline([('vect', CountVectorizer()), ('clf', RandomForestClassifier(n_estimators=500))])

import pandas as pd
from nltk.corpus import stopwords

import numpy as np



# save_file_name = './data/sentiment_3_10321.xlsx'
save_file_name_PL = open('./data/PL_ver4.PL.txt','r') # 1713
save_file_name_nonPL = open('./data/PL_ver4.non-PL.txt','r') # 3759
s1 = save_file_name_PL.readlines()
s2 = save_file_name_nonPL.readlines()
print(save_file_name_PL)

X = [s for s in s1] + [s for s in s2]
X = X[:3426]
y = ['PL' if i < len(s1) else 'non-PL' for i in range(len(X))]
# y = [1 if i < len(s1) else 0 for i in range(len(X))]
print(X[0])
print(y)
SPLIT_PERC = 0.7
split_size = int(len(X)*SPLIT_PERC)


train_indices = np.sort(np.random.choice(len(y), round(SPLIT_PERC * len(y)), replace=False))
test_indices = np.sort(np.array(list(set(range(len(y))) - set(train_indices))))
X_train = np.array([x for ix, x in enumerate(X) if ix in train_indices])
X_test = np.array([x for ix, x in enumerate(X) if ix in test_indices])
y_train = np.array([x for ix, x in enumerate(y) if ix in train_indices])
y_test = np.array([x for ix, x in enumerate(y) if ix in test_indices])


ndarr_label = np.array(y)

print('test size :{}'.format(len(X_test)))
#
# vectorizer = TfidfVectorizer(min_df=5, max_df=0.8, stop_words='english')
# train_corpus_tf_idf = vectorizer.fit_transform(X_train)
# test_corpus_tf_idf = vectorizer.transform(X_test)
#
# print(train_corpus_tf_idf)
# print(test_corpus_tf_idf)
#
# model = MultinomialNB()
# model.fit(train_corpus_tf_idf, y_train)
# result = model.predict(test_corpus_tf_idf)
#
# print(result)
# print(y_test)
# from sklearn.metrics import accuracy_score
#
# print(accuracy_score(y_test, result))

from sklearn import metrics


def train_and_evaluate(clf, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    print("Accuracy on training set : ")
    print(clf.score(X_train, y_train))
    print("Accuracy on testing set : ")
    print(clf.score(X_test, y_test))
    y_pred = clf.predict(X_test)
    print(X_train)

    print("Classification Report : ")
    print(metrics.classification_report(y_test, y_pred))
    print(metrics.confusion_matrix(y_test, y_pred))



from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
# X_set, y_set = X_train, y_train
# X1, X2 = np.meshgrid(np.arange(start=X_set[:, 0].min() - 1, stop=X_set[:, 0].max()+1, step=0.01),
#                      np.arange(start=X_set[:, 1].min() - 1, stop=X_set[:, 1].max() + 1, step=0.01),
#                      )
# plt.contourf(X1, X2, classfier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape()), alpha=0.75, cmap=ListedColormap(('red', 'green')))

train_and_evaluate(clf_1, X_train, X_test, y_train, y_test)



train_and_evaluate(clf_8, X_train, X_test, y_train, y_test)

# datas = ["Yes completely satisfied"]
# print(datas[0])
# print(clf_4.predict([datas[0]]))
#
# def predict(X_train):
#     y_pred = clf_1.predict(X_train)
#     return y_pred
#
# for data in datas:
#     print(data)
#     predict(data)

