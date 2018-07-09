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


# download sample data
#
# save_file_name = 'temp_spam_data.csv'
# if os.path.isfile(save_file_name):
#     text_data = []
#     with open(save_file_name, 'r') as temp_output_file:
#         reader = csv.reader(temp_output_file)
#         for row in reader:
#             text_data.append(row)
#
# else:
#     zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
#     r = requests.get(zip_url)
#     z = ZipFile(io.BytesIO(r.content))
#     file = z.read('SMSSpamCollection')
#
#     #데이터 정리
#
#     text_data = file.decode()
#     text_data = text_data.encode('ascii', errors='ignore')
#     text_data = text_data.decode().split('\n')
#     text_data = [x.split('\t') for x in text_data if len(x)>=1]
#
#     #save .csv
#
#     with open(save_file_name, 'w') as temp_output_file:
#         writer = csv.writer(temp_output_file)
#         writer.writerows(text_data)

# import os


# s1 = []
# filenames = os.listdir('./data/aclImdb/train/neg')
# for filename in filenames:
#     full_filename = os.path.join('./data/aclImdb/train/neg', filename)
#     f = open(full_filename)
#     s1 += f.readlines()
#     f.close()
#
# s2 = []
# filenames = os.listdir('./data/aclImdb/train/pos')
# for filename in filenames:
#     full_filename = os.path.join('./data/aclImdb/train/pos', filename)
#     f = open(full_filename)
#     s2 += f.readlines()
#     f.close()

#
#
# save_file_name_PL = open('./data/PL_ver4.PL.txt','r') # 1713
# save_file_name_nonPL = open('./data/PL_ver4.non-PL.txt','r') # 3759
# s1 = save_file_name_PL.readlines()
# s2 = save_file_name_nonPL.readlines()
# print(save_file_name_PL)
#
# X = [s for s in s1] + [s for s in s2]
# X = X[:3426]
# y = ['PL' if i < len(s1) else 'non-PL' for i in range(len(X))]

X = []
y = []

dirname = '/home/junuwang/data/REVO/refined/'
filenames = os.listdir(dirname)
filenames.sort()


for filename in filenames:
    full_filename = os.path.join(dirname, filename)
    calldata = open(full_filename, 'r', errors='ignore').readlines()

    for call in calldata:
        X.append(call)
        y.append(filename)



# make stop_words

stops = list(set(stopwords.words('english')))
self_stop_words = ['nonpl', 'pl', "'s", "'m", "'d", '!', "'d", "'ll", "'re", "'ve", ',', '?',   # single
                   'hmm', 'huhuh', 'um', 'ahm', 'maam', 'ok', 'hello', 'yeah', 'im', 'right', 'huh', 'yeap', 'dot', 'uh', 'com',
                   'ta', 'umm', 'wo', 'wi', 'fi', 'yup', 'ca', "ma'am",
                   'ahi', 'thank', 'hi', 'yes', 'oh', 'ha', 'bye', 'na', 'buh', 'ah', 'yah', 'uhm',
                   'zero','one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', # number
                   'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z', # alphabet
                   'edward', 'mary', 'sam', 'nancy', 'david', # name
                   'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', # day
                   'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december', # month
                   ]
stop_words = stops + self_stop_words



#
# X = [x[1] for x in s1]
# y = [x[0] for x in s2]


# 스팸은 1, 비스팸은 0으로 표기

text_helpers.normalize_text(X, stops=stops)



# tfidf dictionary size

max_features = 1000

def tokenizer(text):
    words = nltk.word_tokenize(text)
    return words

tfidf = TfidfVectorizer(tokenizer=tokenizer, stop_words=stops, max_features=max_features)

X = text_helpers.normalize_text(X, stops=stop_words)



# Training

SPLIT_PERC = 0.7

train_indices = np.sort(np.random.choice(len(y), round(SPLIT_PERC * len(y)), replace=False))
test_indices = np.sort(np.array(list(set(range(len(y))) - set(train_indices))))

X_train = np.array([x for ix, x in enumerate(X) if ix in train_indices])
X_test = np.array([x for ix, x in enumerate(X) if ix in test_indices])
y_train = np.array([x for ix, x in enumerate(y) if ix in train_indices])
y_test = np.array([x for ix, x in enumerate(y) if ix in test_indices])


tfidf_vec = tfidf.fit(X_train)
tfidf_train = tfidf_vec.transform(X_train).toarray()
tfidf_test = tfidf_vec.transform(X_test).toarray()
print(tfidf_train)

print("Test size : ", len(tfidf_test))

dict_org = tfidf.vocabulary_
print("Tfidf dictionary : ",dict_org)

dict_rev_keys = dict_org.values()
dict_rev_values = dict_org.keys()
dict_rev = dict(zip(dict_rev_keys, dict_rev_values))
print("Tfidf reverse dictionary : ", dict_rev)

# define classification

# RF clf
# from sklearn.model_selection import cross_val_score
# clf1 = MLPClassifier()
# clf1.fit(tfidf_train, y_train)
# clf1_pred = clf1.predict(tfidf_test)

# scores = cross_val_score(clf1, tfidf_test, y_test, cv=5, scoring='accuracy')
# print("Accuracy : {:0.3f} (+/- {:0.3f})".format(scores.mean(), scores.std()))

# SVB clf
clf2 = SVC(kernel='rbf', gamma=0.1)
clf2.fit(tfidf_train, y_train)
clf2_pred = clf2.predict(tfidf_test)

# NB clf
clf3 = MultinomialNB()
clf3.fit(tfidf_train, y_train)
clf3_pred = clf3.predict(tfidf_test)
#
#
# # KFold Cross Validation
# from sklearn.model_selection import cross_val_score
# # Ensemble
# from mlxtend.classifier import EnsembleVoteClassifier
#
#
# eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], weights=[1,1,1])
#
# labels = ['Random forest', 'Support vector machine', 'Naive Bayes', 'Ensemble']
# for clf, label in zip([clf1, clf2, clf3, eclf], labels):
#
#     scores = cross_val_score(clf, tfidf_test, y_test, cv=5, scoring='accuracy')
#     print("Accuracy : {:0.3f} (+/- {:0.3f}) [{}]".format(scores.mean(), scores.std(), label))
#
#

# using XGBoost (to do)



# random sampling

n_samples = 10
init_n = 0
top_n_words = 10


from sklearn.externals import joblib

# joblib.dump(clf1, 'clf1.pkl')
# joblib.dump(clf2, 'clf2.pkl')
# joblib.dump(clf3, 'clf3.pkl')
# joblib.dump(tfidf_vec, 'TFIDF.pkl')
# joblib.dump(dict_rev, 'dict_rev.pkl')


# from wordcloud import WordCloud, STOPWORDS
# import matplotlib.pyplot as plt
#
# text = open('./data/PL_ver4.PL.txt', 'r').read()
# wordcloud = WordCloud(max_font_size=40, stopwords=stop_words).generate(text)
#
# plt.figure(figsize=(12,12))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.show()
#
#
# text = open('./data/PL_ver4.non-PL.txt', 'r').read()
# wordcloud = WordCloud(max_font_size=40, stopwords=stop_words).generate(text)
#
# plt.figure(figsize=(12,12))
# plt.imshow(wordcloud, interpolation='bilinear')
# plt.axis('off')
# plt.show()


from sklearn import metrics

print(metrics.classification_report(y_test, clf2_pred))
print(metrics.confusion_matrix(y_test, clf2_pred))

print(metrics.classification_report(y_test, clf3_pred))
print(metrics.confusion_matrix(y_test, clf3_pred))






# test another example
while(0):
    print("please input text : ")
    s = [input()]
    s_normal = ['. '.join(x.split('.')) for x in s]
    s_normal = ['? '.join(x.split('?')) for x in s]
    s_normal = s_normal[0].split('.')

    tfidf_s = tfidf_vec.transform(s).toarray()
    y_pred = clf1.predict(tfidf_s)

    dict_ix_valuable = np.argsort(-tfidf_s[0])[:top_n_words]
    print("valuable word`s index :",dict_ix_valuable)
    pol_word = [dict_rev.get(ix) for ix in dict_ix_valuable]

    print(pol_word, " : ", -np.sort(-tfidf_s[0])[:top_n_words], "\n")

    print(y_pred)

    # version.1
    print("\nV1 (omit sentences don`t have valuable words) :")
    pol_string = []
    for s in s_normal:
        for w in s.lower().split():
            if w in pol_word:
                pol_string.append(s)
                break
    print(pol_string)

    # version.2
    print("\nV2 (original sentences, appending (*) end of sentences having valuable words :")
    new_string = []
    for s in s_normal:
        for w in s.lower().split():
            if w in pol_word:
                s += "(*)"
                break

        new_string.append(s)
    print(new_string)


