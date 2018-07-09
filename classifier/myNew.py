from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import  MultinomialNB
import numpy as np
from sklearn import metrics
import nltk
from nltk.corpus import stopwords
from NL import text_helpers
import csv
import os
from zipfile import ZipFile
import requests
import io
import string

save_file_name = 'temp_spam_data.csv'
if os.path.isfile(save_file_name):
    text_data = []
    with open(save_file_name, 'r') as temp_output_file:
        reader = csv.reader(temp_output_file)
        for row in reader:
            text_data.append(row)

else:
    zip_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip'
    r = requests.get(zip_url)
    z = ZipFile(io.BytesIO(r.content))
    file = z.read('SMSSpamCollection')

    #데이터 정리

    text_data = file.decode()
    text_data = text_data.encode('ascii', errors='ignore')
    text_data = text_data.decode().split('\n')
    text_data = [x.split('\t') for x in text_data if len(x)>=1]

    #save .csv

    with open(save_file_name, 'w') as temp_output_file:
        writer = csv.writer(temp_output_file)
        writer.writerows(text_data)

texts = [x[1] for x in text_data]
target = [x[0] for x in text_data]

# 스팸은 1, 비스팸은 0으로 표기

y = [1. if x=='spam' else 0. for x in target]

texts = [' '.join(x.split('.')) for x in texts]

# 소문자 변환
texts = [x.lower() for x in texts]

# 문장 부호 제거
texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]

# 숫자 제거
texts = [''.join(c for c in x if c not in '0123456789') for x in texts]

# 공백 제거
X = [' '.join(x.split()) for x in texts]


# add stop words
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

# tfidf dictionary size

max_features = 1000

def tokenizer(text):
    words = nltk.word_tokenize(text)
    return words

tfidf = TfidfVectorizer(tokenizer=tokenizer, stop_words=stop_words, max_features=max_features)

# read data

# save_file_name_PL = open('./data/PL_ver4.PL.txt', 'r') # 1713
# save_file_name_nonPL = open('./data/PL_ver4.non-PL.txt', 'r') # 3759
# s1 = save_file_name_PL.readlines()
# s2 = save_file_name_nonPL.readlines()
#
#
# # normalize input data (preprocessing)
#
# X = [s for s in s1] + [s for s in s2]
X = text_helpers.normalize_text(X, stops=stop_words)
# X = [' '.join(x.split('.')) for x in X]
# X_rest = X[3426:]
# X = X[:3400]
# y = ['PL' if i < len(s1) else 'non-PL' for i in range(len(X))]
# # 1 is 'PL', 0 is 'non-PL'


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


print("Test size : ", len(tfidf_test))

dict_org = tfidf.vocabulary_
print("Tfidf dictionary : ",dict_org)

dict_rev_keys = dict_org.values()
dict_rev_values = dict_org.keys()
dict_rev = dict(zip(dict_rev_keys, dict_rev_values))
print("Tfidf reverse dictionary : ", dict_rev)

# define classification
clf1 = RandomForestClassifier(n_estimators=500)
clf1.fit(tfidf_train, y_train)
clf1_pred = clf1.predict(tfidf_test)

# predict test set
ixlist = []
for ix, arr in enumerate(tfidf_test):
    if clf1.predict([arr]) != [y_test[ix]]:
        ixlist += [ix]

clf2 = SVC(kernel='rbf', gamma=0.1)
clf2.fit(tfidf_train, y_train)
SVC_pred = clf2.predict(tfidf_test)

clf3 = MultinomialNB()


# KFold Cross Validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

print('5-fold cross validation:\n')
labels = ['Random forest', 'Support vector machine', 'Naive bayes']


for clf, label in zip([clf1, clf2, clf3], labels):
    scores = cross_val_score(clf, tfidf_test, y_test, cv=5, scoring='accuracy')

    print("Accuracy : {} (+/- {}) [{}]".format(scores.mean(), scores.std(), label))


from mlxtend.classifier import EnsembleVoteClassifier


eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], weights=[1,1,1])

labels = ['Random forest', 'Support vector machine', 'Naive Bayes', 'Ensemble']
for clf, label in zip([clf1, clf2, clf3, eclf], labels):

    scores = cross_val_score(clf, tfidf_test, y_test,cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))



# using XGBoost




# random sampling

n_samples = 10
init_n = 0
top_n_words = 10

# # print("\nRandom {} sampling : ".format(n_samples))
# rand = np.random.choice(len(y_test), n_samples, replace=False)
# PLdata = np.random.choice(len(y_test[:150]), n_samples, replace=False)

## n_samples PL_nonPL data
# for n_samples in rand:
#     init_n += 1
#     print("#", init_n, "call :", X_test[n_samples])
#     # print(tfidf_test[n_samples])
#     dict_ix = tfidf_test[n_samples].argmax()
#     dict_ix_valuable = np.argsort(-tfidf_test[n_samples])[:top_n_words] # argsort
#     # pol_word = [word for word, ix in dict.items() if ix == dict_ix]
#     pol_word = [dict_rev.get(dict_ix)]
#     pol_word1 = [dict_rev.get(ix) for ix in dict_ix_valuable]
#     # print(pol_word," : ",tfidf_test[n_samples].max(),"\n")
#
#     print(pol_word1, " : ", -np.sort(-tfidf_test[n_samples])[:top_n_words], "\n")

# n_samples of PL data
# for n_samples in PLdata:
#     init_n += 1
#     print("#", init_n, "call :", X_test[n_samples])
#     # print(tfidf_test[n_samples])
#     dict_ix = tfidf_test[n_samples].argmax()
#     dict_ix_valuable = np.argsort(-tfidf_test[n_samples])[:top_n_words] # argsort
#     # pol_word = [word for word, ix in dict.items() if ix == dict_ix]
#     pol_word = [dict_rev.get(dict_ix)]
#     pol_word1 = [dict_rev.get(ix) for ix in dict_ix_valuable]
#     # print(pol_word," : ",tfidf_test[n_samples].max(),"\n")
#
#     print(pol_word1, " : ", -np.sort(-tfidf_test[n_samples])[:top_n_words], "\n")



from sklearn.externals import joblib

joblib.dump(clf1, 'clf1.pkl')
joblib.dump(tfidf_vec, 'TFIDF.pkl')
joblib.dump(dict_rev, 'dict_rev.pkl')


from NL.text_helpers import normalize_text

# test another non-PL
while(1):
    print("please input text : ")
    s = [input()]
    s_normal = ['. '.join(x.split('.')) for x in s]
    s_normal = ['? '.join(x.split('?')) for x in s]
    s_normal = s_normal[0].split('.')

    tfidf_s = tfidf_vec.transform(s).toarray()
    y_pred = clf1.predict(tfidf_s)

    print(tfidf_s)
    dict_ix_valuable = np.argsort(-tfidf_s[0])[:top_n_words]
    print(dict_ix_valuable)
    pol_word = [dict_rev.get(ix) for ix in dict_ix_valuable]

    print(pol_word, " : ", -np.sort(-tfidf_s[0])[:top_n_words], "\n")

    # version.1
    pol_string = []
    for s in s_normal:
        for w in s.lower().split():
            if w in pol_word:
                pol_string.append(s)
                break
    print(pol_string)
    print(y_pred)

    # version.2
    new_string = []
    for s in s_normal:
        for w in s.lower().split():
            if w in pol_word:
                s += "(*)"
                break

        new_string.append(s)

    print(new_string)


