import pandas as pd
import matplotlib.pyplot as plt

input_tsv = './data/caller.txt'
# headers = ['xcompany', 'conn_id', 'file_name', 'sent_id', 'ts', 'te', 'sentence', 'doc_date', 'doc_time', 'side', 'doc_dt', 'create_dt', 'channel', 'mon']
# train = pd.read_csv(input_tsv, sep='\t', names=headers)
train = pd.read_csv(input_tsv, sep='\n', names=['sentence'])

print(train['sentence'])
""" index
    0 :  xcompany, # string
    1 :  conn_id, # int
    2 :  file_name, # string
    3 :  sent_id, # int
    4 :  ts, # int
    5 :  te, # int
    6 :  sentence, # string
    7 :  doc_date, # int
    8 :  doc_time, # string
    9 :  side, # string
    10:  doc_dt, # timestamp
    11:  create_dt, # timestamp
    12:  channel, # string
    13:  mon # int
        
"""
# test code

# print(train[1:5][2]) # 1~5행 까지 2열

import numpy as np
nrow = len(train) # 1562122 행

# from NL import text_helpers
# from nltk.corpus import stopwords
#
# text = "hi, my name is Lee Joon Woo"
# stop = stopwords.words('english')
# text_ = text_helpers.normalize_text(text,stop)
#
# print(text_)



from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from untitled.NL import text_helpers
from nltk.corpus import stopwords

stops = stopwords.words('english')
sentences = []

sentenceID = 'conn_id'
side = 'caller'

for sentence in train['sentence']:
    sentences.append(sentence)


# sentences = text_helpers.normalize_text(sentences, stops=stops)



CV = CountVectorizer(max_df=500, max_features=1000)
print(sentences[0:50])
fit_cv = CV.fit(sentences)
print(CV.get_feature_names())
count = fit_cv.transform(sentences[1500:1600]).toarray().sum(axis=0)
print(count)

idx = np.argsort(-count)
count = count[idx]
feature_name = np.array(CV.get_feature_names())[idx]

plt.bar(range(len(count)), count)
plt.show()

print(list(zip(feature_name, count)))

#
# CV = TfidfVectorizer(max_df=100, max_features=200)
# fit_cv = CV.fit_transform(sentences)
# print(CV.get_feature_names())
# print(fit_cv.toarray().sum(axis=0))
