from sklearn.externals import joblib
import numpy as np
from nltk.corpus import stopwords
import nltk

stops = list(set(stopwords.words('english')))
self_stop_words = ['PL', 'non-pl', 'pl', "'s", "'m", "'d", '!', "'d", "'ll", "'re", "'ve", ',', '?',   # single
                   'hmm', 'huhuh', 'um', 'ahm', 'maam', 'ok', 'hello', 'yeah', 'im', 'right', 'huh', 'yeap', 'dot', 'uh', 'com',
                   'ta', 'umm', 'wo', 'wi', 'fi', 'yup', 'ca',
                   'ahi', 'thank', 'hi', 'yes', 'oh', 'ha', 'bye', 'na', 'buh', 'ah', 'yah', 'uhm', #
                   'zero','one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', # number
                   'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z', # alphabet
                   'edward', 'mary', 'sam', 'nancy', 'david', #name
                   'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
                   'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december',
                   ]
stop_words = stops + self_stop_words

max_features = 500

def tokenizer(text):
    words = nltk.word_tokenize(text)
    return words

top_n_words = 10

clf1 = joblib.load('./clf1.pkl')
clf2 = joblib.load('./clf2.pkl')
clf3 = joblib.load('./clf3.pkl')
tfidf_vec = joblib.load('./TFIDF.pkl')
dict_rev = joblib.load('./dict_rev.pkl')

print(dict_rev)


# test another texts
while(1):
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


