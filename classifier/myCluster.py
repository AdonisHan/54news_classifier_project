from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from nltk.corpus import stopwords
from NL import text_helpers
import sys
print(sys.path)

save_file_name_PL = open('./data/PL_ver4.PL.txt','r') # 1713
save_file_name_nonPL = open('./data/PL_ver4.non-PL.txt','r') # 3759
s1 = save_file_name_PL.readlines()
s2 = save_file_name_nonPL.readlines()

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

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic %d:" % (topic_idx))
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))

X = open('./data/PL_ver4.PL.txt','r')
documents = text_helpers.normalize_text(X, stops=stop_words)

no_features = 1000

# NMF is able to use tf-idf
tfidf_vectorizer = TfidfVectorizer(max_df=0.50, min_df=2, max_features=no_features, stop_words='english')
tfidf = tfidf_vectorizer.fit_transform(documents)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()

# LDA can only use raw term counts for LDA because it is a probabilistic graphical model
tf_vectorizer = CountVectorizer(max_df=0.50, min_df=2, max_features=no_features, stop_words='english')
tf = tf_vectorizer.fit_transform(documents)
tf_feature_names = tf_vectorizer.get_feature_names()

no_topics = 20

# Run NMF
nmf = NMF(n_components=no_topics, random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd').fit(tfidf)

# Run LDA
lda = LatentDirichletAllocation(n_topics=no_topics, max_iter=5, learning_method='online', learning_offset=50.,random_state=0).fit(tf)

no_top_words = 20
display_topics(nmf, tfidf_feature_names, no_top_words)
display_topics(lda, tf_feature_names, no_top_words)

import pyLDAvis
import pyLDAvis.sklearn

pyLDAvis.sklearn.prepare(lda, tf, tf_vectorizer)


"""
명사만 뽑기
documents = []
for idx, r in enumerate(result):
    text = r[0]+' '+r[1]
    values = {'s':text.encode('utf8')}
    r = requests.get("http:// ", params=values)
    
    doc = []
    body = json.loads(r.text)
    for i in body:
        if body[i]['feature'] in ['NNG', 'NNP']:
            doc.append(body[i]['data'])
    documents.append(doc)
"""