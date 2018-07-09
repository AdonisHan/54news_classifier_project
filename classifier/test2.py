from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import string

def normalize_text(texts, stops):
    # 소문자 변환
    texts = [x.lower() for x in texts]
    # 문장 부호 제거
    texts = [''.join(c for c in x if c not in string.punctuation) for x in texts]
    # 숫자 제거
    texts = [''.join(c for c in x if c not in '0123456789') for x in texts]
    # 불용어 제거
    texts = [' '.join([word for word in x.split() if word not in (stops)]) for x in texts]
    # stemming
    #stemmer = SnowballStemmer('english')
    #texts = [' '.join(stemmer.stem(word) for word in x.split()) for x in texts]
    # 공백 제거
    texts = [' '.join(x.split()) for x in texts]

    return(texts)


stops = list(set(stopwords.words('english')))
self_stop_words = ['nonpl', 'pl', "'s", "'m", "'d", '!', "'d", "'ll", "'re", "'ve", ',', '?',   # single
                   'hmm', 'huhuh', 'um', 'ahm', 'maam', 'ok', 'hello', 'yeah', 'im', 'right', 'huh', 'yeap', 'dot', 'uh', 'com',
                   'ta', 'umm', 'wo', 'wi', 'fi', 'yup', 'ca', "ma'am",
                   'ahi', 'thank', 'hi', 'yes', 'oh', 'ha', 'bye', 'na', 'buh', 'ah', 'yah', 'uhm', 'yup',
                   'zero','one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', # number
                   'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z', # alphabet
                   'edward', 'mary', 'sam', 'nancy', 'david', # name
                   'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', # day
                   'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december', # month
                   ]
stop_words = stops + self_stop_words

X = open('/home/junuwang/data/REVO/refined/HHP.txt', 'r', errors='ignore').readlines()
print(X)
X = normalize_text(X, stops=stops)

calldata = []
for call in X:
    blob = TextBlob(call)
    calldata.append(blob.noun_phrases)

print(calldata[:10])
print(list(calldata[0]))