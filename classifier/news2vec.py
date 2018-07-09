from bs4 import BeautifulSoup
from konlpy.tag import Twitter
import os
import pickle
# # total news data
#
# filenames = os.listdir('/home/junuwang/data/IWM_태깅문서_151021/refined')
# twitter = Twitter()
# sentences = []
# for filename in filenames:
#     full_filename = os.path.join('/home/junuwang/data/IWM_태깅문서_151021/refined', filename)
#
#     with open(full_filename) as f:
#         soup = BeautifulSoup(f, 'html.parser')
#         contents = soup.find_all('content')
#
#
#         for content in contents:
#             pos = twitter.pos(content.get_text(), norm=True, stem=True)
#             NV_content = [word for word, tag in pos if tag[0] in 'NV']
#             sentences.append(NV_content)
#             # with open('/home/junuwang/data/IWM_태깅문서_151021/refined/all_news.txt', 'a') as news_f:
#             #     news_f.write('{}\n'.format(NV_content))
#
#
# from gensim.models.word2vec import Word2Vec
#
#
# # save model
#
# model = Word2Vec(sentences, window=5, min_count=4, size=100)
#
# with open('/home/junuwang/data/IWM_태깅문서_151021/refined/word2vec.bin', 'wb') as f:
#     pickle.dump(model, f)


with open('/home/junuwang/data/IWM_태깅문서_151021/refined/word2vec.bin', 'rb') as f:
    model = pickle.load(f)


# t-SNE 시각화

from sklearn.manifold import TSNE
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.cluster import KMeans

n_clusters = 50
n_datas = 100



# matplot 한국어 폰트 추가
font_location = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
font_name = fm.FontProperties(fname=font_location).get_name()
mpl.rc('font', family=font_name)


mpl.rcParams['axes.unicode_minus'] = False

model_name = 'news_data'
vocab = list(model.wv.vocab) # vocab 리스트
print(len(vocab))
X = model[vocab] # vocab 의 vector 값
tsne = TSNE(n_components=2) # 2차원으로 차원 축소
word_vectors = model.wv.syn0


kmeans_clustering = KMeans(n_clusters=n_clusters)
idx = kmeans_clustering.fit_predict(word_vectors)

# word/index 사전
idx = list(idx)
names = model.wv.index2word
print(names)
word_centroid_map = {names[i]:idx[i] for i in range(len(names))}

for cluster in range(10):
    # cluster 번호 출력
    print("\nCluster {}".format(cluster))

    words = []
    for i in range(len(list(word_centroid_map.values()))):
        if (list(word_centroid_map.values())[i] == cluster):
            words.append(list(word_centroid_map.keys())[i])
    print(words)





X_tsne = tsne.fit_transform(X[:n_datas,:])

import pandas as pd
df = pd.DataFrame(X_tsne, index=vocab[:n_datas], columns=['x', 'y'])
print(df.shape)

fig = plt.figure()
fig.set_size_inches(40, 40)
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df['x'], df['y'])

print('hi')
for word, pos in df.iterrows():
    ax.annotate(word, pos, fontsize=10)
print('end')
print('end1')

plt.show()
#
# while(1):
#     input_str = input()
#     print(input_str)
#     print(model.most_similar(positive=input_str, topn=10))
#
# print(model.most_similar(positive=['반려동물']))