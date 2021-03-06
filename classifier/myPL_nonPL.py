import tensorflow as tf
import matplotlib.pyplot as plt
import csv
import numpy as np
import os
import string
import requests
import io
import nltk
from zipfile import ZipFile
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
from nltk.corpus import stopwords
from NL import text_helpers

sess = tf.Session()
batch_size = 500
max_features = 1000

save_file_name_PL = open('./data/PL_ver4.PL.txt','r') # 1713
save_file_name_nonPL = open('./data/PL_ver4.non-PL.txt','r') # 3759
s1 = save_file_name_PL.readlines()
s2 = save_file_name_nonPL.readlines()

sentences = [s for s in s1] + [s for s in s2]
sentences = sentences[:3400]
sentiment = [1. if i < len(s1) else 0. for i in range(len(sentences))]

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
sentences = text_helpers.normalize_text(sentences, stops=stop_words)

print(sentences)

def tokenizer(text):
    words = nltk.word_tokenize(text)
    return words

# 문서의 tf-idf 생성

tfidf = TfidfVectorizer(min_df=2, max_df=500, tokenizer=tokenizer, stop_words=stop_words, max_features=max_features)
sparse_tfidf_sentences = tfidf.fit_transform(sentences)
print(sparse_tfidf_sentences)
train_indices = np.random.choice(sparse_tfidf_sentences.shape[0], round(0.8*sparse_tfidf_sentences.shape[0]), replace=False)

test_indices = np.array(list(set(range(sparse_tfidf_sentences.shape[0])) - set(train_indices)))

texts_train = sparse_tfidf_sentences[train_indices]

texts_test = sparse_tfidf_sentences[test_indices]

target_train = np.array([x for ix, x in enumerate(sentiment) if ix in train_indices])

target_test = np.array([x for ix, x in enumerate(sentiment) if ix in test_indices])


A = tf.Variable(tf.random_normal(shape=[max_features, 1]))
b = tf.Variable(tf.random_normal(shape=[1,1]))

x_data = tf.placeholder(shape=[None, max_features], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)

model_output = tf.add(tf.matmul(x_data, A), b)
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=model_output, labels=y_target))

prediction = tf.round(tf.sigmoid(model_output))
prediction_correct = tf.cast(tf.equal(prediction, y_target), tf.float32)
accuracy = tf.reduce_mean(prediction_correct)

my_opt = tf.train.GradientDescentOptimizer(0.0005)
train_step = my_opt.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

# 모델 상태 출력

i_data = []
train_loss = []
test_loss = []
train_acc = []
test_acc = []

for i in range(100000):
    rand_index = np.random.choice(texts_train.shape[0], size=batch_size)
    rand_x = texts_train[rand_index].todense()
    rand_y = np.transpose([target_train[rand_index]])
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})

    # 100회 마다 비용 함수 값과 정확도 기록

    if (i+1) % 100 == 0:
        i_data.append(i+1)
        train_loss_temp = sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})
        train_loss.append(train_loss_temp)

        test_loss_temp = sess.run(loss, feed_dict={x_data: texts_test.todense(), y_target: np.transpose([target_test])})
        test_loss.append(test_loss_temp)

        train_acc_temp = sess.run(accuracy, feed_dict={x_data: rand_x, y_target: rand_y})
        train_acc.append(train_acc_temp)

        test_acc_temp = sess.run(accuracy, feed_dict={x_data: texts_test.todense(), y_target: np.transpose([target_test])})
        test_acc.append(test_acc_temp)

    if (i+1) % 500 == 0:
        acc_and_loss = [i+1, train_loss_temp, test_loss_temp, train_acc_temp, test_acc_temp]
        acc_and_loss = [np.round(x, 2) for x in acc_and_loss]

        print('Generation # {}. Train Loss (Test Loss) : {:.2f} ({:2f}). Train Acc (Test Acc) : {:.2f} ({:2f})'.format(*acc_and_loss))


plt.plot(i_data, train_loss, 'k--', label='Train Loss')
plt.plot(i_data, test_loss, 'r--', label='Test Loss', linewidth=4)
plt.title('Cross Entropy Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Cross Entropy Loss')
plt.legend(loc='upper right')
plt.show()

plt.plot(i_data, train_acc, 'k--', label='Train Loss Accuracy')
plt.plot(i_data, test_acc, 'r--', label='Test Loss Accuracy', linewidth=4)
plt.title('Train and Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()