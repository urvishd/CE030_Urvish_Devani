# -*- coding: utf-8 -*-


import nltk
from nltk.corpus import twitter_samples 
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

nltk.download('twitter_samples')
nltk.download('stopwords')

import re
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer

#clean our tweets,remove unwanted words and char.
def process_tweet(tweet):
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('english')
    tweet = re.sub(r'\$\w*', '', tweet)
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    tweet = re.sub(r'#', '', tweet)
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
      if(word not in stopwords_english and word not in string.punctuation):
        stem_word = stemmer.stem(word)
        tweets_clean.append(stem_word)
    return tweets_clean

#check freq of words with pred. value
def build_freqs(tweets, ys):
    yslist = np.squeeze(ys).tolist()
    freqs = {}

    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
              freqs[pair] += 1
            else:
              freqs[pair] = 1
    return freqs

all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

#take data
test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]
train_x = train_pos + train_neg
test_x = test_pos + test_neg

train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)

Final_data = all_positive_tweets+all_negative_tweets
data =np.append(np.ones((len(all_positive_tweets), 1)), np.zeros((len(all_negative_tweets), 1)), axis=0)
train_x,test_x,train_y,test_y = train_test_split(Final_data,data,test_size=0.25,random_state= 26)

#our word dataset with freq
freqs = build_freqs(train_x , train_y)
print("type(freqs) = " + str(type(freqs)))
print("len(freqs) = " + str(len(freqs.keys())))

def extract_features(tweet, freqs): 
    word_l = process_tweet(tweet)
    x = np.zeros((1, 2)) 
    for word in word_l:
        if((word,1) in freqs):
          x[0,0]+=freqs[word,1]
        if((word,0) in freqs):
          x[0,1]+=freqs[word,0]
    
    assert(x.shape == (1, 2))
    return x[0]

#pred function
def predict_tweet(tweet):
    with tf.Session() as sess:
      saver.restore(sess,save_path='TSession')
      data_i=[]
      for t in tweet:
        data_i.append(extract_features(t,freqs))
      data_i=np.asarray(data_i)
      return sess.run(tf.nn.sigmoid(tf.add(tf.matmul(a=data_i,b=W,transpose_b=True),bias)))
    print("--Fail--")
    return

bias=tf.Variable(np.random.randn(1),name="Bias")
W=tf.Variable(np.random.randn(1,2),name="Weight")

data=[]
for t in train_x:
  data.append(extract_features(t,freqs))
data=np.asarray(data)

Y_hat = tf.nn.sigmoid(tf.add(tf.matmul(np.asarray(data), W,transpose_b=True), bias)) 
ta=np.asarray(train_y)
Total_cost = tf.nn.sigmoid_cross_entropy_with_logits(logits = Y_hat, labels = ta) 
print(Total_cost)

# Gradient Descent Optimizer 
optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.00001 ,name="GradientDescent").minimize(Total_cost) 
# Global Variables Initializer 
init = tf.global_variables_initializer()

saver = tf.train.Saver()
with tf.Session() as sess:
  
  sess.run(init)
  print("Bias",sess.run(bias))
  print("Weight",sess.run(W))
  for epoch in range(1000):
    sess.run(optimizer)
    preds=sess.run(Y_hat)
    acc=((preds==ta).sum())/len(train_y)
    Accuracy=[]
    repoch=False
    if repoch:
      Accuracy.append(acc)
    if epoch % 1000 == 0:
      print("Accuracy",acc)
    saved_path = saver.save(sess, 'TSession')

preds=predict_tweet(test_x)
print(preds,len(test_y))

def accuracy(x,y):
  return ((x==y).sum())/len(y)

print(accuracy(preds,test_y))