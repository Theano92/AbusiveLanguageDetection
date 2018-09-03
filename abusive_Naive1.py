import json
import csv
import nltk
import re
import pandas as pd
from nltk.classify import NaiveBayesClassifier
import nltk.classify.util
from nltk.corpus import stopwords
from sklearn import naive_bayes
from sklearn.naive_bayes import MultinomialNB
from scipy import sparse
import scipy
from sklearn import metrics
import numpy as np



tweets_annotation = []
tweets_text = []
stop_words = set(stopwords.words('english'))

#open the json file and read the text line by line
with open('amateur_expert.json', 'rU') as File:
    data = File.readlines()
	



# create another file to write only the category and the tweet inside it
with open('data.txt', 'w') as outfile:
    for line in data:
        tweets = json.loads(line) #convert the json tweet to python 
        tweets_annotation = tweets['Annotation'] # retrieve the labels 
        tweets_text = tweets['text'] # retrieve the tweets
        outfile.write(tweets_annotation)
        outfile.write(",") # seperate the labels and the tweet with a ","
        outfile.write(repr((tweets_text).encode('utf8')))
        outfile.write("\n")
        
        

# create a dataframe with two columns
data = pd.read_csv('data.txt',sep = ',',names = ['category','tweet'])


# select only the tweets in "neither" category
neither_category = data.loc[data['category']=='Neither','tweet']


# Dictionary for neither tweets
neither_tweets = []

for neither_tweet in neither_category: 
	tokenize_neither_tweets = neither_tweet.split(" ")    # split each "tweet" in neither category
	useful_words = [word for word in tokenize_neither_tweets if not word in stop_words]  #remove the stop words from the tweets
	my_dict = dict([(word,True) for word in useful_words])
	neither_tweets.append((my_dict,"Neither"))


# select only the tweets in "Sexism" category
sexism_category = data.loc[data['category']=='Sexism','tweet']

# Dictionary for sexism tweets
sexism_tweets = []

for sexism_tweet in sexism_category: 
	tokenize_sexism_tweets = sexism_tweet.split(" ") # split each "tweet" in "sexism" category
	useful_words = [word for word in tokenize_sexism_tweets if not word in stop_words] #remove the stop words from the tweets
	my_dict = dict([(word,True) for word in useful_words])
	sexism_tweets.append((my_dict,"Sexism"))


# select only the tweets in "Racism" category
racism_category = data.loc[data['category'] == 'Racism','tweet']

# Dictionary for racism tweets
racism_tweets = []

for racism_tweet in racism_category: 
	tokenize_racism_tweets = racism_tweet.split(" ") # split each "tweet" in "sexism" category
	useful_words = [word for word in tokenize_racism_tweets if not word in stop_words] #remove the stop words from the tweets
	my_dict = dict([(word,True) for word in useful_words])
	racism_tweets.append((my_dict,"Racism"))



print('The Neither category contains: {} tweets'.format(len(neither_tweets)))
print('The Racism category contains: {} tweets'.format(len(racism_tweets)))
print('The Sexism category contains: {} tweets'.format(len(sexism_tweets)))


# Split the dataset in training data and test data 80% and 20% respectively
train_set = neither_tweets[:4676] + sexism_tweets[:728] + racism_tweets[:79]
test_set = neither_tweets[4676:] + sexism_tweets[728:] + racism_tweets[79:]



# Train the model
classifier = NaiveBayesClassifier.train(train_set)


# Test the model in another dataset
accuracy = nltk.classify.util.accuracy(classifier, test_set)
print('The Accuracy is: {} '.format(accuracy * 100))



checkTweet = []

# Tweet to examine if the model works properly or not
tweets = ["Women should not drive!", "I forgot my phone","Israel is SOOO #racist - sure. after the #arab #terror #attack today"]


for tweet in tweets:
  words = tweet.split(" ")
  useful_words = [word for word in words if not word in stop_words]
  my_dict = dict([(word,True) for word in useful_words])
  check = classifier.classify(my_dict)
  print('The tweet "{}" belongs to : {} category'.format(tweet,check))






















 


 
