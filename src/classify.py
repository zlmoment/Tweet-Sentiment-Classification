#!/usr/bin/python
# -*- coding: utf-8 -*-

# ECE/CS 7720
# @author: Zhaoyu Li
# @mail: zlht3@mail.missouri.edu

#
# preprocess the tweet, build and train the classifier, and do cross validation
#

import csv
import sys
import sklearn
from nltk.classify import *

import sklearn
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from utils import *

# the main class
class Classify():

	trainingDataFile = 'training.test.csv'
	tweets = []
	featureList = []
	training_set = []

	n_fold = 10

	def __init__(self, trainingDataFile):
		self.trainingDataFile = trainingDataFile

	# preprocess tweet
	# this function invokes anther function prcessTweet() in util.py
	# to do the preprocess
	def preProcessTweets(self):
		log("Pre-processing tweets...")

		#Read the tweets one by one and pre-process it
		with open(self.trainingDataFile,'r') as csvfile:
			inpTweets = csv.DictReader(csvfile, delimiter=',')
			for line in inpTweets:
				self.tweets.append((processTweet(line['text']), line['class']))
		csvfile.close()

	# read feature list from the file generated before
	def readFeatureList(self):
		log("Read feature list...")
		f = open("feature_list.data",'r')
		content = f.readlines()
		self.featureList = content[0].split(',')
		f.close()

	# extract features
	# to create the term vector
	def extract_features(self, tweet):
		tweet_words = set(tweet)
		features = {}
		for word in self.featureList:
			if word in tweet_words:
				features[word] = 1
			else:
				features[word] = 0
		return features

	# get the cross validation part
	def getCrossValidationData(self, tweets, n):
		result = []
		length = len(tweets)
		idx_point = range(0, length, length/self.n_fold)
		testcv = []
		traincv = []
		testlabel = []
		test_range = range(idx_point[n], idx_point[n]+length/self.n_fold)
		for i in range(0, length):
			if i in test_range:
				testcv.append(self.extract_features(tweets[i][0]))
				testlabel.append(tweets[i][1])
			else:
				traincv.append(tweets[i])
		traincv = nltk.classify.util.apply_features(self.extract_features, traincv)
		result.extend([traincv, testcv, testlabel])
		return result

	# get the rate of accuracy
	def getAccuracy(self, classifier):
		classifier = SklearnClassifier(classifier)
		accuracy = 0
		for fold in range(0, self.n_fold):
			log(str(fold+1) + " iteration...")
			log("    Partitioning...")
			datacv = self.getCrossValidationData(self.tweets, fold)
			traincv = datacv[0]
			testcv = datacv[1]
			testlabel = datacv[2]
			log("    Training...")
			classifier.train(traincv)
			log("    Classifying...")
			label_pred = classifier.batch_classify(testcv)
			tempScore = accuracy_score(testlabel, label_pred)
			log("    Accuracy for this iteration: " + str(tempScore))
			accuracy += tempScore
		return accuracy/self.n_fold

	# the main loop
	def classify(self):

		self.preProcessTweets()
		self.readFeatureList()

		log("Start KNN")
		accuracy_knn = self.getAccuracy(KNeighborsClassifier(5))
		implog('KNN`s average accuracy is %f' % accuracy_knn)

		log("Start BernoulliNB")
		accuracy_bernoulliNB = self.getAccuracy(BernoulliNB())
		implog('BernoulliNB`s average accuracy is %f' % accuracy_bernoulliNB)
		
		log("Start MultinomialNB")
		accuracy_multinomialNB = self.getAccuracy(MultinomialNB())
		implog('MultinomiaNB`s average accuracy is %f' % accuracy_multinomialNB)

# the main entrance
if __name__ == '__main__': 
	if len(sys.argv) != 2:
		log("Please add training data file.")
		exit()
	# get the address of training data
	trainingDataFile = sys.argv[1]
	# create class and start to train and classify
	cl = Classify(trainingDataFile)
	cl.classify()