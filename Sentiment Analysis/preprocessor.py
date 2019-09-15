import numpy as np 
import pandas as pd 
import re
import nltk 
import seaborn as sns
import matplotlib.pyplot as plt
from utlis import *
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer	
from sklearn.model_selection import train_test_split

class preprocessor:

	def __init__(self):

		self.dataLink = "https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv"
		self.helper = utils()
		self.data = pd.read_csv(self.dataLink)
		self.features = None
		self.labels = None
		self.processed_features = None
		self.clean()

	def analyze(self):

		self.data = pd.read_csv(self.dataLink)
		plot_size = plt.rcParams["figure.figsize"] 
		print(plot_size[0]) 
		print(plot_size[1])
		plot_size[0] = 8
		plot_size[1] = 6
		plt.rcParams["figure.figsize"] = plot_size 
		self.data.airline.value_counts().plot(kind='pie', autopct='%1.0f%%')
		self.helper.showPlt(plt)
		self.data.airline_sentiment.value_counts().plot(kind='pie', autopct='%1.0f%%', colors=["red", "yellow", "green"])
		self.helper.showPlt(plt)
		airline_sentiment = self.data.groupby(['airline', 'airline_sentiment']).airline_sentiment.count().unstack()
		airline_sentiment.plot(kind='bar')
		self.helper.showPlt(plt)
		sns.barplot(x='airline_sentiment', y='airline_sentiment_confidence' , data=self.data)
		self.helper.showPlt(plt)

	def clean(self):

		self.features = self.data.iloc[:, 10].values
		self.labels = self.data.iloc[:, 1].values

		self.processed_features = []

		for sentence in range(0, len(self.features)):
			self.processed_feature = re.sub(r'\W', ' ', str(self.features[sentence]))
			self.processed_feature = re.sub(r'\s+[a-zA-Z]\s+', ' ', self.processed_feature)
			self.processed_feature = re.sub(r'\^[a-zA-Z]\s+', ' ', self.processed_feature) 
			self.processed_feature = re.sub(r'\s+', ' ', self.processed_feature, flags=re.I)
			self.processed_feature = re.sub(r'^b\s+', '', self.processed_feature)
			self.processed_feature = self.processed_feature.lower()
			self.processed_features.append(self.processed_feature)
		
		self.features2Num()

	def features2Num(self):
		vectorizer = TfidfVectorizer (max_features=2500, min_df=7, max_df=0.8, stop_words=stopwords.words('english'))
		self.processed_features = vectorizer.fit_transform(self.processed_features).toarray()




