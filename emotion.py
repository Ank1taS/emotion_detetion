import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud

# read data-set
data_set = pd.read_csv('./Emotion_final.csv')

data = data_set.head()
print(data)
data = data_set[10:15]
print(data)

# check for null values
print(data.isnull().sum())

# plot emotions - ferquency graph
sns.countplot(x='Emotion', data=data_set, palette="Set1", dodge=False)
plt.xlabel('Emotions:')
plt.ylabel('frequency:')
plt.title('frequency of each emotions:')
plt.savefig("frequency_graph1.png")

data_set.Emotion = pd.Categorical(data_set.Emotion)
# print(Emotion_catagory)
# added a catagorical column
data_set['Emotion_categories'] = data_set.Emotion.cat.codes
print(data_set.head())

# plot emotion_categories and ferquency
sns.set_theme(style="darkgrid")
sns.countplot(x='Emotion_categories', data=data_set, palette='Set1', dodge=False)
plt.xlabel('Emotion_categories:')
plt.ylabel('frequency:')
plt.title("frequency of Emotion_categories:")
plt.savefig("frequency_graph2.png")
