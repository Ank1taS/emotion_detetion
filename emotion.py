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

print(f"shape of data set:  {data_set.shape}")

# check for null values
print(data.isnull().sum())

# # plot emotions - ferquency graph
# sns.countplot(x='Emotion', data=data_set, palette="Set1", dodge=False)
# plt.xlabel('Emotions:')
# plt.ylabel('frequency:')
# plt.title('frequency of each emotions:')
# plt.savefig("frequency_graph1.png")

data_set.Emotion = pd.Categorical(data_set.Emotion)
# print(Emotion_catagory)
# added a catagorical column
data_set['Emotion_categories'] = data_set.Emotion.cat.codes
print(data_set.head())

# # plot emotion_categories and ferquency
# # sns.set_theme(style="darkgrid")
# sns.countplot(x='Emotion_categories', data=data_set, palette='Set1', dodge=False)
# plt.xlabel('Emotion_categories:')
# plt.ylabel('frequency:')
# plt.title("frequency of Emotion_categories:")
# plt.savefig("frequency_graph2.png")

# data cleaning
def get_clean_text(text):
    text = re.sub(r'@[A-Za-z0-9]*', '', text)               # to remove @mentions
    text = re.sub(r'#', '', text)                           # to remove # tag
    text = re.sub(r'RT[\s]+', '', text)                     # to remove retweets
    text = re.sub(r'https?:\/\/\S+', '', text)              # to remove hyperlinks
    text = re.sub('(\\\\u([a-z]|[0-9])+)', '', text)        # to remove unicode escape sequence character
    # text = re.sub(r'\\\\u([0-9A-Fa-f]{4})', '', text)       # to remove unicode escape sequence character
    text = re.sub(r'"', '', text)
    text = re.sub(r':', '', text)
    text = re.sub(r'https?:?', '', text)
    text = re.sub(r'href', '', text)

    return text

print(data_set.dtypes)

print(data_set['Text'][323])
data_set['Text'] = data_set['Text'].apply(get_clean_text)
print(data_set['Text'][323])
