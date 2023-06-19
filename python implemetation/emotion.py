import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
# from sklearn.feature_selection import chi2
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
import sklearn.metrics as metrics
from wordcloud import WordCloud

# read data-set
data_set = pd.read_csv('./Emotion_final.csv')

data = data_set.head()
print(data)
data = data_set[10:15]
print(data)

print(f"shape of data set:  {data_set.shape}")

# print each emotion count in data set
emotions = data_set['Emotion'].value_counts()
print(emotions)
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

# forms word cloud / tokens
words = ' '.join([token for token in data_set['Text']])
word_cloud = WordCloud(width=1000, height=1000, random_state=21, min_font_size=20, max_font_size=120).generate(words)
plt.figure(figsize=(20,20))
plt.imshow(word_cloud, interpolation='spline16')
plt.axis('off')
plt.savefig("wordcould.png")

# CountVectorizer to converting a collection of text documents into a matrix of token counts.
vectorizer = CountVectorizer(max_features=1500, min_df=5, max_df = 0.7, stop_words='english')
X = vectorizer.fit_transform(data_set['Text']).toarray()
# The goal of using tf-idf instead of the raw frequencies of occurrence of a token in a given document is to scale down the impact of tokens that occur very frequently in a given corpus and that are hence empirically less informative than features that occur in a small fraction of the training corpus.
# TfidfTransformer object from the scikit-learn library. The TfidfTransformer is used to transform a count matrix (such as the one obtained from CountVectorizer) into a TF-IDF (Term Frequency-Inverse Document Frequency) representation.
tfidfconverter = TfidfTransformer()
X = tfidfconverter.fit_transform(X).toarray()

# split data_set into training set and testing set
X_train, X_test, y_train, y_test = train_test_split(X, data_set['Emotion'], test_size=0.18, random_state=0)

# ========== RandomForestClassifier ==========
from sklearn.ensemble import RandomForestClassifier
classifier1 = RandomForestClassifier(n_estimators=100, random_state=0)
classifier1.fit(X_train, y_train)

y_predict = classifier1.predict(X_test)

print("Accuracy[RandomForestClassifier]: ", metrics.accuracy_score(y_test, y_predict))

# predicts the class or label for a new input using a trained classifier model. It uses the predict method of the classifier to make predictions on a new text input, which is represented as a feature vector using a vectorizer object.
print(classifier1.predict(vectorizer.transform(["my pc is broken but i am really very happy as i got a new pc from my wife "])))

# =========== logisticRegression ============
lr = LogisticRegression(max_iter=1000, multi_class='multinomial')
lr.fit(X_train, y_train)
y_predict_logistic_regression = lr.predict(X_test)
accuracy = metrics.accuracy_score(y_test, y_predict_logistic_regression)

print("Accuracy[Logistic regression]: ", accuracy)
print(lr.predict(vectorizer.transform(["my pc is broken but i am really very happy as i got a new pc from my wife "])))

# =========== Multinomial Naive Bayes ===========
classifier3 = MultinomialNB()
classifier3.fit(X_train, y_train)
y_predict2 = classifier3.predict(X_test)

print("Accuracy[Multinomial Naive Bayes]: ", metrics.accuracy_score(y_test, y_predict2))
print(classifier3.predict(vectorizer.transform(["my pc is broken but i am really very happy as i got a new pc from my wife "])))

