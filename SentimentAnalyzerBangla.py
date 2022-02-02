import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils import shuffle
import numpy as np
from sklearn.naive_bayes import ComplementNB


data = pd.read_csv('D:\\PythonProgramming\\training_data.csv')
bangla_stopwords_csv = pd.read_csv('D:\\PythonProgramming\\bangla_stopwords.csv')
bangla_stopwords_csv = bangla_stopwords_csv.drop('ID', axis = 1)
bangla_stopword_lists = bangla_stopwords_csv.values.tolist()

bangla_stopword_list = []
for stopword in bangla_stopword_lists:
    bangla_stopword_list.append(stopword[0])

bangla_stopwords = frozenset(bangla_stopword_list)

#data = data.sample(frac=1).reset_index(drop=True)


# Split into training and testing data
x = data['comment']
y = data['polarity']

x, x_test, y, y_test = train_test_split(x,y, stratify=y, test_size=0.25, random_state=42)

# Vectorize text reviews to numbers
vec = CountVectorizer(stop_words=bangla_stopwords)
x = vec.fit_transform(x).toarray()
x_test = vec.transform(x_test).toarray()

model = ComplementNB()
model.fit(x, y)

print(model.score(x_test, y_test))

print(model.predict(vec.transform(['সব পরিমনীর নাটক'])))
print(model.predict(vec.transform(['পরিমনী ভাল মেয়ে'])))