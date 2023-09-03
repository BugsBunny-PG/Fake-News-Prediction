
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

# Load the CSV file
df=pd.DataFrame()
df=pd.read_csv('fakeNewsDatset.csv')
# print(y_df)

# Replace NaN values in "Body" with values from "Headline"
df['Body'].fillna(df['Headline'], inplace=True)

# Define the target (y) and features (X)
y = df['Label']
X = df['Body']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Applying TF-IDF to the dataset
tfidf_vect = TfidfVectorizer(stop_words='english')
tfidf_train = tfidf_vect.fit_transform(X_train)
tfidf_test = tfidf_vect.transform(X_test)

# Applying Naive Bayes
clf = MultinomialNB()
clf.fit(tfidf_train, y_train)
pred = clf.predict(tfidf_test)
score = metrics.accuracy_score(y_test, pred)
print("Accuracy: %0.3f" % score)
cm = metrics.confusion_matrix(y_test, pred)
print(cm)

#save model
import pickle

filename='finalized_model.pkl'

pickle.dump(clf,open(filename,'wb'))

file='vectorizer.pkl'

pickle.dump(tfidf_vect,open(file,'wb'))