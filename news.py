import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score,confusion_matrix

df=pd.DataFrame()
df=pd.read_csv('news.csv')
y_df=df.label
# print(y_df)


x_train,x_test,y_train,y_test=train_test_split(df['text'],y_df,test_size=0.2,random_state=20)
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import PassiveAggressiveClassifier

#initialize a IfidVectorizer
vector = TfidfVectorizer(stop_words='english',max_df=0.7)

#fit and transform

tf_train=vector.fit_transform(x_train)
tf_test=vector.transform(x_test)

model=PassiveAggressiveClassifier(max_iter=50)
model.fit(tf_train,y_train)
pred=model.predict(tf_test)
sc=accuracy_score(y_test,pred)
print(sc)
print(confusion_matrix(y_test,pred))

#save model
import pickle

filename='finalized_model.pkl'

pickle.dump(model,open(filename,'wb'))

file='vectorizer.pkl'

pickle.dump(vector,open(file,'wb'))