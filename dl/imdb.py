import pandas as pd

df=pd.read_csv('./imdb-dataset.csv')
# If reading CSV gives error, use this. One by one carefully.
# df = pd.read_csv('./imdb-dataset.csv', header = None, quoting=3, error_bad_lines=False)
# df.rename(columns = { 1: 'sentiment', 0:'review',}, inplace = True)
# df = df.iloc[1: , :]

df.sample(1000)
df.head()

df['sentiment'].replace({'positive':1 , 'negative':0} , inplace=True)

df.head()

import re
def removeHtml(text):
    c=re.compile('<.*?>')
    return re.sub(c,' ',text)

df['review']=df['review'].apply(removeHtml)

df.head()

def toLower(text):
    return text.lower()

df['review']=df['review'].apply(toLower)

df.head()

def remSp(text):
    t=''
    for ch in text:
        if ch.isalnum():
            t+=ch
        else:
            t+=' '
    return t

df['review']=df['review'].apply(remSp)

df.head()

from sklearn.feature_extraction.text import TfidfVectorizer
vect=TfidfVectorizer()

x=vect.fit_transform(df['review'])
y=df['sentiment']

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=42)

from sklearn.decomposition import TruncatedSVD
# Create an instance of TruncatedSVD
svd = TruncatedSVD(n_components=100)  # Set the desired number of components
# Fit TruncatedSVD on the training data and apply the transformation to both training and test data
xtrain_svd = svd.fit_transform(xtrain)
xtest_svd = svd.transform(xtest)

xtrain=xtrain_svd
xtest=xtest_svd

xtest.shape
xtrain.shape

import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers
ann=models.Sequential([
    layers.Dense(units=16,activation='relu',input_dim=xtrain.shape[1]),
    layers.Dense(units=8,activation='relu'),
    layers.Dense(units=1,activation='sigmoid')
])
ann.compile(optimizer='adam',
           loss='binary_crossentropy',
           metrics=['accuracy'])

ann.fit(xtrain,ytrain,epochs=10)

ann.evaluate(xtest,ytest)

xtest[:5]

ytest[:5]

ypred=ann.predict(xtest)

yclass=[round(e[0]) for e in ypred]

yclass[:5]