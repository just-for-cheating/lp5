#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import re 
from bs4 import BeautifulSoup
import contractions

# import BeautifulSoup


# In[2]:


data = pd.read_csv('IMDB Dataset.csv')
data.head()


# In[3]:


# Combining all the above stundents 
# from tqdm import tqdm
preprocessed_reviews = []
# tqdm is for printing the status bar
for sentance in data['review'].values:
    sentance = re.sub(r"http\S+", "", sentance)
    sentance = BeautifulSoup(sentance, 'lxml').get_text()
    sentance = contractions.fix(sentance)
    sentance = re.sub("\S*\d\S*", "", sentance).strip()
    sentance = re.sub('[^A-Za-z]+', ' ', sentance)
    # https://gist.github.com/sebleier/554280
    # sentance = ' '.join(e.lower() for e in sentance.split() if e.lower() not in stopwords)
    preprocessed_reviews.append(sentance.strip())


# In[5]:


from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

lem = WordNetLemmatizer()
stopwords = stopwords.words('english')


def ta(sent):
    words = word_tokenize(sent)

    ret_sentr = ''
    for i in words:
        if i not in stopwords:
            ret_sentr = ret_sentr + ' ' + lem.lemmatize(i)
    ret_sentr = ret_sentr[1:]
    return ret_sentr


# In[6]:


reviwes = []
for i in preprocessed_reviews:
    sent = ta(i)
    reviwes.append(sent)


# In[7]:


x = reviwes
y = data['sentiment']


# In[9]:


from sklearn.preprocessing import OneHotEncoder
le = OneHotEncoder()
y = le.fit_transform(y.values.reshape(-1,1))


# In[10]:


y_new = y.toarray()


# In[11]:


y_new


# In[12]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y_new,test_size=0.2)


# In[13]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf2 = TfidfVectorizer(ngram_range=(1,2),max_features=5000)
x_train_new = tfidf2.fit_transform(x_train)
x_test_new = tfidf2.transform(x_test)


# In[14]:


y_test.shape


# <h3>Model

# In[15]:


import tensorflow as tf


# In[16]:


x_train_new = x_train_new.toarray()
x_test_new = x_test_new.toarray()


# In[18]:


x_train_new.shape


# In[19]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32,input_shape=(5000,)),
    tf.keras.layers.Dense(16,activation='relu'),
    tf.keras.layers.Dense(8,activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(4,activation='relu'),
    tf.keras.layers.Dense(2,activation='sigmoid')
])


# In[20]:


model.compile(optimizer='Adam',loss='binary_crossentropy' ,metrics=['accuracy'])


# In[21]:


model.summary()


# In[22]:



model.fit(x_train_new,y_train,epochs=10,batch_size=128,validation_data=(x_test_new,y_test))


# In[ ]:





# In[23]:


y_pred= model.predict(x_test_new)


# In[26]:


np.set_printoptions(suppress=True)
y_pred


# In[27]:


y_test[0]


# In[ ]:




