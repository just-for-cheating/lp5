#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


# from tensorflow.keras.datasets import fashion_mnist


# In[3]:


# fashion_mnist


# In[2]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Rescaling
from tensorflow.keras.utils import to_categorical


# In[3]:


from tensorflow.keras.layers import Flatten


# In[4]:


# (x_train,y_train),(x_test,y_test) = fashion_mnist.load_data()
import pandas as pd
train = pd.read_csv('../prac_codes/fashion-mnist_train.csv')
test = pd.read_csv('../prac_codes/fashion-mnist_test.csv')
train.head()


# In[5]:


x_train = train.drop('label',axis=1)
y_train = train['label']
x_test = test.drop('label',axis=1)
y_test = test['label']


# In[6]:


y_train


# In[7]:


x_train = x_train.values.reshape(x_train.shape[0],28,28,1)
x_test = x_test.values.reshape(x_test.shape[0],28,28,1)


# In[8]:


x_train.shape


# In[9]:


import numpy as np


# In[10]:


print(x_train.shape)
print(x_test.shape)


# In[11]:


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


# In[13]:


y_train[0].shape


# <h4>Model

# In[12]:


model = Sequential([
    Rescaling(1./255),
    Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(100,activation='relu'),
    Dense(10,activation='softmax')
])


# In[13]:


model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


# In[14]:


history = model.fit(x_train,y_train,epochs=5,batch_size=32,validation_data=(x_test,y_test))


# In[ ]:





# In[ ]:





# In[ ]:




