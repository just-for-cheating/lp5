#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# In[2]:

data = pd.read_csv('./HousingData.csv')


# In[3]:


data


# In[4]:


data.head()


# In[5]:


data.isnull().sum()


# In[6]:


data1 = data.dropna()


# In[7]:


data1.columns


# In[8]:


data=data1


# In[9]:


x = data.drop('MEDV',axis=1)
y=data['MEDV']


# In[ ]:





# In[10]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


# In[11]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
sc=StandardScaler()
x_train_new = sc.fit_transform(x_train)
x_test_new = sc.transform(x_test)


# In[12]:


sc1 = MinMaxScaler()
y_train_new = sc1.fit_transform(y_train.values.reshape(-1,1))
y_test_new = sc1.transform(y_test.values.reshape(-1,1))


# In[13]:


y_train_new


# In[14]:


# y_train_new


# In[15]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# In[16]:


model = Sequential([
    Dense(16,activation='relu',input_shape=(13,)),
    Dense(16,activation='relu'),
    Dense(8,activation='relu'),
    Dense(4,activation='relu'),
    Dense(2,activation='relu'),
    Dense(1)
])


# In[17]:


model.summary()


# In[18]:


model.compile(optimizer='Adam',loss='mse',metrics=['mse'])


# In[19]:


model.fit(x_train_new,y_train_new,epochs=5,validation_data=(x_test_new,y_test_new))


# In[ ]:


point = x_test_new[0].reshape(1,-1)


# In[ ]:


pred = model.predict(point)
print(pred)


# In[ ]:





# In[21]:


y_pred = model.predict(x_test_new)


# In[22]:


y_pred


# In[23]:


y_test_new

