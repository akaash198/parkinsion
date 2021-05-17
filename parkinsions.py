#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[15]:


df=pd.read_csv('parkinsons.csv')
df.head()


# In[16]:


features=df.loc[:,df.columns!='status'].values[:,1:]
labels=df.loc[:,'status'].values


# In[17]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(features)
y=labels


# In[18]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x, y, test_size=0.3, random_state=7)


# In[19]:


import keras
from keras.models import Sequential
from keras.layers import Dense


# In[20]:


classifier = Sequential()


# In[21]:


classifier.add(Dense(units = 128, kernel_initializer = 'uniform', activation = 'relu', input_dim = 22))


# In[22]:


classifier.add(Dense(units = 64, kernel_initializer = 'uniform', activation = 'relu'))


# In[23]:


classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))


# In[24]:


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])


# In[25]:


classifier.fit(x_train, y_train,batch_size =len(x_train),validation_data=(x_test,y_test),epochs = 200)


# In[26]:


y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)


# In[27]:


classifier.save('suba.h5')


# In[ ]:




