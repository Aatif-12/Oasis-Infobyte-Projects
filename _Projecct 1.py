#!/usr/bin/env python
# coding: utf-8

# ## Project 1 :  IRIS FLOWER CLASSIFICATION.

# In[11]:


import pandas as pd
import numpy as np


# In[12]:


Iris=pd.read_csv("E:\Oasis Infobyte Intern\Iris.csv")
Iris


# In[13]:


Iris.drop(columns =["Id"],inplace=True)
Iris


# In[14]:


Iris.head()


# In[19]:


Iris["Species"].replace({"Iris-setosa":1,"Iris-versicolor":2,"Iris-virginica":3},inplace=True)
Iris


# In[28]:


x=pd.DataFrame(Iris,columns=["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]).values
x


# In[31]:


y=Iris.Species.values.reshape(-1,1)
y


# In[44]:


from sklearn.neighbors import KNeighborsClassifier      #import required libraries
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[32]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)
x_train.shape


# In[33]:


y_train.shape


# In[38]:


a=6
knclr=KNeighborsClassifier(a)


# In[39]:


knclr


# In[40]:


knclr.fit(x_train,y_train)


# In[43]:


y_pred=knclr.predict(x_test)
metrics.accuracy_score(y_test,y_pred)*100

