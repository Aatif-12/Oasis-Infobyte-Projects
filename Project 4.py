#!/usr/bin/env python
# coding: utf-8

# ## Project 4 : EMAIL SPAM DETECTION WITH MACHINE LEARNING.

# In[32]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[33]:


data=pd.read_csv(r"E:\Oasis Infobyte Intern\4 spam detection.csv\spam.csv",encoding="ISO-8859-1")
data


# In[34]:


data.describe()


# In[35]:


data.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
data


# In[36]:


data=data.rename(columns={'v1':'Target','v2':'Message'})
data


# In[38]:


data.isnull().sum()
#data.drop_duplicates(keep='first',inplace=True)
#data.duplicated().sum()
#data.size


# In[39]:


data.drop_duplicates(keep='first',inplace=True)
data.duplicated().sum()


# In[40]:


data.size


# In[42]:


from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
data['Target']=label_encoder.fit_transform(data['Target'])
data['Target']
data.head()


# In[43]:


plt.pie(data['Target'].value_counts(), labels = ['ham','Spam'], autopct = '%0.2f')
plt.show()


# In[44]:


x=data['Message']
y=data['Target']
print(x)


# In[45]:


y


# In[47]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=3)


# In[48]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm


# In[52]:


cv=CountVectorizer()
x_train_cv=cv.fit_transform(x_train)
x_test_cv=cv.transform(x_test)
print(x_train_cv)


# In[55]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train_cv,y_train)
prediction_train=lr.predict(x_train_cv)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_train,prediction_train)*100)


# In[57]:


prediction_test = lr.predict(x_test_cv)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,prediction_test)*100)

