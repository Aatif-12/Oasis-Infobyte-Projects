#!/usr/bin/env python
# coding: utf-8

# ## Project 3 :- CAR PRICE PREDICTION WITH MACHINE LEARNING

# In[2]:


import pandas as pd
import numpy as np


# In[4]:


car_data=pd.read_csv(r"E:\Oasis Infobyte Intern\raw.githubusercontent.com_amankharwal_Website-data_master_CarPrice.csv")
car_data.head()


# In[10]:


car_data.info()


# In[14]:


car_data.shape


# In[15]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split as tts


# In[24]:


plt.figure(figsize=(12,8))
sns.heatmap(car_data.corr(),annot=True)
plt.title("Correlation betweeen the columns")


# In[30]:


x=car_data.drop(['price','CarName','fueltype','car_ID','aspiration','doornumber','carbody','drivewheel','enginelocation','enginetype','cylindernumber','fuelsystem'],axis=1)
y=car_data['price'].values
print(f" x shape : {x.shape} y shape : {y.shape}")


# In[36]:


x_train,x_test,y_train,y_test=tts(x,y,test_size=0.25,random_state=42)
print("X-train : ",x_train.shape)
print("X-test : ",x_test.shape)
print("Y-train : ",y_train.shape)
print("Y-train : ",y_test.shape)


# In[39]:


model=LinearRegression()
model.fit(x_train,y_train)


# In[41]:


pred=model.predict(x_test)


# In[42]:


import sklearn.metrics as metrics
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
print("MSE : ",(metrics.mean_squared_error(pred,y_test)))
print("MAE : ",(metrics.mean_absolute_error(pred,y_test)))
print("R2 Score : ",(metrics.r2_score(pred,y_test)))


# In[48]:


pred2=model.predict([[3,88.6,168.8,64.1,48.8,2548,130,3.47,2.68,9,111,5000,21,27]])
pred2.round(2)

