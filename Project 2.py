#!/usr/bin/env python
# coding: utf-8

# ## Project 2 : UNEMPLOYMENT ANALYSIS WITH PYTHON

# In[1]:


import pandas as pd
import numpy as np


# In[6]:


data=pd.read_csv(r"E:\Oasis Infobyte Intern\archive.csv\Unemployment in India.csv")
data=pd.read_csv(r"E:\Oasis Infobyte Intern\archive.csv\Unemployment_Rate_upto_11_2020.csv")
print(data.head())


# In[17]:


print(data.isnull().sum())


# In[21]:


data.columns=["States","Date","Frequency",
             "Estimated Unemployment Rate",
             "Estimated Employed","Estimated Labour Participation Rate",
             "Region","longitude","latitude"]


# In[23]:


import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[28]:


plt.style.use('seaborn-whitegrid')
plt.figure(figsize=(14,10))
sns.heatmap(data.corr())
plt.show


# In[43]:


data.columns=["States","Date","Frequency",
             "Estimated Unemployment Rate",
             "Estimated Employed","Estimated Labour Participation Rate",
             "Region","longitude","latitude"]
plt.figure(figsize=(10,8))
plt.title("Indian Unemployment")
sns.histplot(x="Estimated Employed",hue="Region",data=data)
plt.show()


# In[42]:


plt.figure(figsize=(10,8))
plt.title("Indian Unemployment")
sns.histplot(x="Estimated Unemployment Rate",hue="Region",data=data)
plt.show()


# In[51]:


unemployment=data[["States","Region","Estimated Unemployment Rate"]]
figure=px.sunburst(unemployment,path=["Region","States"],
                   values="Estimated Unemployment Rate",
                   width=900, height=900,color_continuous_scale="RdYlGn",
                   title="Unemployment Rate in India")
figure.show()

