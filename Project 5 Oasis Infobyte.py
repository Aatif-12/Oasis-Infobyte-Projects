#!/usr/bin/env python
# coding: utf-8

# ## Project 5 : SALES PREDICTION USING PYTHON.

# In[2]:


import pandas as pd 
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


data = pd.read_csv(r"E:\Oasis Infobyte Intern\archive5.csv\Advertising.csv")
data


# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:


data.describe()


# In[7]:


data.isnull().sum()


# In[8]:


fig, axs = plt.subplots(3, figsize = (5,5))
plt1 = sns.boxplot(data['TV'], ax = axs[0])
plt2 = sns.boxplot(data['Radio'], ax = axs[1])
plt3 = sns.boxplot(data['Newspaper'], ax = axs[2])
plt.tight_layout()


# In[9]:


sns.distplot(data['Newspaper'])


# In[11]:


irq = data.Newspaper.quantile(0.75) - data.Newspaper.quantile(0.25)


# In[12]:


lower_bridge = data["Newspaper"].quantile(0.25) - (irq*1.5)
upper_bridge = data["Newspaper"].quantile(0.75) + (irq*1.5)
print(lower_bridge)
print(upper_bridge)


# In[21]:


df = data.copy()
df.loc[df['Newspaper']>=93, 'Newspaper'] = 93
sns.boxplot(df['Newspaper'])


# In[22]:


sns.boxplot(df['Sales']);


# In[23]:


sns.pairplot(df, x_vars=['TV', 'Radio', 'Newspaper'],
            y_vars='Sales',height=4,aspect=1,kind='scatter')
plt.show()


# In[24]:


sns.heatmap(df.corr(), cmap = "YlGnBu", annot = True)
plt.show()


# In[27]:


important_feature=list(df.corr()['Sales'][(df.corr()['Sales']>+0.5)|(df.corr()['Sales']<-0.5)].index)
print(important_feature)


# In[31]:


x=data['TV']
y=data['Sales']
X=x.values.reshape(-1,1)
X


# In[32]:


y


# In[33]:


print(X.shape,y.shape)


# In[35]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,y,test_size=0.33)
print(x_train.shape,y_train.shape)


# In[37]:


from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# In[38]:


knm = KNeighborsRegressor().fit(x_train,y_train)
knm


# In[41]:


knm_train_pred = knm.predict(x_train)
knm_test_pred = knm.predict(x_test)
print(knm_train_pred,knm_test_pred)


# In[42]:


Result=pd.DataFrame(columns=['Models','Train R2','Test R2','Test RMSE','Variance'])


# In[45]:


r2 = r2_score(y_test,knm_test_pred)
r2_train = r2_score(y_train,knm_train_pred)
rmse = np.sqrt(mean_squared_error(y_test,knm_test_pred))
variance = r2_train -r2
Result = Result.append({'Model':'K-Nearest Neighbors','Train R2':r2_train,'Test R2':r2,'Test RMSE':rmse,'Variance':variance},ignore_index=True)
print("R2:",r2)
print('RMSE:',rmse)


# In[46]:


Result.head()

