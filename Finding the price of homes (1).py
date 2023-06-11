#!/usr/bin/env python
# coding: utf-8

# In[13]:


import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics


# ### Importing file

# In[14]:


boston_housing = sklearn.datasets.load_boston()


# In[15]:


print(boston_housing)


# In[16]:


boston_housing_read = pd.DataFrame(boston_housing.data, columns=boston_housing.feature_names)


# In[17]:


boston_housing_read.head()


# In[18]:


boston_housing_read['price'] = boston_housing.target


# In[19]:


boston_housing_read.head()


# In[20]:


boston_housing_read.isnull().sum()


# In[21]:


boston_housing_read.describe()


# In[22]:


correlation = boston_housing_read.corr()


# In[26]:


plt.figure(figsize=(20,20))
sns.heatmap(correlation,cbar=True,square = True, fmt='.1f', annot=True, annot_kws={'size':10})


# In[24]:





# In[34]:


X = boston_housing_read.drop(['price'],axis=1)
Y = boston_housing_read['price']


# In[26]:


print(X)


# In[35]:


print(Y)


# In[30]:


X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.15, random_state = 3)


# In[31]:


print(X.shape,X_train.shape,X_test.shape)


# In[32]:


model = XGBRegressor()


# In[33]:


model.fit(X_train,Y_train)


# In[ ]:




