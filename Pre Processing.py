
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import seaborn as sns
from random import randrange, uniform


# In[ ]:


arihant.jec2013@gmail.com


# In[2]:


os.chdir('C:/Users/sarath chandra/Desktop/data science/new project')


# In[3]:


train=pd.read_csv('Train_data.csv')


# In[4]:


train.shape


# In[7]:


for i in range(0,train.shape[1]):
    
    if(train.iloc[:,i].dtypes.name=='object'):
        print(i)
        train.iloc[:,i]=pd.Categorical(train.iloc[:,i])
        train.iloc[:,i] = train.iloc[:,i].cat.codes 
        train.iloc[:,i] = train.iloc[:,i].astype('category')
        
        
        


# In[8]:


df=train.copy()


# In[11]:


cname=[]
for i in list(train):
    if train[i].dtypes.name != 'category':
         print(i)
         cname.append(i)


# In[12]:


df_corr = train.loc[:,cname]


# In[13]:


for i in cname:
    q75, q25 = np.percentile(train[i], [75 ,25])

# #Calculate IQR
    iqr = q75 - q25

# #Calculate inner and outer fence
     minimum = q25 - (iqr*1.5)
     maximum = q75 + (iqr*1.5)

# #Replace with NA
train.loc[train['custAge'] < minimum,:'custAge'] = np.nan
train.loc[train['custAge'] > maximum,:'custAge'] = np.nan

#Impute with KNN
train = pd.DataFrame(KNN(k = 3).complete(train), columns = train.columns)


# In[14]:


f, ax = plt.subplots(figsize=(7, 5))

#Generate correlation matrix
corr = df_corr.corr()

#Plot using seaborn library
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)


# In[16]:


catname=[]
for i in list(train):
    if train[i].dtypes.name == 'category':
         print(i)
         catname.append(i)


# In[17]:


for i in catname:
    print(i)
    chi2, p, dof, ex = chi2_contingency(pd.crosstab(train['Churn'], train[i]))
    print(p)


# In[18]:


for i in cname:
    print(i)
    train[i] = (train[i] - min(train[i]))/(max(train[i]) - min(train[i]))


# In[26]:


get_ipython().run_line_magic('matplotlib', 'inline')
for i in cname:
    print(i)
    train[i].plot.hist(bins=10)


# In[33]:


train['account length'].plot.hist(bins=100)


# In[41]:


train['total day charge'].plot.hist(bins=30)

