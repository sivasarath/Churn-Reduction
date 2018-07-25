
# coding: utf-8

# In[2]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import seaborn as sns
from random import randrange, uniform


# In[3]:


os.chdir('C:/Users/sarath chandra/Desktop/data science/new project')


# In[4]:


test=pd.read_csv('Test_data.csv')
train=pd.read_csv('Train_data.csv')
test.dtypes


# In[4]:


for i in range(0,train.shape[1]):
    
    if(train.iloc[:,i].dtypes.name=='object'):
        print(i)
        train.iloc[:,i]=pd.Categorical(train.iloc[:,i])
        train.iloc[:,i] = train.iloc[:,i].cat.codes 
        train.iloc[:,i] = train.iloc[:,i].astype('category')


# In[5]:


test.head()


# In[6]:


for i in range(0,train.shape[1]):
    
    if(test.iloc[:,i].dtypes.name=='object'):
        print(i)
        test.iloc[:,i]=pd.Categorical(test.iloc[:,i])
        test.iloc[:,i] = test.iloc[:,i].cat.codes 
        test.iloc[:,i] = test.iloc[:,i].astype('category')


# In[7]:


from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split


# In[8]:


train['Churn'] = train['Churn'].replace(0, 'No')
train['Churn'] = train['Churn'].replace(1, 'Yes')
test['Churn'] = test['Churn'].replace(0, 'No')
test['Churn'] = test['Churn'].replace(1, 'Yes')


# In[9]:


X_train = train.values[:, 0:19]
y_train = train.values[:,20]
X_test=test.values[:,0:19]
y_test=test.values[:,20]



# In[10]:


C50_model = tree.DecisionTreeClassifier(criterion='entropy').fit(X_train, y_train)

#predict new test cases
C50_Predictions = C50_model.predict(X_test)


# In[11]:


CM = pd.crosstab(y_test, C50_Predictions)

#let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]


# In[12]:


((TP+TN)*100)/(TP+TN+FP+FN)


# In[34]:


y_train.size


# In[13]:


(FN*100)/(FN+TP)


# In[35]:


from sklearn.ensemble import RandomForestClassifier

RF_model = RandomForestClassifier(n_estimators = 20).fit(X_train, y_train)


# In[36]:


RF_Predictions = RF_model.predict(X_test)


# In[37]:


CM = pd.crosstab(y_test, RF_Predictions)

#let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]


# In[38]:


((TP+TN)*100)/(TP+TN+FP+FN)


# In[39]:


(FN*100)/(FN+TP)


# In[51]:


from sklearn.naive_bayes import GaussianNB

#Naive Bayes implementation
NB_model = GaussianNB().fit(X_train, y_train)


# In[52]:


NB_Predictions = NB_model.predict(X_test)


# In[53]:


CM = pd.crosstab(y_test, NB_Predictions)

#let us save TP, TN, FP, FN
TN = CM.iloc[0,0]
FN = CM.iloc[1,0]
TP = CM.iloc[1,1]
FP = CM.iloc[0,1]


# In[54]:


((TP+TN)*100)/(TP+TN+FP+FN)


# In[55]:


(FN*100)/(FN+TP)

