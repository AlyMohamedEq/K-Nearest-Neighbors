#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt


# In[2]:


all_data = pd.read_csv("E:\Basics of ML & Python\creditcard.csv")


# In[3]:


all_data.head()


# In[4]:


all_data[all_data['Class'] == 0]


# In[5]:


all_data[all_data['Class'] == 1]


# In[8]:


norm_trx = all_data[all_data['Class'] == 0]


# In[9]:


fraud_trx = all_data[all_data['Class'] == 1]


# In[11]:


norm_trx[norm_trx['Amount']<=2500].Amount


# In[12]:


from sklearn.preprocessing import StandardScaler


# In[13]:


all_data['norm_amount'] = StandardScaler().fit_transform(all_data['Amount'].values.reshape(-1,1))


# In[16]:


all_data.head()


# In[15]:


norm_data = all_data.drop(['Time', 'Amount'], axis=1)


# In[17]:


norm_data.head()


# In[18]:


number_of_fraud = len(norm_data[norm_data['Class']== 1])
fraud_index = np.array(norm_data[norm_data['Class']== 1].index)


# In[19]:


all_normal_index = np.array(norm_data[norm_data['Class']== 0].index)


# In[20]:


random_normal_index = np.random.choice(all_normal_index, number_of_fraud)


# In[21]:


balanced_index = np.concatenate([fraud_index, random_normal_index])


# In[22]:


balanced_data = norm_data.iloc[balanced_index,:]


# In[23]:


sns.countplot('Class', data = balanced_data)


# In[34]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[25]:


model = KNeighborsClassifier(n_neighbors=3)


# In[28]:


train , test = train_test_split(balanced_data , test_size = 0.3)


# In[36]:


train_x = train.drop('Class', axis=1)
train_y = train['Class']

test_x = test.drop('Class', axis=1)
test_y = test['Class']


# In[37]:


model.fit(train_x,train_y)


# In[38]:


y_predicted = model.predict(test_x)


# In[39]:


print("Accuracy:",metrics.accuracy_score(test_y, y_predicted))


# In[ ]:




