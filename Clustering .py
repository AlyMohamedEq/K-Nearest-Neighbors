#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 


# In[2]:


dataset = pd.read_csv('E:\\Basics of ML & Python\\Mall_Customers.csv')


# In[3]:


dataset.head()


# In[4]:


dataset.info()


# In[6]:


data = dataset.iloc[:,[3,4]]


# In[7]:


data 


# In[ ]:





# In[8]:


from sklearn.cluster import KMeans


# In[13]:


wcss = []
for i in range(1,11):
    km = KMeans(n_clusters=i, init='k-means++', random_state=0)
    km.fit(data)
    wcss.append(km.inertia_)


# In[14]:


wcss

plt.plot(wcss,'ro-')
# In[16]:


#  to get the number of clusters
plt.plot(wcss,'ro--')


# In[24]:


kmeans2 = KMeans(n_clusters= 6)
x_clusters = kmeans2.fit_predict(data)


# In[25]:


x_clusters


# In[26]:


plt.figure(figsize=(10,7))
plt.scatter(x = data.iloc[:,0], y = data.iloc[:,1], c= x_clusters)


# In[29]:


data['x_clusters'] = x_clusters


# In[30]:


data


# In[ ]:




