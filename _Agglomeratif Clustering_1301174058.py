#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Aqmarina Alifah I
# 1301174058

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


data = pd.DataFrame ({
    'x': [20,15,60,33,55,8,10,50,44,5],
    'y': [52,50,22,16,38,25,47,36,20,6]
})


# In[3]:


import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering

dendrogram = sch.dendrogram(sch.linkage(data, method='ward'))


# In[4]:


plt.scatter(data['x'],data['y'])


# In[5]:


hc = AgglomerativeClustering(n_clusters = 4, affinity = 'euclidean', linkage = 'ward')
hc.fit_predict(data)

print(hc.labels_)


# In[6]:


plt.scatter(data['x'], data['y'], c=hc.labels_, cmap='rainbow')
# https://stackabuse.com/hierarchical-clustering-with-python-and-scikit-learn/

