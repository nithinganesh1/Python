#!/usr/bin/env python
# coding: utf-8

# #**Demo: K-Means Clustering Using Python**

# ###**Problem Definition**
# 
# Perform K-Means Clustering on a dataset containing shopping details of customers from various cities to understand how clustering segregates the data similar to each other.
# 
# 
# 
# 
# ###**Dataset Description**
# 
# The dataset is based on the the shopping details of customers of few cities. The data has been populated randomly and holds no such relation to any real systems data or confidential data. It has 8 dimensions or features with 200 entries.
# 
# The Attributes are:
# 
# >* CustomerID
# >* CustomerGender
# >* CustomerAge
# >* CustomerCity
# >* AnnualIncome
# >* CreditScore
# >* SpendingScore
# >* CustomerCityID
# 
# ###**Tasks to be performed**
# 
# 
# >* Importing Required Libraries
# >* Analyzing the data
# >* Understanding K-Means
# >* Implementing K-Means from Scratch
# >* Implementing K-Means using sklearn library 

# ###**Importing Required Libraries**
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score,confusion_matrix
import warnings
warnings.filterwarnings("ignore")


# In[3]:


df1 = pd.read_csv('Shopping_CustomerData.csv')
df1.head()


# ####**Analyzing the Data**

# In[4]:


#Here, we will take only two features and top 400 entries of each feature from the dataset just to make it easy to visualize the steps.
df_new = df1[["CustomerAge","SpendingScore"]]
df_new.head()


# In[5]:


df_new.describe()


# In[6]:


#Checking for Null Values
df_new.isnull().sum()


# **Lets plot these two variables and visualize them**

# In[7]:


plt.scatter(df_new.iloc[:,0],df_new.iloc[:,1])
plt.xlabel('CustomerAge')
plt.ylabel('SpendingScore')
plt.title('Plot of Unclustered Data')
plt.show()


# ##**Implementing K-Means using scikit learn**

# In[8]:


#Here, we will take only two features and top 400 entries of each feature from the dataset just to make it easy to visualize the steps.
df = df1[["CustomerAge","SpendingScore"]]
df.head()


# In[9]:


#Here, we are assuming the value of k as 5
kmeans = KMeans(n_clusters=3)#Creating a K-Means Object
kmeans.fit(df)#Fitting the Model


# In[10]:


#Here, we are generating Labels of each point
labels = kmeans.predict(df)
labels


# In[11]:


#printing the centroids of each cluster
centroids = kmeans.cluster_centers_
centroids


# In[12]:


#Sum of squared distances of data-points to their closest cluster center. It tells us how well the formed clusters are
kmeans.inertia_


# ###**Let's visualize the Clustered Data**

# In[13]:


plt.figure(figsize=(10, 5))
colmap = {1:'y',2:'g',3:'b',4:'r',5:'c'}
colors = map(lambda x: colmap[x+1], labels)
print(colors)
colors1=list(colors)

plt.scatter(df['CustomerAge'], df['SpendingScore'], color=colors1, alpha=0.5)
for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid, color=colmap[idx+1])
plt.xlabel('Age')
plt.ylabel('Spending Score')
plt.title('Plot of Clustered Data')
plt.show()


# ###**How to determine the value of K?**
# 
# >* If we know how many classes we want to classify, then we use that value as 'k'. For Example - All of us have heard of the Iris data or even worked with it earlier. It has three classes we could classify our flowers into. So, in that case the value of k could be taken as 3.
# >* If we don't know how many classes we want, then we will have to decide what the best 'k' value is. A very popular to find the value of 'k' is **Elbow Method**

# ###**Elbow Method**

# In[14]:


inertia_list = []
for num_clusters in np.arange(1, 21):
    kmeans =KMeans(n_clusters=num_clusters)
    kmeans.fit(df)
    inertia_list.append(kmeans.inertia_)


# In[15]:


inertia_list


# In[16]:


#Plotting the Elbow Curve
plt.figure(figsize=(10, 5))
plt.plot(np.arange(1, 21), inertia_list)
plt.grid(True)
plt.xlabel('Values of K')
plt.ylabel('Inertia')
plt.title('Elbow Curve')
plt.show()


# **From above, we select the optimum value of k by determining the Elbow Point - a point after which the inertia starts decreasing linearly. In this case, we can select the value of k as 10.**

# In[ ]:




