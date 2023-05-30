#!/usr/bin/env python
# coding: utf-8

# ### K Means

# K-means is an unsupervised learning algorithm that groups data points into a specified number of clusters.<br>
# 
# K-means works by iteratively assigning data points to the cluster with the closest mean, until the cluster means no longer change. The number of clusters is specified by the user. K-means is a simple and efficient algorithm that is often used for data exploration and clustering

# ### Spotify Recommendation

# The Spotify recommendation dataset is a collection of data about songs and users, including song features, user listening history, and user playlists. This data can be used to train machine learning models to recommend songs to users.

# In[36]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import plotly.express as px 


# In[3]:


data = pd.read_csv(r"E:\PYTHON\python Datasets\data\data.csv")
genre_data = pd.read_csv(r"E:\PYTHON\python Datasets\data\data_by_genres.csv")
year_data = pd.read_csv(r"E:\PYTHON\python Datasets\data\data_by_year.csv")


# In[4]:


data.info()


# In[7]:


genre_data.info()


# In[8]:


year_data.info()


# In[20]:


y,X = data.popularity, data.drop('popularity',axis = 1)


# In[44]:


from yellowbrick.target import FeatureCorrelation

feature_names = ['acousticness', 'danceability', 'energy', 'instrumentalness',
       'liveness', 'loudness', 'speechiness', 'tempo', 'valence','duration_ms','explicit','key','mode','year']

X, y = data[feature_names], data['popularity']

# Create a list of the feature names
features = np.array(feature_names)

# Instantiate the visualizer
visualizer = FeatureCorrelation(labels=features)

plt.rcParams['figure.figsize']=(10,6)
visualizer.fit(X, y)
visualizer.show()
plt.show()


# In[49]:


duration_by_year = data.groupby('year')['duration_ms'].mean()


# In[62]:


sns.lineplot(data = duration_by_year, x = duration_by_year.index, y = duration_by_year)
plt.show()


# In[63]:


sound_features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'valence']
fig = px.line(year_data, x='year', y=sound_features)
fig.show()


# ### clustering genre_data

# I am clustering the genres dataset because this dataset contains different audio features. These audio features can be used to recommend different songs to different types of listeners. For example, if a listener likes songs with a lot of bass, I can recommend songs that have a lot of bass. Or, if a listener likes songs with a lot of vocals, I can recommend songs that have a lot of vocals. By clustering the genres dataset, I can create a more personalized song recommendation experience for each listener.

# In[64]:


genre_data.head()


# In[66]:


genre_data.genres.nunique()


# In[78]:


top_gen = genre_data.nlargest(10,'popularity')


# In[79]:


top_gen


# In[94]:


plt.pie(top_gen.popularity,labels=top_gen.genres,autopct="%1.2f%%")
plt.show()


# In[105]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# In[107]:


genre_data1 = genre_data.select_dtypes(include=['number'])

wss = []
for i in range(1, 10):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(genre_data1)
    wss.append(kmeans.inertia_)

plt.plot(range(1, 10), wss)
plt.xlabel('Number of clusters')
plt.ylabel('Within-cluster sum of squares')
plt.show()


# In[118]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=5))])
X = genre_data.select_dtypes(include=['number'])
cluster_pipeline.fit(X)
genre_data['cluster'] = cluster_pipeline.predict(X)
genre_data1['cluster'] = cluster_pipeline.predict(X)


# In[119]:


genre_data.cluster.value_counts()


# In[126]:


from sklearn.manifold import TSNE

tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, learning_rate='auto', verbose=1))])
genre_embedding = tsne_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=genre_embedding)
projection['genres'] = genre_data['genres']
projection['cluster'] = genre_data['cluster']

fig = px.scatter(
    projection, x='x', y='y', color='cluster', hover_data=['x', 'y', 'genres'])
fig.show()


# ### clustering data (song)

# In[127]:


data.head()


# In[178]:


cluster_pipeline = Pipeline([('scaler', StandardScaler()), ('kmeans', KMeans(n_clusters=20))])
X = data.select_dtypes(include=['number'])
cluster_pipeline.fit(X)
data['cluster'] = cluster_pipeline.predict(X)
data1['cluster'] = cluster_pipeline.predict(X)


# In[179]:


data.cluster.value_counts()


# In[185]:


tsne_pipeline = Pipeline([('scaler', StandardScaler()), ('tsne', TSNE(n_components=2, learning_rate='auto', verbose=1))])
data_embedding = tsne_pipeline.fit_transform(X)
projection = pd.DataFrame(columns=['x', 'y'], data=data_embedding)
projection['name'] = data['name']
projection['mode'] = data['mode']
projection['cluster'] = data['cluster']

fig = px.scatter(
    projection, x='x', y='y', color='cluster', hover_data=['x', 'y','name','mode'])
fig.show()

