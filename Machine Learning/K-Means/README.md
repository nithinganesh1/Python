# K Means

**K-Means is a very simple algorithom which cluster data into a K number of cluster**
# K Means Use Case
**1. Image Segmentation** <br>
**2. Clustering Gene Segementation Data** <br>
**3. News Article Clustering** <br>
**4. Clustering Languages** <br>
**5. Species Clustering** <br>
**6. Anomaly Detection** 
# Choosing the Value of K
**We often know the value of K. In that case we use the value of K. Else we use the Elbow Method.
We run the algorithm for different values of K(say K = 10 to 1) and plot the K values against SSE(Sum of Squared Errors). And select the value of the k for the elbow point**


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import pandas as pd

data = pd.read_excel(r"E:\PYTHON\python Datasets\Dataset-Kmeans-xclara.csv.xlsx",sheet_name="Sheet1")
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>V1</th>
      <th>V2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.072345</td>
      <td>-3.241693</td>
    </tr>
    <tr>
      <th>1</th>
      <td>17.936710</td>
      <td>15.784810</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.083576</td>
      <td>7.319176</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.120670</td>
      <td>14.406780</td>
    </tr>
    <tr>
      <th>4</th>
      <td>23.711550</td>
      <td>2.557729</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.shape
```




    (3000, 2)




```python
v1=data.V1.values
v2=data.V2.values
X=np.array(list(zip(v1,v2)))
#NumPy arrays are optimized for numerical computations and can be processed more efficiently than DataFrames
plt.scatter(v1,v2,c='g',s=7)
plt.show()
```


    
![output_4_0](https://user-images.githubusercontent.com/122164879/224533208-45125feb-0320-442a-9bda-9267eca7215c.png)

    



```python
#Euclidean Destence calculator
def dist(a,b,ax=1):
    return np.linalg.norm(a-b,axis=ax)
```


```python
k=3
# X coordinates of random centroids
C_x = np.random.randint(0, np.max(X)-20, size=k)
# Y coordinates of random centroids
C_y= np.random.randint(0, np.max(X)-20, size=k)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
print (C)
```

    [[76. 13.]
     [45. 48.]
     [38. 56.]]
    


```python
plt.scatter(v1,v2,c='g',s=7)
plt.scatter(C_x, C_y, marker='*', s=200, c='r')
plt.show()
```


    
![output_7_0](https://user-images.githubusercontent.com/122164879/224533215-6b533943-69ca-49e1-b195-0c735d237c12.png)

    



```python
from copy import deepcopy

# Number of clusters
k = 3

# Randomly initialize centroids
C = X[np.random.choice(len(X), size=k, replace=False)]

# Initialize variables
error = 1e9
tolerance = 1e-4

# Loop will run till the error becomes smaller than tolerance
while error > tolerance:
    # Assigning each value to its closest cluster
    distances = np.linalg.norm(X[:, np.newaxis, :] - C, axis=-1)
    clusters = np.argmin(distances, axis=1)

    # Storing the old centroid values
    C_old = deepcopy(C)

    # Finding the new centroids by taking the average value
    C = np.array([np.mean(X[clusters == i], axis=0) for i in range(k)])

    # Compute the error
    error = np.linalg.norm(C - C_old)

# Plot the clusters
colors = ['r', 'g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots()
for i in range(k):
    points = X[clusters == i]
    ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')
plt.show()
```


    

    


**we can speed up this code using sklearn**


```python
from sklearn.cluster import KMeans
# Number of clusters kmeans = KMeans (n_clusters=3)
kmeans = KMeans (n_clusters=3)
# Fitting the input data kmeans = kmeans.fit(X)
kmeans = kmeans.fit(X)
# Getting the cluster Labels labels = kmeans.predict(X) # Centroid values
centroids = kmeans.cluster_centers_
# Comparing with scikit-Learn centroids 
print (C) # From Scratch 
print(centroids) # From sci-kit learn
```

    [[ 69.92418447 -10.11964119]
     [  9.4780459   10.686052  ]
     [ 40.68362784  59.71589274]]
    [[ 69.92418447 -10.11964119]
     [ 40.68362784  59.71589274]
     [  9.4780459   10.686052  ]]
    


```python
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Plot the results
colors =['r', 'g', 'b', 'y', 'c', 'm']
fig, ax = plt.subplots()
for i in range(3):
    points = np.array([X[j] for j in range(len(X)) if labels[j] == i])
    ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
ax.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=200, c='#050505')
plt.show()
```


    
![output_11_0](https://user-images.githubusercontent.com/122164879/224533241-19bc18e2-2e44-4578-af5b-8a3e06e3b0ed.png)

    

