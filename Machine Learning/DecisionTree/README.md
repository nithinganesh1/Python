# DecisionTree
1. It belongs to the class of supervised learning algorithms
2. where it can be used for both classification and regression purposes.
3. At the beginning, the whole training set is considered as the root.
4. Feature values need to be categorical. If the values are continuous then they are discretized prior to building the        model.
5. Records are distributed recursively on the basis of attribute values.
6. Order to placing attributes as root or internal node of the tree is done by using some statistical approach.
### Attribute selection measures

#### Entropy
The ID3 (Iterative Dichotomiser) Decision Tree algorithm uses entropy to calculate information gain. So, by calculating decrease in entropy measure of each attribute we can calculate their information gain.

![entropy-formula](https://user-images.githubusercontent.com/122164879/215848799-61179717-6900-4126-afd7-37bf3d6e43c5.png)


```python

```
value between 0 to 1
zero means pure node
one means bad nodeHere, c is the number of classes and pi is the probability associated with the ith class.
#### Gini index
Another attribute selection measure that CART (Categorical and Regression Trees) uses is the Gini index. It uses the Gini method to create split points.

![images](https://user-images.githubusercontent.com/122164879/215848903-b4134866-a3c6-4406-8460-4f0a2751dd8d.png)

```python

```
1. Calculate Gini for sub-nodes, using formula sum of the square of probability for success and failure (p^2+q^2).
2. Calculate Gini for split using weighted Gini score of each node of that split.

```python

```


```python
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv(r"E:\PYTHON\python Datasets\Tree\PlayTennis.csv")
```

### Sample DecisionTree Code


```python
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
      <th>outlook</th>
      <th>temp</th>
      <th>humidity</th>
      <th>windy</th>
      <th>play</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>sunny</td>
      <td>hot</td>
      <td>high</td>
      <td>False</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>sunny</td>
      <td>hot</td>
      <td>high</td>
      <td>True</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>overcast</td>
      <td>hot</td>
      <td>high</td>
      <td>False</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>rainy</td>
      <td>mild</td>
      <td>high</td>
      <td>False</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>4</th>
      <td>rainy</td>
      <td>cool</td>
      <td>normal</td>
      <td>False</td>
      <td>yes</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.preprocessing import LabelEncoder
Le = LabelEncoder()

data['outlook'] = Le.fit_transform(data['outlook'])
data['temp'] = Le.fit_transform(data['temp'])
data['humidity'] = Le.fit_transform(data['humidity'])
data['windy'] = Le.fit_transform(data['windy'])
data['play'] = Le.fit_transform(data['play'])
```


```python
y = data['play']
x = data.drop(['play'],axis=1)
```


```python
display(y.head(),x.head())
```


    0    0
    1    0
    2    1
    3    1
    4    1
    Name: play, dtype: int32



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
      <th>outlook</th>
      <th>temp</th>
      <th>humidity</th>
      <th>windy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


#### importing tree / entropy base


```python
from sklearn import tree
play=tree.DecisionTreeClassifier(criterion='entropy')
```


```python
play.fit(x,y)
```




    DecisionTreeClassifier(criterion='entropy')




```python
plt.figure(figsize=(15,5))
tree.plot_tree(play)
plt.show()
```


 
 ![output_22_0](https://user-images.githubusercontent.com/122164879/215850214-98be0eb6-cdcf-4cd3-960e-392364ff8107.png)


    



```python
y_pred = play.predict(x)
```

#### evaluation


```python
from sklearn.metrics import classification_report
print(classification_report(y,y_pred))
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00         5
               1       1.00      1.00      1.00         9
    
        accuracy                           1.00        14
       macro avg       1.00      1.00      1.00        14
    weighted avg       1.00      1.00      1.00        14
    
    


```python

```
