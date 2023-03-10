
# Random Forests

### Sample Code

#### Social_Network_Ads Datasets

> **__**

1. i am already done eda and other steps in social network ads in decision tree
2. so here i am only doing Random Forest Sample code


```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```


```python
dataset=pd.read_csv(r"E:\PYTHON\python Datasets\Social_Network_Ads.csv")
dataset.head()
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
      <th>User ID</th>
      <th>Gender</th>
      <th>Age</th>
      <th>EstimatedSalary</th>
      <th>Purchased</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>15624510</td>
      <td>Male</td>
      <td>19</td>
      <td>19000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>15810944</td>
      <td>Male</td>
      <td>35</td>
      <td>20000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15668575</td>
      <td>Female</td>
      <td>26</td>
      <td>43000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15603246</td>
      <td>Female</td>
      <td>27</td>
      <td>57000</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>15804002</td>
      <td>Male</td>
      <td>19</td>
      <td>76000</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### train test


```python
X = dataset.iloc[:, [2, 3]].values
y = dataset.Purchased.values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
```

### Feature Scaling


```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

### Random Forest


```python
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 15)
classifier.fit(X_train, y_train)
```




    RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=15)



### Predicting


```python
y_pred = classifier.predict(X_test)
```

### Confusion Matrix


```python
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,y_pred))
```

    [[64  4]
     [ 3 29]]
    


```python
print(classification_report(y_test,y_pred))
```

                  precision    recall  f1-score   support
    
               0       0.96      0.94      0.95        68
               1       0.88      0.91      0.89        32
    
        accuracy                           0.93       100
       macro avg       0.92      0.92      0.92       100
    weighted avg       0.93      0.93      0.93       100
    
    


```python
from sklearn.metrics import roc_curve, roc_auc_score

# calculate the ROC curve and AUC score
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred)

# plot the ROC curve
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}',c='g')
plt.plot([0, 1], [0, 1], '--',c='m')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()
```


![output_17_0](https://user-images.githubusercontent.com/122164879/218781387-10f8b319-76f8-414b-ae9b-a66b75ae000e.png)



    

