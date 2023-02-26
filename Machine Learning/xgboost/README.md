### xgboost Social_Network_Ads dataset

### sample code


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```


```python
data=pd.read_csv(r"E:\PYTHON\python Datasets\Social_Network_Ads.csv")
```


```python
X = data.iloc[:, [2, 3]].values
y = data.iloc[:, 4].values
```


```python
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size = 0.25, random_state = 0)
```


```python
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()#when huge variable is scaled it changing vary match then use this
X_train=sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```


```python
from xgboost import XGBClassifier
from xgboost import plot_importance
```


```python
model=XGBClassifier(n_estimators=200,max_depth=1)
model.fit(X_train,y_train)
```




    XGBClassifier(base_score=None, booster=None, callbacks=None,
                  colsample_bylevel=None, colsample_bynode=None,
                  colsample_bytree=None, early_stopping_rounds=None,
                  enable_categorical=False, eval_metric=None, feature_types=None,
                  gamma=None, gpu_id=None, grow_policy=None, importance_type=None,
                  interaction_constraints=None, learning_rate=None, max_bin=None,
                  max_cat_threshold=None, max_cat_to_onehot=None,
                  max_delta_step=None, max_depth=1, max_leaves=None,
                  min_child_weight=None, missing=nan, monotone_constraints=None,
                  n_estimators=200, n_jobs=None, num_parallel_tree=None,
                  predictor=None, random_state=None, ...)




```python
y_pred=model.predict(X_test)
```


```python
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
```

                  precision    recall  f1-score   support
    
               0       0.96      0.96      0.96        68
               1       0.91      0.91      0.91        32
    
        accuracy                           0.94       100
       macro avg       0.93      0.93      0.93       100
    weighted avg       0.94      0.94      0.94       100
    
    


```python
plot_importance(model)
plt.show()
```


    
![output_11_0](https://user-images.githubusercontent.com/122164879/221417629-1b21797d-2f87-484b-8a55-14002c82d221.png)

    



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




```python
model.feature_importances_
```




    array([0.71431464, 0.2856854 ], dtype=float32)



