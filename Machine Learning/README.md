#Machine Learning Sample code

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
```

### Importing Dataset


```python
data=pd.read_csv(r"C:\Users\Nithin\OneDrive\Desktop\iris.csv")
```


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
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.shape
```




    (150, 5)




```python
data.size
```




    750




```python
data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 150 entries, 0 to 149
    Data columns (total 5 columns):
     #   Column        Non-Null Count  Dtype  
    ---  ------        --------------  -----  
     0   sepal_length  150 non-null    float64
     1   sepal_width   150 non-null    float64
     2   petal_length  150 non-null    float64
     3   petal_width   150 non-null    float64
     4   species       150 non-null    object 
    dtypes: float64(4), object(1)
    memory usage: 6.0+ KB
    


```python
from pandas_profiling import ProfileReport
Report=ProfileReport(data)
Report.to_file(output_file='irisReport.html')
```


    Summarize dataset:   0%|          | 0/5 [00:00<?, ?it/s]



    Generate report structure:   0%|          | 0/1 [00:00<?, ?it/s]



    Render HTML:   0%|          | 0/1 [00:00<?, ?it/s]



    Export report to file:   0%|          | 0/1 [00:00<?, ?it/s]

1. data have 5 columns and 150 rows
2. zero missing values
3. two duplicated rows  

```python
data.drop_duplicates(keep='last',inplace=True)
```


```python
data.describe().T
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
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>sepal_length</th>
      <td>147.0</td>
      <td>5.856463</td>
      <td>0.829100</td>
      <td>4.3</td>
      <td>5.1</td>
      <td>5.8</td>
      <td>6.4</td>
      <td>7.9</td>
    </tr>
    <tr>
      <th>sepal_width</th>
      <td>147.0</td>
      <td>3.055782</td>
      <td>0.437009</td>
      <td>2.0</td>
      <td>2.8</td>
      <td>3.0</td>
      <td>3.3</td>
      <td>4.4</td>
    </tr>
    <tr>
      <th>petal_length</th>
      <td>147.0</td>
      <td>3.780272</td>
      <td>1.759111</td>
      <td>1.0</td>
      <td>1.6</td>
      <td>4.4</td>
      <td>5.1</td>
      <td>6.9</td>
    </tr>
    <tr>
      <th>petal_width</th>
      <td>147.0</td>
      <td>1.208844</td>
      <td>0.757874</td>
      <td>0.1</td>
      <td>0.3</td>
      <td>1.3</td>
      <td>1.8</td>
      <td>2.5</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.groupby('species').size()
```




    species
    Iris-setosa        48
    Iris-versicolor    50
    Iris-virginica     49
    dtype: int64



### Visualisation


```python
data.groupby('species').size().plot(kind='bar',color='g',figsize=(4,4))
plt.show()
```


    
![output_13_0](https://user-images.githubusercontent.com/122164879/216912739-b3bff7fe-e2de-48cc-bbdf-dd79a4cd7f98.png)

    



```python
data.plot(kind='box',subplots=True,layout=(2,2),figsize=(8,5),color='g')
plt.show()
```


    

![output_14_0](https://user-images.githubusercontent.com/122164879/216912790-f1214d0a-3525-4249-8f8b-3f27ca18a54a.png)




```python
sns.heatmap(data.corr(),annot=True,cmap=('Greens'))
```




    <AxesSubplot:>




    
![output_15_1](https://user-images.githubusercontent.com/122164879/216912856-93c3f0e4-510b-4a77-bcde-d2204ad0b733.png)

    



```python
data.hist(layout=(2,2),figsize=(8,5),color='g')
plt.show()
```


    
![output_16_0](https://user-images.githubusercontent.com/122164879/216912912-0bb0e334-c1e2-4126-8df3-b96b3d0cdac6.png)

    



```python
sns.pairplot(data,hue='species',markers=["o", "s", "D"])
plt.show()
```


    
![output_17_0](https://user-images.githubusercontent.com/122164879/216912942-d1ba1975-3a4d-44c7-b1cc-2813f8da73ee.png)

    


### Splitting


```python
x=data.iloc[:,:4]
y=data.iloc[:,4]
```


```python
display(x.sample(3),y.sample(3))
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
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>83</th>
      <td>6.0</td>
      <td>2.7</td>
      <td>5.1</td>
      <td>1.6</td>
    </tr>
    <tr>
      <th>80</th>
      <td>5.5</td>
      <td>2.4</td>
      <td>3.8</td>
      <td>1.1</td>
    </tr>
    <tr>
      <th>65</th>
      <td>6.7</td>
      <td>3.1</td>
      <td>4.4</td>
      <td>1.4</td>
    </tr>
  </tbody>
</table>
</div>



    48        Iris-setosa
    68    Iris-versicolor
    58    Iris-versicolor
    Name: species, dtype: object



```python
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=15)
```


```python
x_train.shape,x_test.shape,y_train.shape,y_test.shape
```




    ((117, 4), (30, 4), (117,), (30,))



### Buliding Models


```python
#Various Machine Learning Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
```


```python
models=[]
models.append(('lr',LogisticRegression(solver='liblinear',multi_class='ovr')))
models.append(('CART',DecisionTreeClassifier()))
models.append(('KNN',KNeighborsClassifier()))
models.append(('LDA',LinearDiscriminantAnalysis()))
models.append(('NB',GaussianNB()))
models.append(('SVM',SVC(gamma='auto')))
```


```python
models
```




    [('lr', LogisticRegression(multi_class='ovr', solver='liblinear')),
     ('CART', DecisionTreeClassifier()),
     ('KNN', KNeighborsClassifier()),
     ('LDA', LinearDiscriminantAnalysis()),
     ('NB', GaussianNB()),
     ('SVM', SVC(gamma='auto'))]




```python
from sklearn import model_selection
names=[]
results=[]
for name, model in models:
    Kfold=model_selection.KFold(n_splits=10,random_state=15,shuffle=True)
    cv_results=model_selection.cross_val_score(model, x_train, y_train, cv=Kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f"{name}: {cv_results.mean()} , {cv_results.std()}")
```

    lr: 0.9393939393939392 , 0.06801325965898011
    CART: 0.9492424242424242 , 0.0415010482569768
    KNN: 0.9484848484848485 , 0.05856408512147784
    LDA: 0.9825757575757574 , 0.03488963315051351
    NB: 0.9484848484848485 , 0.04215281134316231
    SVM: 0.9568181818181818 , 0.07009520614636609
    
1.It is best to use LDA here, i.e., acc = 97.5, and standing diveation is low as well
2.we can Visualise the models

```python
results=pd.DataFrame(results,index=names)
```


```python
results=results.T
```


```python
results
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
      <th>lr</th>
      <th>CART</th>
      <th>KNN</th>
      <th>LDA</th>
      <th>NB</th>
      <th>SVM</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.833333</td>
      <td>0.916667</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.916667</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.000000</td>
      <td>0.916667</td>
      <td>0.916667</td>
      <td>1.000000</td>
      <td>0.916667</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.916667</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.916667</td>
      <td>0.916667</td>
      <td>0.916667</td>
      <td>0.916667</td>
      <td>1.000000</td>
      <td>0.916667</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1.000000</td>
      <td>0.916667</td>
      <td>0.916667</td>
      <td>1.000000</td>
      <td>0.916667</td>
      <td>0.833333</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.916667</td>
      <td>0.916667</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.916667</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>0.909091</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.909091</td>
      <td>0.909091</td>
      <td>0.818182</td>
      <td>0.909091</td>
      <td>0.909091</td>
      <td>0.818182</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.818182</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
results.plot(kind='box',color='g')
```




    <AxesSubplot:>




    
![output_32_1](https://user-images.githubusercontent.com/122164879/216913052-71e47f4b-cf6f-4fc8-903b-069ec961ab6b.png)

    



```python
sns.violinplot(data=results)
plt.title("Violin Plot of Model")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.show()

```


    
![output_33_0](https://user-images.githubusercontent.com/122164879/216913086-918c4037-fc91-4624-b04f-75441c702dfd.png)

    

1. here LDA has maximum accuracy
2. and there is no minumum value only maximum value
3. LDA have 2 outlayers is that outlayers remove then it will give 100%
4. LDA values are distributed around median
### Prediction
###### since Accuracy will not always be the metrics to select best model
LDA

```python
LDA=LinearDiscriminantAnalysis()
LDA.fit(x_train,y_train)
y_pred_LDA=LDA.predict(x_test)
```


```python
y_pred_LDA
```




    array(['Iris-setosa', 'Iris-versicolor', 'Iris-versicolor',
           'Iris-virginica', 'Iris-versicolor', 'Iris-versicolor',
           'Iris-versicolor', 'Iris-setosa', 'Iris-virginica',
           'Iris-versicolor', 'Iris-versicolor', 'Iris-virginica',
           'Iris-setosa', 'Iris-setosa', 'Iris-virginica', 'Iris-setosa',
           'Iris-virginica', 'Iris-virginica', 'Iris-versicolor',
           'Iris-virginica', 'Iris-setosa', 'Iris-virginica',
           'Iris-versicolor', 'Iris-virginica', 'Iris-setosa', 'Iris-setosa',
           'Iris-versicolor', 'Iris-setosa', 'Iris-virginica',
           'Iris-virginica'], dtype='<U15')


SVM

```python
SVM=SVC(gamma='auto')
SVM.fit(x_train,y_train)
y_pred_SVM=SVM.predict(x_test)
```


```python
y_pred_SVM
```




    array(['Iris-setosa', 'Iris-versicolor', 'Iris-versicolor',
           'Iris-virginica', 'Iris-versicolor', 'Iris-versicolor',
           'Iris-versicolor', 'Iris-setosa', 'Iris-virginica',
           'Iris-versicolor', 'Iris-versicolor', 'Iris-virginica',
           'Iris-setosa', 'Iris-setosa', 'Iris-virginica', 'Iris-setosa',
           'Iris-virginica', 'Iris-virginica', 'Iris-versicolor',
           'Iris-virginica', 'Iris-setosa', 'Iris-virginica',
           'Iris-versicolor', 'Iris-virginica', 'Iris-setosa', 'Iris-setosa',
           'Iris-versicolor', 'Iris-setosa', 'Iris-versicolor',
           'Iris-virginica'], dtype=object)



### Evaluation


```python
from sklearn.metrics import confusion_matrix,classification_report
cm_LDA=confusion_matrix(y_test,y_pred_LDA)
print(f"confusion_matrix LDA: \n {cm_LDA} \n ")
print(f" report LDA: \n {classification_report(y_test,y_pred_LDA)}")
```

    confusion_matrix LDA: 
     [[ 9  0  0]
     [ 0 10  1]
     [ 0  0 10]] 
     
     report LDA: 
                      precision    recall  f1-score   support
    
        Iris-setosa       1.00      1.00      1.00         9
    Iris-versicolor       1.00      0.91      0.95        11
     Iris-virginica       0.91      1.00      0.95        10
    
           accuracy                           0.97        30
          macro avg       0.97      0.97      0.97        30
       weighted avg       0.97      0.97      0.97        30
    
    


```python
from sklearn.metrics import confusion_matrix,classification_report
cm_SVM=confusion_matrix(y_test,y_pred_SVM)
print(f"confusion_matrix SVM: \n {cm_SVM} \n ")
print(f" report SVM: \n {classification_report(y_test,y_pred_SVM)}")
```

    confusion_matrix SVM: 
     [[ 9  0  0]
     [ 0 11  0]
     [ 0  0 10]] 
     
     report SVM: 
                      precision    recall  f1-score   support
    
        Iris-setosa       1.00      1.00      1.00         9
    Iris-versicolor       1.00      1.00      1.00        11
     Iris-virginica       1.00      1.00      1.00        10
    
           accuracy                           1.00        30
          macro avg       1.00      1.00      1.00        30
       weighted avg       1.00      1.00      1.00        30
    
    


```python
sns.heatmap(cm_SVM, annot=True, fmt='d', cmap='Greens')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
```


    
![output_45_0](https://user-images.githubusercontent.com/122164879/216913139-a21d75dd-f2ea-47ed-9aee-b86ced1bbde2.png)

    



```python
#Here, SVM predicts better than LDA, so SVM is the best model
```

