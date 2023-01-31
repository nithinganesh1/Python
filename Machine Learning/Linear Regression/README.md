# Linear regression
1. Linear Regression is a machine learning algorithm based on supervised learning.
2. Regrssion model predict a dependent variable value (y) based on a given independent variable (x).
3. It is used for predicting the continuous dependent variable with the help of independent variables.
4. goal: find the best fit line that can accurately predict the output for the continuous dependent variable

1. Simple Linear Regression
        there is only single indipendent variable
    
2. Multiple Linear Regression
        there are two or more inipendent variable

### Assumptions of feature selection
1. The relationship between the independent and dependent variable should be linear.

2. The independent variables should not be highly correlated with each other.

3. The model should not omit any important independent variables.

4. The observations used to estimate the relationship should be independent of each other.

5. The error should be normally distributed.

6. The variance of error should be constant v(e)=K

7. E(e)=0
# Linear Regression Code Sample

### Step 1 | Data Pre-Processing

#### train_test_split


```python
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
```

### Step 2 | Linear Regression Model

##### Fitting Linear Regression to the Training set


```python
from sklearn.linear_model import LinearRegression
obj=LinearRegression()

#train the model
obj.fit(X_train,y_train)
```

### Step 3 | Predection


```python
y_pred=pd.DataFrame(obj.predict(X_test))
```

### Step 4 | Evaluating The Predection


```python
from sklearn.metrics import r2_score
r2_score(y_test, y_pred)
```


```python
obj.score(X_test,y_test)
```
