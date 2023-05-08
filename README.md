# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file

# CODE
```
Program Developed By:Sathiya Narayanan G
Register number:212221220049
from sklearn.datasets import load_boston
boston_data=load_boston()
import pandas as pd
boston = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
boston['MEDV'] = boston_data.target
dummies = pd.get_dummies(boston.RAD)
boston = boston.drop(columns='RAD').merge(dummies,left_index=True,right_index=True)
X = boston.drop(columns='MEDV')
y = boston.MEDV
boston.head(10)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from math import sqrt

cv = KFold(n_splits=10, random_state=None, shuffle=False)
classifier_pipeline = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10))
y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)
print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),2)))
print("R_squared: " + str(round(r2_score(y,y_pred),2)))

boston.var()

X = X.drop(columns = ['NOX','CHAS'])
y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)
print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),2)))
print("R_squared: " + str(round(r2_score(y,y_pred),2)))

# Filter Features by Correlation
import seaborn as sn
import matplotlib.pyplot as plt
fig_dims = (12, 8)
fig, ax = plt.subplots(figsize=fig_dims)
sn.heatmap(boston.corr(), ax=ax)
plt.show()
abs(boston.corr()["MEDV"])
abs(boston.corr()["MEDV"][abs(boston.corr()["MEDV"])>0.5].drop('MEDV')).index.tolist()
vals = [0.1,0.2,0.3,0.4,0.5,0.6,0.7]
for val in vals:
    features = abs(boston.corr()["MEDV"][abs(boston.corr()["MEDV"])>val].drop('MEDV')).index.tolist()
    
    X = boston.drop(columns='MEDV')
    X=X[features]
    
    print(features)

    y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)
    print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),2)))
    print("R_squared: " + str(round(r2_score(y,y_pred),2)))

# Feature Selection Using a Wrapper

boston = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
boston['MEDV'] = boston_data.target
boston['RAD'] = boston['RAD'].astype('category')
dummies = pd.get_dummies(boston.RAD)
boston = boston.drop(columns='RAD').merge(dummies,left_index=True,right_index=True)
X = boston.drop(columns='MEDV')
y = boston.MEDV

from mlxtend.feature_selection import SequentialFeatureSelector as SFS

sfs1 = SFS(classifier_pipeline, 
           k_features=1, 
           forward=False, 
           scoring='neg_mean_squared_error',
           cv=cv)

X = boston.drop(columns='MEDV')
sfs1.fit(X,y)
sfs1.subsets_

X = boston.drop(columns='MEDV')[['CRIM','RM','PTRATIO','LSTAT']]
y = boston['MEDV']
y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)
print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),3)))
print("R_squared: " + str(round(r2_score(y,y_pred),3)))

boston[['CRIM','RM','PTRATIO','LSTAT','MEDV']].corr()

boston['RM*LSTAT']=boston['RM']*boston['LSTAT']

X = boston.drop(columns='MEDV')[['CRIM','RM','PTRATIO','LSTAT']]
y = boston['MEDV']
y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)
print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),3)))
print("R_squared: " + str(round(r2_score(y,y_pred),3)))

sn.pairplot(boston[['CRIM','RM','PTRATIO','LSTAT','MEDV']])

boston = boston.drop(boston[boston['MEDV']==boston['MEDV'].max()].index.tolist())

X = boston.drop(columns='MEDV')[['CRIM','RM','PTRATIO','LSTAT','RM*LSTAT']]
y = boston['MEDV']
y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)
print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),3)))
print("R_squared: " + str(round(r2_score(y,y_pred),3)))

boston['LSTAT_2']=boston['LSTAT']**2

X = boston.drop(columns='MEDV')[['CRIM','RM','PTRATIO','LSTAT']]
y_pred = cross_val_predict(classifier_pipeline, X, y, cv=cv)
print("RMSE: " + str(round(sqrt(mean_squared_error(y,y_pred)),3)))
print("R_squared: " + str(round(r2_score(y,y_pred),3)))
```


# OUPUT
![image](https://user-images.githubusercontent.com/94505585/170406591-47feac04-6cc9-4892-b5c3-433f1d8d50bb.png)
![image](https://user-images.githubusercontent.com/94505585/170406697-6a97d8fd-b310-4352-a30a-c1adfc4a0bcb.png)
![image](https://user-images.githubusercontent.com/94505585/170406735-a7d9e5a9-c2fa-4879-ac98-26abb47d0ee4.png)
![image](https://user-images.githubusercontent.com/94505585/170406761-e9529da2-9b56-4590-bb4f-6d52cf94af8b.png)
![image](https://user-images.githubusercontent.com/94505585/170406784-a6c8952a-a730-477e-9415-a8e3409866fc.png)
![image](https://user-images.githubusercontent.com/94505585/170406804-8adcb545-9eb2-47f4-a344-c444b038c0a5.png)
![image](https://user-images.githubusercontent.com/94505585/170406816-235fd4c5-f4f9-4cc7-9054-c284c6925144.png)
![image](https://user-images.githubusercontent.com/94505585/170406839-a4534b51-9a1f-4628-90ba-38393d30e65f.png)
![image](https://user-images.githubusercontent.com/94505585/170406858-b2d8e732-8d3a-4284-8e65-9737d5badc74.png)
![image](https://user-images.githubusercontent.com/94505585/170406874-74641eeb-e3e5-45c4-bc29-9b13309261e0.png)
![image](https://user-images.githubusercontent.com/94505585/170406890-c590bde6-d009-47ad-9a3d-2ec4f3c16781.png)
![image](https://user-images.githubusercontent.com/94505585/170406901-b9ab21e5-11ed-4ccb-9f7a-3ef0aaa5dd62.png)
![image](https://user-images.githubusercontent.com/94505585/170406920-1ed23c3b-fd72-41fb-a5e4-6caf586b9620.png)
![image](https://user-images.githubusercontent.com/94505585/170406931-60102e9b-3b65-4466-b683-f5941fe113b3.png)
![image](https://user-images.githubusercontent.com/94505585/170406949-019df07e-1ef1-49fd-880a-6b84c914d1bf.png)
![image](https://user-images.githubusercontent.com/94505585/170406966-4fafb6f2-64b9-4b47-84ef-b532d498f145.png)

### RESULT
The various feature selection techniques has been performed on a dataset and saved the data to a file.












