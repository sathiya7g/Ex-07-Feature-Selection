## Ex-07 Feature Selection

## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

## Explanation
Feature selection is to find the best set of features that allows one to build useful models.Selecting the best features helps the model to perform well. 

## ALGORITHM
### STEP 1

Read the given Data

### STEP 2

Clean the Data Set using Data Cleaning Process

### STEP 3

Apply Feature selection techniques to all the features of the data set

### STEP 4

Save the data to the file


## CODE

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

df=pd.read_csv('/content/titanic_dataset.csv')

df.head()

df.isnull().sum()

df.drop('Cabin',axis=1,inplace=True)

df.drop('Name',axis=1,inplace=True)

df.drop('Ticket',axis=1,inplace=True)

df.drop('PassengerId',axis=1,inplace=True)

df.drop('Parch',axis=1,inplace=True)

df.head()

df['Age']=df['Age'].fillna(df['Age'].median())

df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])

df.isnull().sum()

plt.title("Dataset with outliers")

df.boxplot()

plt.show()

cols = ['Age','SibSp','Fare']

Q1 = df[cols].quantile(0.25)

Q3 = df[cols].quantile(0.75)

IQR = Q3 - Q1

df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

plt.title("Dataset after removing outliers")

df.boxplot()

plt.show()

from sklearn.preprocessing import OrdinalEncoder

climate = ['C','S','Q']

en= OrdinalEncoder(categories = [climate])

df['Embarked']=en.fit_transform(df[["Embarked"]])

df

climate = ['male','female']

en= OrdinalEncoder(categories = [climate])

df['Sex']=en.fit_transform(df[["Sex"]])

df

from sklearn.preprocessing import RobustScaler

sc=RobustScaler()

df=pd.DataFrame(sc.fit_transform(df),columns=['Survived','Pclass','Sex','Age','SibSp','Fare','Embarked'])

df

import statsmodels.api as sm

import numpy as np

import scipy.stats as stats

from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal',n_quantiles=692)

df1=pd.DataFrame()

df1["Survived"]=np.sqrt(df["Survived"])

df1["Pclass"],parameters=stats.yeojohnson(df["Pclass"])

df1["Sex"]=np.sqrt(df["Sex"])

df1["Age"]=df["Age"]

df1["SibSp"],parameters=stats.yeojohnson(df["SibSp"])

df1["Fare"],parameters=stats.yeojohnson(df["Fare"])

df1["Embarked"]=df["Embarked"]

df1.skew()

import matplotlib

import seaborn as sns

import statsmodels.api as sm

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

X = df1.drop("Survived",1) 

y = df1["Survived"]   

plt.figure(figsize=(12,10))

cor = df1.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.RdPu)

plt.show()

cor_target = abs(cor["Survived"])

relevant_features = cor_target[cor_target>0.5]

relevant_features

X_1 = sm.add_constant(X)

model = sm.OLS(y,X_1).fit()

model.pvalues

cols = list(X.columns)

pmax = 1

while (len(cols)>0):

    p= []
    
    X_1 = X[cols]
    
    X_1 = sm.add_constant(X_1)
    
    model = sm.OLS(y,X_1).fit()
    
    p = pd.Series(model.pvalues.values[1:],index = cols)  
    
    pmax = max(p)
    
    feature_with_p_max = p.idxmax()
    
    if(pmax>0.05):
    
        cols.remove(feature_with_p_max)
        
    else:
    
        break
        
selected_features_BE = cols

print(selected_features_BE)

model = LinearRegression()

rfe = RFE(model,step= 4)

X_rfe = rfe.fit_transform(X,y)  

model.fit(X_rfe,y)

print(rfe.support_)

print(rfe.ranking_)

nof_list=np.arange(1,6)   

high_score=0

nof=0    

score_list =[]

for n in range(len(nof_list)):

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
    
    model = LinearRegression()
    
    rfe = RFE(model,step=nof_list[n])
    
    X_train_rfe = rfe.fit_transform(X_train,y_train)
    
    X_test_rfe = rfe.transform(X_test)
    
    model.fit(X_train_rfe,y_train)
    
    score = model.score(X_test_rfe,y_test)
    
    score_list.append(score)
    
    if(score>high_score):
    
        high_score = score
        
        nof = nof_list[n]
        
print("Optimum number of features: %d" %nof)

print("Score with %d features: %f" % (nof, high_score))

cols = list(X.columns)

model = LinearRegression()

rfe = RFE(model, step=2) 

X_rfe = rfe.fit_transform(X,y)  

model.fit(X_rfe,y)        

temp = pd.Series(rfe.support_,index = cols)

selected_features_rfe = temp[temp==True].index

print(selected_features_rfe)

reg = LassoCV()

reg.fit(X, y)

print("Best alpha using built-in LassoCV: %f" % reg.alpha_)

print("Best score using built-in LassoCV: %f" %reg.score(X,y))

coef = pd.Series(reg.coef_, index = X.columns)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

imp_coef = coef.sort_values()

import matplotlib

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Feature importance using Lasso Model")

plt.show()

## OUPUT

![op1](https://user-images.githubusercontent.com/112301582/236690606-a1c6b1b4-2e35-4d4d-a378-c6418ede4a16.png)
![op2](https://user-images.githubusercontent.com/112301582/236690619-dbefe79c-1204-435a-a0e4-a6c978ecccee.png)
![op3](https://user-images.githubusercontent.com/112301582/236690622-1a3bde39-fe10-4e4c-96b8-38f2068708b0.png)
![op4](https://user-images.githubusercontent.com/112301582/236690626-b9b72b04-a5f3-44d2-b13f-e4b6bc69eaab.png)
![op5](https://user-images.githubusercontent.com/112301582/236690628-01c63d83-46a3-4201-aa25-badcc8122cee.png)
![op6](https://user-images.githubusercontent.com/112301582/236690631-2400b5ec-1a06-4dc3-b0d2-f674bc332c25.png)
![op7](https://user-images.githubusercontent.com/112301582/236690634-7369159f-5903-44b7-9389-8369b7212161.png)
![op8](https://user-images.githubusercontent.com/112301582/236690635-e03618d8-864f-4501-a320-b80a12251995.png)
![op9](https://user-images.githubusercontent.com/112301582/236690637-83e0c1be-31cb-4b95-a95c-31844c66582a.png)
![op10](https://user-images.githubusercontent.com/112301582/236690638-7ca0edec-5fdd-4b37-ae73-1f2a6a3d973b.png)
![op11](https://user-images.githubusercontent.com/112301582/236690649-5459ac2f-368d-44ca-a81a-1a46bc39cdb0.png)
![op12](https://user-images.githubusercontent.com/112301582/236690654-ba155e60-03e0-44f6-a310-a2a04ef35ca0.png)
![op13](https://user-images.githubusercontent.com/112301582/236690657-d7f26b8a-89da-42da-b729-22b22c06a1dc.png)
![op14](https://user-images.githubusercontent.com/112301582/236690658-8360e36a-1424-417f-b7c9-cb6516bc3847.png)
![op15](https://user-images.githubusercontent.com/112301582/236690660-a220a12d-89b8-4722-b0ff-f061ad938e7b.png)

## RESULT

The various feature selection techniques are performed on a dataset and saved the data to a file. 

