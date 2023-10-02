# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 1. Import the standard libraries.
 2. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
 3. Import LabelEncoder and encode the dataset.
 4. Import LogisticRegression from sklearn and apply the model on the dataset.
 5. Predict the values of array.
 6. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
 7.  Apply new unknown values

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
# Developed by: PRAISEY S
# RegisterNumber:  212222040117
# Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Read The File
dataset=pd.read_csv('Placement_Data_Full_Class.csv')
dataset
dataset.head(10)
dataset.tail(10)
# Dropping the serial number and salary column
dataset=dataset.drop(['sl_no','ssc_p','workex','ssc_b'],axis=1)
dataset
dataset.shape
dataset.info()
dataset["gender"]=dataset["gender"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset.info()
dataset["gender"]=dataset["gender"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset.info()
dataset
# selecting the features and labels
x=dataset.iloc[:, :-1].values
y=dataset.iloc[: ,-1].values
y
# dividing the data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
dataset.head()
y_train.shape
x_train.shape
# Creating a Classifier using Sklearn
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(random_state=0,solver='lbfgs',max_iter=1000).fit(x_train,y_train)
# Printing the acc
clf=LogisticRegression()
clf.fit(x_train,y_train)
clf.score(x_test,y_test)
# Predicting for random value
clf.predict([[1	,78.33,	1,	2,	77.48,	2,	86.5,	0,	66.28]])  
*/
```

## Output:

READ CSV FILE:

![out 4 1](https://github.com/PRAISEYSOLOMON/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394259/36d3f1bd-59af-41bf-87d8-46b4aad61248)

TO READ DATA(HEAD):

![out 4 2](https://github.com/PRAISEYSOLOMON/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394259/39800032-031b-4a4a-9a38-f981eea6b941)

TO READ DATA(TAIL):

![out 4 3](https://github.com/PRAISEYSOLOMON/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394259/75b557e0-eaa3-472c-93a4-37646f5f76a7)

Dropping the serial number and salary column:

![out 4 4](https://github.com/PRAISEYSOLOMON/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394259/6dbab79f-eeec-425f-a224-9c308facfadf)

Dataset Information:

![out 4 5](https://github.com/PRAISEYSOLOMON/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394259/34ede9ec-a964-4177-86b4-b7853227a1fd)

Dataset after changing object into category:

![out 4 6](https://github.com/PRAISEYSOLOMON/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394259/07c00c9d-4f15-4ae8-93c7-a56abbb6d706)

Dataset after changing category into integer:

![out 4 7](https://github.com/PRAISEYSOLOMON/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394259/47f33abe-1932-4ae7-b6f2-0563f533851a)

Selecting the features and labels:

![out 4 8](https://github.com/PRAISEYSOLOMON/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394259/ce812540-5a0b-4286-8048-e54f4add0980)

Dividing the data into train and test:

![out 4 9](https://github.com/PRAISEYSOLOMON/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394259/eb12e1e9-0895-47c9-9c13-d8d968eaa885)

Creating a Classifier using Sklearn:

![out 4 10](https://github.com/PRAISEYSOLOMON/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394259/635a1808-d793-413a-bea8-515e2319b5d6)

Predicting for random value:

![out 4 11](https://github.com/PRAISEYSOLOMON/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119394259/bb0af8b2-628d-4aab-b839-ecc0ea9b9990)


## Result:

Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
