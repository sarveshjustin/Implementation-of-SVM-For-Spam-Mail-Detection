# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
Import the required packages.
Import the dataset to operate on.
Split the dataset.
Predict the required output.
End the program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: sarvesh.s
RegisterNumber:  212222230135
*/
import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extractiaon.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
### Data Head:
![173077929-279a193a-55f7-4de7-b705-e7260abc5290](https://github.com/Afsarjumail/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118343395/35bb4548-eeaa-4efe-9130-14fbbbfe82e4)


### Data Info:
![173077947-8ca5a120-b620-4691-8485-70c09b0e6255](https://github.com/Afsarjumail/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118343395/4c542d8b-1d5b-4e1c-9609-0604c7b22cb4)


### Data isnull():
![image](https://github.com/Afsarjumail/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118343395/2a4c9cd4-be61-46e4-8596-e2601bba9f9f)


### y_pred:
![173077974-78d5cb5d-6b93-4039-9dfe-34f08aee366b](https://github.com/Afsarjumail/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118343395/9db820a3-77ca-4c28-b2c3-96b9f809d3d6)



### Accuracy:
![image](https://github.com/Afsarjumail/Implementation-of-SVM-For-Spam-Mail-Detection/assets/118343395/5f4283c1-8b82-478c-a3e8-4e666da8c2af)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
