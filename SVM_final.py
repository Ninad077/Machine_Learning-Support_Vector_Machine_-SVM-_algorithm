#1. Importing the libraries
import numpy as np
import pandas as pd
import joblib

#2. Read the training & testing data
train_data = pd.read_csv(r'risk_analytics_train.csv',index_col=0,header=0)
test_data=pd.read_csv(r'risk_analytics_test.csv',index_col=0, header=0)

#Preprocessing the training data

print (train_data.shape)
print(test_data.shape)


train_data. head()

test_data.head()
train_data.head()
train_data.isnull().sum()
test_data.isnull().sum()
for i in train_data.columns:
    print({i:train_data[i].unique()})
for i in test_data.columns:
    print({i:train_data[i].unique()})
#imputing categorical missing data with mode value

colname1=["Gender", "Married", "Dependents", "Self_Employed", "Loan_Amount_Term"]

for x in colname1:
    train_data[x]. fillna(train_data [x].mode () [0], inplace=True)
    test_data[x]. fillna(test_data[x].mode() [0], inplace=True)

    print(train_data.isnull() .sum())

    print(test_data.isnull().sum())

#imputing numerical missing data with mean value
train_data["LoanAmount"]. fillna(round(train_data["LoanAmount"].mean(), 0) , inplace=True)
test_data["LoanAmount"]. fillna(round(test_data ["LoanAmount"].mean(), 0), inplace=True)

print(train_data.isnull() .sum())

print(test_data.isnull().sum())

#imputing numerical missing data with mean value
train_data["LoanAmount"]. fillna(round(train_data["LoanAmount"].mean(), 0) , inplace=True)
test_data["LoanAmount"]. fillna(round(test_data["LoanAmount"].mean(), 0), inplace=True)

print(train_data.isnull().sum())

print(test_data.isnull().sum())

#imputing values for Credit_History column differently
train_data["Credit_History"]. fillna(value=0, inplace=True)
test_data["Credit_History"]. fillna(value=0, inplace=True)

print(train_data.isnull().sum())

print(test_data.isnull().sum() )

#transforming Categorical data to Numerical
from sklearn. preprocessing import LabelEncoder

colname=["Gender", "Married", "Education", "Self_Employed", "Property_Area", "Loan_Status"]

le=LabelEncoder()

for x in colname:
    train_data[x]=le.fit_transform(train_data [x])
train_data.head()

#transforming Categorical data to Numerical
from sklearn. preprocessing import LabelEncoder

colname=["Gender", "Married", "Education", "Self_Employed", "Property_Area"]

le=LabelEncoder()

for x in colname:
    test_data[x]=le.fit_transform(test_data [x])
test_data.head()
corr_df=train_data.corr()
corr_df

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(20,30))
sns.heatmap(corr_df,vmin =- 1.0,vmax=1.0, annot=True)

plt. show()

# Creating training and testing datasets and running the model
X_train=train_data.values [:,0 :- 1]
Y_train=train_data.values [:,-1]
Y_train=Y_train.astype(int)
X_train.shape
Y_train.shape
#test_data.head()

#not to split into X & Y bcoz there are no dependent variable or Y column present in test data 
X_test=test_data.values[:,:]
X_test.shape
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

scaler.fit(X_train)
X_train = scaler.transform(X_train)

X_test = scaler.transform(X_test)
print(X_train)
from sklearn.svm import SVC
svc_model=SVC(kernel='rbf', C=20, gamma=0.01)
svc_model.fit(X_train, Y_train)
Y_pred=svc_model.predict(X_test)
print (list(Y_pred) )
svc_model.score(X_train, Y_train)

#score -- >
#Y_pred=svc_model.predict(X_train)
#accuracy_score(Y_train, Y_pred)
#Y -- >1 -- >Eligible
#N -- >0 -- >Not Eligible
from sklearn.metrics import confusion_matrix, classification_report

Y_pred_new=svc_model.predict(X_train)
confusion_matrix(Y_train, Y_pred_new)

print (classification_report(Y_train, Y_pred_new) )

#We wont find the accuracy score bcoz our test data does not consist of Y values
test_data=pd.read_csv(r'risk_analytics_test.csv', header=0)
test_data["Y_predictions"]=Y_pred
test_data. head ()

test_data["Y_predictions"]=test_data["Y_predictions"]. map({1:"Eligible", 0:"Not Eligible"})
test_data.head()
test_data.to_csv(r'test_data_output.csv',index=False)
test_data.Y_predictions.value_counts()

