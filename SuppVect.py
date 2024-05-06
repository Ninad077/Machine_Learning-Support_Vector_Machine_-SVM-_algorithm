
# Reading the files
import numpy as np
import pandas as pd
import joblib

# 2. Reading the training & testing data

train_data = pd.read_csv(r'risk_analytics_train.csv',index_col=0,header=0)
test_data = pd.read_csv(r'risk_analytics_test.csv',index_col=0,header=0)


# 3. Preprocessing
# 3.1 Checking the dimensions

print(train_data.shape)
print(test_data.shape)

#Preview the train & test data

train_data.head()
test_data.head()


# 3.2 Null value & special characters treatment
#Check for null values
train_data.isnull().sum()
test_data.isnull().sum()

# Treating the null values
# Inputing Categorical missing data with "mode" value

colname1 = ["Gender","Married","Dependents","Self_Employed","Loan_Amount_Term"]

for x in colname1:
    train_data[x].fillna(train_data[x].mode()[0],inplace=True)
    test_data[x].fillna(test_data[x].mode()[0],inplace=True)
    
print(train_data.isnull().sum())
print(test_data.isnull().sum())


# Inputing the numeric values with "mean" value

train_data["LoanAmount"].fillna(round(train_data["LoanAmount"].mean(),0),inplace = True)
test_data["LoanAmount"].fillna(round(test_data["LoanAmount"].mean(),0),inplace = True)

print(train_data.isnull().sum())
print(test_data.isnull().sum())


# So from above we got to know that we are still yet to replace the null values from credit history

train_data["Credit_History"].fillna(value=0,inplace = True)
test_data["Credit_History"].fillna(value=0,inplace = True)

print(train_data.isnull().sum())
print(test_data.isnull().sum())


# 3.3 Label Encoding

#Converting all the categorical data into numerical for train_data

from sklearn.preprocessing import LabelEncoder

colname = ["Gender","Married","Education","Self_Employed","Dependents","Self_Employed","Property_Area","Loan_Status"]

le = LabelEncoder()

for x in colname:
    train_data[x]=le.fit_transform(train_data[x])

train_data.head()


#Converting all the categorical data into numerical for test_data

from sklearn.preprocessing import LabelEncoder

colname = ["Gender","Married","Education","Self_Employed","Property_Area"]

le = LabelEncoder()

for x in colname:
    test_data[x]=le.fit_transform(test_data[x])


test_data.head()


# 3.4 Checking Correlation

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(20,30))
sns.heatmap(train_data.corr(),vmin=-1.0,vmax=1.0,annot=True)


# 4. Splitting Training & Testing data

"""One thing we have to remember here is that the train & test data are already split and given to us.
    So here instead of splitting in X & Y and then applying train test split, we are directly defining our 
    X_train & Y_train.
    In the code below astype(int) will retuen the integer value of the last column of train data
    (since Y_train = train_data.values[:,-1] )"""

X_train = train_data.values[:,0:-1]
Y_train = train_data.values[:,-1]
Y_train = Y_train.astype(int)

X_train.shape


"""We are NOT splitting the test data here since we do not have any dependent column y present.
    Here we are keeping the test data as it is"""

X_test = test_data.values[:,:]

X_test.shape


# 5. Scaling the data

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

print(X_train)

print(X_test)

"""In SVM we have 3 hyperparameters: C (Cost penalty), Kernel & gamma.
    The values .... are base values of the model & if one is changing these base values, then one is performing
    Hyperparameter tuning. Hyperparameter tuning improves the accuracy of the model."""

from sklearn.svm import SVC
svc_model = SVC(kernel ='rbf',C=20, gamma =0.01)
svc_model.fit(X_train,Y_train)


# 7. Predicting Y based on X_test


Y_pred = svc_model.predict(X_test)
print(list(Y_pred))


# 8. Evaluation of the model

svc_model.score(X_train,Y_train)

from sklearn.metrics import confusion_matrix, classification_report

Y_pred_new=svc_model.predict(X_train)
confusion_matrix(Y_train, Y_pred_new)

print (classification_report(Y_train, Y_pred_new) )

""""Here we are not getting any accuracy score, because our test data do not consist of any Y-values"""


# 9. Adding Y-predictions column to the test data to display Eligibilty/Non-Eligbility

test_data=pd.read_csv(r'risk_analytics_test.csv', header=0)
test_data


test_data["Y_predictions"]=Y_pred
test_data.head()

# 10. Mapping Eligibilty: "1"/Non-Eligbility: "0" in Y_predictions col

test_data["Y_predictions"]=test_data["Y_predictions"]. map({1:"Eligible", 0:"Not Eligible"})
test_data.head()


# 11. Outsourcing the Loan elibilty test data to the client

test_data.to_csv(r'test_data_output.csv',index=False)


test_data.Y_predictions.value_counts()


Y_pred_new=svc_model.predict(X_train)
joblib.dump(svc_model,"mymodel.joblib")

print("Model trained successfully...")
