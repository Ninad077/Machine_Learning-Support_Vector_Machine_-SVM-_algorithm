{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "71320d9a",
   "metadata": {},
   "source": [
    "# <font color= black>Support Vector Machine (SVM)</font>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "edf49eb3",
   "metadata": {},
   "source": [
    "**<font color = blue>1. When to use SVM?</font>**"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f2f470e1",
   "metadata": {},
   "source": [
    "The major limitation of K-NN (K nearest neighbour) is that it is difficult to categorize data when it is dense. \n",
    "For instance, if the co-ordinates are closely spaced to each other, the machine finds it very difficult to categroize it\n",
    "into one class.\n",
    "This limitation of K-NN is then overcomed using SVM (i.e. Space Vector Machine) algorithm.\n",
    "SVM is used for both classification & regression problems.\n",
    "Primarily, SVM is used for binary & multi-class Classification.\n",
    "\n",
    "<font color = red>**SVM is suited when we have more number of columns than the number of rows.**</font>\n",
    "Such a data is called high-dimensional data. It is the data where number of variables are high as compared to the number\n",
    "of observations.\n",
    "\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5c7ca110",
   "metadata": {},
   "source": [
    "<font color = blue>**2. Linearly and Non-linearly separable data**</font>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "63891ce2",
   "metadata": {},
   "source": [
    "Linearly separable data is the data where you can partition the data into classes simply by putting a line between them.\n",
    "\n",
    "However, there are scenarios where one cannot separate the data by simply putting a linear line. Part of the reason might\n",
    "be due to it's non-linear nature, such data is called as Non-linearly separable data.\n",
    "One example of a Non-linearly separable data is data in concentric patches.\n",
    "\n",
    "SVM works on both Linearly & Non-linearly separable data."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1de6ca8e",
   "metadata": {},
   "source": [
    "**<font color = blue>3. What does one mean by Support Vector?</font>**"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4521ad1d",
   "metadata": {},
   "source": [
    "**<font color = red>The datapoint of a class which is at the extreme ends of the class as well as the one which is closest to another class \n",
    "is called as the Support Vector of that class.</font>**\n",
    "Once we find the Support vectors we draw a hyperplane passing through both these co-ordinates and then draw a margin throughthe centre of this hyperplane, which we define as maximum margin.\n",
    "\n",
    "So in the case of a densely populated data, even if the data points are closely packed we could easily differentiate with\n",
    "the help of this maximum margin. This makes the classification a lot simpler.\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f37af80f",
   "metadata": {},
   "source": [
    "<img src = \"SVM_image.png\">"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ac1b09f7",
   "metadata": {},
   "source": [
    "If we look at the above figure,it tells us clearly how support vectors can be used to create hyperplane & maximum\n",
    "margin which can further classify the data better.\n",
    "The dimensions of the hyperplane depends upon the features present in the dataset, which means if there are 2 features,\n",
    "then the hyperplane will be a straight line, if there are 3 features then the hyperplane will be a 2 dimensional plane.\n",
    "\n",
    "We always create a hyperplane that has maximum margin, which means maximum distance between the data points."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7cc3f38e",
   "metadata": {},
   "source": [
    "**<font color = blue>4. Hyperparameter</font>**"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9f655725",
   "metadata": {},
   "source": [
    "In Machine Learning it is difficult to tune the algorithm based on different datasets. To make it impossible, we tune\n",
    "the model using a hyperparameter, which is basically a user defined variable which can be used to tune the model as per\n",
    "the different datasets, so that we achieve the desired accuracy for that model.\n",
    "\n",
    "In SVM, we have 3 hyperparameters alpha, beta & gamma."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "fff74578",
   "metadata": {},
   "source": [
    "------------------------------------------------------------------------------------------------------------------------"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "86248157",
   "metadata": {},
   "source": [
    "# Code"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4af978c6",
   "metadata": {},
   "source": [
    "### **<font color = blue>1. Importing the Libraries</font>**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "618bc7d3",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "92310d76",
   "metadata": {},
   "source": [
    "### **<font color = blue>2. Reading the training & testing data</font>**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "a92931d1",
   "metadata": {},
   "outputs": [],
   "source": [
    "train_data = pd.read_csv(r'risk_analytics_train.csv',index_col=0,header=0)\n",
    "test_data = pd.read_csv(r'risk_analytics_test.csv',index_col=0,header=0)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e14539e1",
   "metadata": {},
   "source": [
    "### **<font color = blue>3. Preprocessing</font>**"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "14d3b0c8",
   "metadata": {},
   "source": [
    "#### **<font color = green>3.1 Checking the dimensions</font>**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "60b29221",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(614, 12)\n",
      "(367, 11)\n"
     ]
    }
   ],
   "source": [
    "print(train_data.shape)\n",
    "print(test_data.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "b080ab9e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Gender</th>\n",
       "      <th>Married</th>\n",
       "      <th>Dependents</th>\n",
       "      <th>Education</th>\n",
       "      <th>Self_Employed</th>\n",
       "      <th>ApplicantIncome</th>\n",
       "      <th>CoapplicantIncome</th>\n",
       "      <th>LoanAmount</th>\n",
       "      <th>Loan_Amount_Term</th>\n",
       "      <th>Credit_History</th>\n",
       "      <th>Property_Area</th>\n",
       "      <th>Loan_Status</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Loan_ID</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>LP001002</th>\n",
       "      <td>Male</td>\n",
       "      <td>No</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>5849</td>\n",
       "      <td>0.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "      <td>Y</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>LP001003</th>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>4583</td>\n",
       "      <td>1508.0</td>\n",
       "      <td>128.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Rural</td>\n",
       "      <td>N</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>LP001005</th>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>Yes</td>\n",
       "      <td>3000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>66.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "      <td>Y</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>LP001006</th>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Not Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>2583</td>\n",
       "      <td>2358.0</td>\n",
       "      <td>120.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "      <td>Y</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>LP001008</th>\n",
       "      <td>Male</td>\n",
       "      <td>No</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>6000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>141.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "      <td>Y</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         Gender Married  Dependents     Education Self_Employed  \\\n",
       "Loan_ID                                                           \n",
       "LP001002   Male      No         0.0      Graduate            No   \n",
       "LP001003   Male     Yes         1.0      Graduate            No   \n",
       "LP001005   Male     Yes         0.0      Graduate           Yes   \n",
       "LP001006   Male     Yes         0.0  Not Graduate            No   \n",
       "LP001008   Male      No         0.0      Graduate            No   \n",
       "\n",
       "          ApplicantIncome  CoapplicantIncome  LoanAmount  Loan_Amount_Term  \\\n",
       "Loan_ID                                                                      \n",
       "LP001002             5849                0.0         NaN             360.0   \n",
       "LP001003             4583             1508.0       128.0             360.0   \n",
       "LP001005             3000                0.0        66.0             360.0   \n",
       "LP001006             2583             2358.0       120.0             360.0   \n",
       "LP001008             6000                0.0       141.0             360.0   \n",
       "\n",
       "          Credit_History Property_Area Loan_Status  \n",
       "Loan_ID                                             \n",
       "LP001002             1.0         Urban           Y  \n",
       "LP001003             1.0         Rural           N  \n",
       "LP001005             1.0         Urban           Y  \n",
       "LP001006             1.0         Urban           Y  \n",
       "LP001008             1.0         Urban           Y  "
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Preview the train & test data\n",
    "\n",
    "train_data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "4e7aaaa0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Gender</th>\n",
       "      <th>Married</th>\n",
       "      <th>Dependents</th>\n",
       "      <th>Education</th>\n",
       "      <th>Self_Employed</th>\n",
       "      <th>ApplicantIncome</th>\n",
       "      <th>CoapplicantIncome</th>\n",
       "      <th>LoanAmount</th>\n",
       "      <th>Loan_Amount_Term</th>\n",
       "      <th>Credit_History</th>\n",
       "      <th>Property_Area</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Loan_ID</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>LP001015</th>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>5720</td>\n",
       "      <td>0</td>\n",
       "      <td>110.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>LP001022</th>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>3076</td>\n",
       "      <td>1500</td>\n",
       "      <td>126.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>LP001031</th>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>2.0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>5000</td>\n",
       "      <td>1800</td>\n",
       "      <td>208.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>LP001035</th>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>2.0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>2340</td>\n",
       "      <td>2546</td>\n",
       "      <td>100.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Urban</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>LP001051</th>\n",
       "      <td>Male</td>\n",
       "      <td>No</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Not Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>3276</td>\n",
       "      <td>0</td>\n",
       "      <td>78.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         Gender Married  Dependents     Education Self_Employed  \\\n",
       "Loan_ID                                                           \n",
       "LP001015   Male     Yes         0.0      Graduate            No   \n",
       "LP001022   Male     Yes         1.0      Graduate            No   \n",
       "LP001031   Male     Yes         2.0      Graduate            No   \n",
       "LP001035   Male     Yes         2.0      Graduate            No   \n",
       "LP001051   Male      No         0.0  Not Graduate            No   \n",
       "\n",
       "          ApplicantIncome  CoapplicantIncome  LoanAmount  Loan_Amount_Term  \\\n",
       "Loan_ID                                                                      \n",
       "LP001015             5720                  0       110.0             360.0   \n",
       "LP001022             3076               1500       126.0             360.0   \n",
       "LP001031             5000               1800       208.0             360.0   \n",
       "LP001035             2340               2546       100.0             360.0   \n",
       "LP001051             3276                  0        78.0             360.0   \n",
       "\n",
       "          Credit_History Property_Area  \n",
       "Loan_ID                                 \n",
       "LP001015             1.0         Urban  \n",
       "LP001022             1.0         Urban  \n",
       "LP001031             1.0         Urban  \n",
       "LP001035             NaN         Urban  \n",
       "LP001051             1.0         Urban  "
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_data.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "82c774c7",
   "metadata": {},
   "source": [
    "#### **<font color = green>3.2 Null value & special characters treatment</font>**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "id": "73000468",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Gender               13\n",
       "Married               3\n",
       "Dependents           15\n",
       "Education             0\n",
       "Self_Employed        32\n",
       "ApplicantIncome       0\n",
       "CoapplicantIncome     0\n",
       "LoanAmount           22\n",
       "Loan_Amount_Term     14\n",
       "Credit_History       50\n",
       "Property_Area         0\n",
       "Loan_Status           0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 52,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Check for null values\n",
    "train_data.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 53,
   "id": "4104fcfd",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Gender               11\n",
       "Married               0\n",
       "Dependents           10\n",
       "Education             0\n",
       "Self_Employed        23\n",
       "ApplicantIncome       0\n",
       "CoapplicantIncome     0\n",
       "LoanAmount            5\n",
       "Loan_Amount_Term      6\n",
       "Credit_History       29\n",
       "Property_Area         0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 53,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_data.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 54,
   "id": "838b2de4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Gender                0\n",
      "Married               0\n",
      "Dependents            0\n",
      "Education             0\n",
      "Self_Employed         0\n",
      "ApplicantIncome       0\n",
      "CoapplicantIncome     0\n",
      "LoanAmount           22\n",
      "Loan_Amount_Term      0\n",
      "Credit_History       50\n",
      "Property_Area         0\n",
      "Loan_Status           0\n",
      "dtype: int64\n",
      "Gender                0\n",
      "Married               0\n",
      "Dependents            0\n",
      "Education             0\n",
      "Self_Employed         0\n",
      "ApplicantIncome       0\n",
      "CoapplicantIncome     0\n",
      "LoanAmount            5\n",
      "Loan_Amount_Term      0\n",
      "Credit_History       29\n",
      "Property_Area         0\n",
      "dtype: int64\n"
     ]
    }
   ],
   "source": [
    "# Treating the null values\n",
    "# Inputing Categorical missing data with \"mode\" value\n",
    "\n",
    "colname1 = [\"Gender\",\"Married\",\"Dependents\",\"Self_Employed\",\"Loan_Amount_Term\"]\n",
    "\n",
    "for x in colname1:\n",
    "    train_data[x].fillna(train_data[x].mode()[0],inplace=True)\n",
    "    test_data[x].fillna(test_data[x].mode()[0],inplace=True)\n",
    "    \n",
    "print(train_data.isnull().sum())\n",
    "print(test_data.isnull().sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 55,
   "id": "5286092a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Gender                0\n",
      "Married               0\n",
      "Dependents            0\n",
      "Education             0\n",
      "Self_Employed         0\n",
      "ApplicantIncome       0\n",
      "CoapplicantIncome     0\n",
      "LoanAmount            0\n",
      "Loan_Amount_Term      0\n",
      "Credit_History       50\n",
      "Property_Area         0\n",
      "Loan_Status           0\n",
      "dtype: int64\n",
      "Gender                0\n",
      "Married               0\n",
      "Dependents            0\n",
      "Education             0\n",
      "Self_Employed         0\n",
      "ApplicantIncome       0\n",
      "CoapplicantIncome     0\n",
      "LoanAmount            0\n",
      "Loan_Amount_Term      0\n",
      "Credit_History       29\n",
      "Property_Area         0\n",
      "dtype: int64\n"
     ]
    }
   ],
   "source": [
    "# Inputing the numeric values with \"mean\" value\n",
    "\n",
    "train_data[\"LoanAmount\"].fillna(round(train_data[\"LoanAmount\"].mean(),0),inplace = True)\n",
    "test_data[\"LoanAmount\"].fillna(round(test_data[\"LoanAmount\"].mean(),0),inplace = True)\n",
    "\n",
    "print(train_data.isnull().sum())\n",
    "print(test_data.isnull().sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 56,
   "id": "a0c0084f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Gender               0\n",
      "Married              0\n",
      "Dependents           0\n",
      "Education            0\n",
      "Self_Employed        0\n",
      "ApplicantIncome      0\n",
      "CoapplicantIncome    0\n",
      "LoanAmount           0\n",
      "Loan_Amount_Term     0\n",
      "Credit_History       0\n",
      "Property_Area        0\n",
      "Loan_Status          0\n",
      "dtype: int64\n",
      "Gender               0\n",
      "Married              0\n",
      "Dependents           0\n",
      "Education            0\n",
      "Self_Employed        0\n",
      "ApplicantIncome      0\n",
      "CoapplicantIncome    0\n",
      "LoanAmount           0\n",
      "Loan_Amount_Term     0\n",
      "Credit_History       0\n",
      "Property_Area        0\n",
      "dtype: int64\n"
     ]
    }
   ],
   "source": [
    "# So from above we got to know that we are still yet to replace the null values from credit history\n",
    "\n",
    "train_data[\"Credit_History\"].fillna(value=0,inplace = True)\n",
    "test_data[\"Credit_History\"].fillna(value=0,inplace = True)\n",
    "\n",
    "print(train_data.isnull().sum())\n",
    "print(test_data.isnull().sum())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ab10b96b",
   "metadata": {},
   "source": [
    "#### **<font color = green>3.3 Label Encoding</font>**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 57,
   "id": "2503a1d7",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Converting all the categorical data into numerical for train_data\n",
    "\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "\n",
    "colname = [\"Gender\",\"Married\",\"Education\",\"Self_Employed\",\"Dependents\",\"Self_Employed\",\"Property_Area\",\"Loan_Status\"]\n",
    "\n",
    "le = LabelEncoder()\n",
    "\n",
    "for x in colname:\n",
    "    train_data[x]=le.fit_transform(train_data[x])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "id": "8858dee9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Gender</th>\n",
       "      <th>Married</th>\n",
       "      <th>Dependents</th>\n",
       "      <th>Education</th>\n",
       "      <th>Self_Employed</th>\n",
       "      <th>ApplicantIncome</th>\n",
       "      <th>CoapplicantIncome</th>\n",
       "      <th>LoanAmount</th>\n",
       "      <th>Loan_Amount_Term</th>\n",
       "      <th>Credit_History</th>\n",
       "      <th>Property_Area</th>\n",
       "      <th>Loan_Status</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Loan_ID</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>LP001002</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>5849</td>\n",
       "      <td>0.0</td>\n",
       "      <td>146.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>LP001003</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>4583</td>\n",
       "      <td>1508.0</td>\n",
       "      <td>128.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>LP001005</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>3000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>66.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>LP001006</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>2583</td>\n",
       "      <td>2358.0</td>\n",
       "      <td>120.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>LP001008</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>6000</td>\n",
       "      <td>0.0</td>\n",
       "      <td>141.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          Gender  Married  Dependents  Education  Self_Employed  \\\n",
       "Loan_ID                                                           \n",
       "LP001002       1        0           0          0              0   \n",
       "LP001003       1        1           1          0              0   \n",
       "LP001005       1        1           0          0              1   \n",
       "LP001006       1        1           0          1              0   \n",
       "LP001008       1        0           0          0              0   \n",
       "\n",
       "          ApplicantIncome  CoapplicantIncome  LoanAmount  Loan_Amount_Term  \\\n",
       "Loan_ID                                                                      \n",
       "LP001002             5849                0.0       146.0             360.0   \n",
       "LP001003             4583             1508.0       128.0             360.0   \n",
       "LP001005             3000                0.0        66.0             360.0   \n",
       "LP001006             2583             2358.0       120.0             360.0   \n",
       "LP001008             6000                0.0       141.0             360.0   \n",
       "\n",
       "          Credit_History  Property_Area  Loan_Status  \n",
       "Loan_ID                                               \n",
       "LP001002             1.0              2            1  \n",
       "LP001003             1.0              0            0  \n",
       "LP001005             1.0              2            1  \n",
       "LP001006             1.0              2            1  \n",
       "LP001008             1.0              2            1  "
      ]
     },
     "execution_count": 58,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_data.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 59,
   "id": "af9073f1",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Converting all the categorical data into numerical for test_data\n",
    "\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "\n",
    "colname = [\"Gender\",\"Married\",\"Education\",\"Self_Employed\",\"Property_Area\"]\n",
    "\n",
    "le = LabelEncoder()\n",
    "\n",
    "for x in colname:\n",
    "    test_data[x]=le.fit_transform(test_data[x])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 60,
   "id": "3dd8506b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Gender</th>\n",
       "      <th>Married</th>\n",
       "      <th>Dependents</th>\n",
       "      <th>Education</th>\n",
       "      <th>Self_Employed</th>\n",
       "      <th>ApplicantIncome</th>\n",
       "      <th>CoapplicantIncome</th>\n",
       "      <th>LoanAmount</th>\n",
       "      <th>Loan_Amount_Term</th>\n",
       "      <th>Credit_History</th>\n",
       "      <th>Property_Area</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>Loan_ID</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>LP001015</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>5720</td>\n",
       "      <td>0</td>\n",
       "      <td>110.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>LP001022</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>3076</td>\n",
       "      <td>1500</td>\n",
       "      <td>126.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>LP001031</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>5000</td>\n",
       "      <td>1800</td>\n",
       "      <td>208.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>LP001035</th>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>2.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2340</td>\n",
       "      <td>2546</td>\n",
       "      <td>100.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>LP001051</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>3276</td>\n",
       "      <td>0</td>\n",
       "      <td>78.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "          Gender  Married  Dependents  Education  Self_Employed  \\\n",
       "Loan_ID                                                           \n",
       "LP001015       1        1         0.0          0              0   \n",
       "LP001022       1        1         1.0          0              0   \n",
       "LP001031       1        1         2.0          0              0   \n",
       "LP001035       1        1         2.0          0              0   \n",
       "LP001051       1        0         0.0          1              0   \n",
       "\n",
       "          ApplicantIncome  CoapplicantIncome  LoanAmount  Loan_Amount_Term  \\\n",
       "Loan_ID                                                                      \n",
       "LP001015             5720                  0       110.0             360.0   \n",
       "LP001022             3076               1500       126.0             360.0   \n",
       "LP001031             5000               1800       208.0             360.0   \n",
       "LP001035             2340               2546       100.0             360.0   \n",
       "LP001051             3276                  0        78.0             360.0   \n",
       "\n",
       "          Credit_History  Property_Area  \n",
       "Loan_ID                                  \n",
       "LP001015             1.0              2  \n",
       "LP001022             1.0              2  \n",
       "LP001031             1.0              2  \n",
       "LP001035             0.0              2  \n",
       "LP001051             1.0              2  "
      ]
     },
     "execution_count": 60,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_data.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "74a3ab19",
   "metadata": {},
   "source": [
    "#### **<font color = green>3.4 Checking Correlation</font>**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "5ceaf1a9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<Axes: >"
      ]
     },
     "execution_count": 61,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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Pn26tmzZok2bNqljx466++671axZM02fPl3bt29v6DgBAAAAAAAAAEAFdUryl9ezZ0/df//9uvvuu5WTk6N33nlHffr0Uf/+/bV79+6GiBEAAAAAAAAAGpzFbOHRiB5nqzon+YuKivTpp59q+PDhatWqldauXatXX31VCQkJOnz4sMLDw3X99dc3ZKwAAAAAAAAAAKAcY10WmjZtmlauXClJuvXWW/XMM8+oW7du1te9vLz01FNPqXXr1g0SJAAAAAAAAAAAqKxOSf49e/bolVde0ejRo+Xq6lplnebNm+vXX3+tV3AAAAAAAAAAAKB6dUry//zzzydesdGogQMH1mX1AAAAAAAAAADgJNQpyS9JUVFRWrJkifbu3SuDwaDOnTtrxowZioiIaMj4AAAAAAAAAABANep04921a9eqS5cu2rhxo3r06KFu3bppw4YN6tq1q3788ceGjhEAAAAAAAAAAFShTjP577//fs2aNUtPPfVUpfL77rtPgwcPbpDgAAAAAAAAAABA9eo0k3/v3r2aNGlSpfKJEydqz5499Q4KAAAAAAAAAACcWJ1m8gcFBWnbtm1q3769Tfm2bdsUHBzcIIEBAAAAAAAAwClltncAQP3VKck/ZcoUTZ06VYcOHdKFF14og8GgP/74Q08//bTmzJnT0DECAAAAAAAAAIAq1CnJ//DDD6tp06Z6/vnnNX/+fElS8+bN9eijj2r69OkNGiAAAAAAAAAAAKhanZL8BoNBs2bN0qxZs5SVlSVJatq0aYMGBgAAAAAAAAAAalanJH95JPcBAAAAAAAAALCPk07y9+rVSwaD4aTqbtmypc4BAQAAAAAAAACAk3PSSf6RI0eewjAAAAAAAAAAAEBtnXSSf8GCBacyDgAAAAAAAAAAUEv1uib/pk2btHfvXhkMBnXu3Fl9+vRpqLgAAAAAAAAAAMAJ1CnJf+zYMd100036888/5evrK0lKT0/XhRdeqJUrVyo8PLwhYwQAAAAAAACABmcx2zsCoP6c6rLQxIkTVVRUpL179yo1NVWpqanau3evLBaLJk2a1NAxAgAAAAAAAACAKtRpJv/vv/+uv/76Sx07drSWdezYUa+88oouuuiiBgsOAAAAAAAAAABUr04z+Vu2bKmioqJK5SaTSWFhYfUOCgAAAAAAAAAAnFidZvI/88wzmjZtml577TX16dNH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ISgoCOnp6ahbty50dXUVjru7u79xOCIiIiIiIiIiIiIierbXLvL36NEDABAQECAfk0gkkMlk3HiXiIiIiIiIiIiIiKgcvHaRPyUl5W3mICIiIiIiIiIiIiKiV/TaRX57e/u3mYOIiIiIiIiIiIiIiF7Raxf5n7hw4QJSU1Px+PFjhXFvb+83/auJiIiIiIiIiIiIiOg5XrvIf+XKFXTv3h1nz56V9+IHSvryA2BPfiIiIiIiIiIiIlJp0n9rmkTqTOt1f3DMmDFwdHTE7du3UbFiRZw/fx7//PMPGjdujL///vstRiQiIiIiIiIiIiIiorK89kz+o0ePIjIyEpaWltDS0oKWlhY+/vhjfPvttxg9ejTOnDnzNnMSEREREREREREREdFTXnsmf3FxMQwNDQEAFhYWuHXrFoCSDXkTExPfTjoiIiIiIiIiIiIiInqm157JX6dOHcTHx8PJyQlNmzbF3LlzoaenhxUrVsDJyeltZiQiIiIiIiIiIiIiojK8dpF/8uTJyM3NBQDMnDkTXbp0QfPmzVG5cmX89ttvby0gERERERERERERERGV7bWL/B06dJD/2cnJCRcuXEBmZibMzMwgkUjeSjgiIiIiIiIiIiIiInq21y7yl3bjxg1IJBLY2tq+jb+OiIiIiIiIiIiIiIhewmtvvCuVSjFjxgyYmJjA3t4e1apVg6mpKb755htIpdK3mZGIiIiIiIiIiIiIiMrw2jP5J02ahPDwcMyZMweenp6QyWSIjo7G9OnTUVBQgFmzZr3NnERERERERERERERvlQwy0RGI3thrF/nXrFmDsLAweHt7y8fq1asHW1tbjBgxgkV+IiIiIiIiIiIiIqJ37LXb9WRmZqJmzZpK4zVr1kRmZuYbhSIiIiIiIiIiIiIiohd77SJ/vXr1sHTpUqXxpUuXwt3d/Y1CERERERERERERERHRi712u565c+eic+fO2L9/Pzw8PCCRSHDkyBFcv34du3btepsZiYiIiIiIiIiIiIioDK89k79ly5a4dOkSunfvjqysLGRmZsLX1xfnz5/HqlWr3mZGIiIiIiIiIiIiIiIqw2vP5AcAGxsbpQ124+LisGbNGkRERLxRMCIiIiIiIiIiIiIier7XnslPRERERERERERERERischPRERERERERERERKSmWOQnIiIiIiIiIiIiIlJTr9yT39fX97nHs7KyXjcLERERERERERERUbmRymSiIxC9sVcu8puYmLzw+MCBA187EBERERERERERERERvZxXLvKvWrXqXeQgIiIiIiIiIiIiIqJXxJ78RERERERERERERERqikV+IiIiIiIiIiIiIiI1xSI/EREREREREREREZGaYpGfiIiIiIiIiIiIiEhNschPRERERERERERERKSmWOQnIiIiIiIiIiIiIlJTOqIDEBEREREREREREYkgk8lERyB6Y5zJT0RERERERERERESkpljkJyIiIiIiIiIiIiJSUyzyExERERERERERERGpKRb5iYiIiIiIiIiIiIjUFIv8RERERERERERERERqikV+IiIiIiIiIiIiIiI1xSI/EREREREREREREZGaYpGfiIiIiIiIiIiIiEhN6YgOQERERERERERERCSCDDLREYjeGGfyExERERERERERERGpKRb5iYiIiIiIiIiIiIjUFIv8RERERERERERERERqikV+IiIiIiIiIiIiIiI1xSI/EREREREREREREZGaYpGfiIiIiIiIiIiIiEhNschPRERERERERERERKSmWOQnIiIiIiIiIiIiIlJTOqIDEBEREREREREREYkgk8lERyB6Y5zJT0RERERERERERESkpljkJyIiIiIiIiIiIiJSUyzyExERERERERERERGpKRb5iYiIiIiIiIiIiIjUFIv8RERERERERERERERqikV+IiIiIiIiIiIiIiI1xSI/EREREREREREREZGaYpGfiIiIiIiIiIiIiEhN6YgOQERERERERERERCSCTCYTHYHojXEmPxERERERERERERGRmmKRn4iIiIiIiIiIiIhITbHIT0RERERERERERESkpljkJyIiIiIiIiIiIiJSUyzyExERERERERERERGpKRb5iYiIiIiIiIiIiIjUFIv8RERERERERERERERqikV+IiIiIiIiIiIiIiI1pSM6ABEREREREREREZEIMtEBiN4CzuQnIiIiIiIiIiIiIlJTLPITEREREREREREREakpFvmJiIiIiIiIiIiIiNQUi/xERERERERERERERGqKRX4iIiIiIiIiIiIiomeYNWsWmjVrhooVK8LU1PSlfkYmk2H69OmwsbFBhQoV0KpVK5w/f17hnEePHmHUqFGwsLBApUqV4O3tjRs3brxyPhb5iYiIiIiI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",
      "text/plain": [
       "<Figure size 2000x3000 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "plt.figure(figsize=(20,30))\n",
    "sns.heatmap(train_data.corr(),vmin=-1.0,vmax=1.0,annot=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "457dc9b3",
   "metadata": {},
   "source": [
    "### **<font color = blue>4. Splitting Training & Testing data</font>**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 64,
   "id": "f2f67f92",
   "metadata": {},
   "outputs": [],
   "source": [
    "\"\"\"One thing we have to remember here is that the train & test data are already split and given to us.\n",
    "    So here instead of splitting in X & Y and then applying train test split, we are directly defining our \n",
    "    X_train & Y_train.\n",
    "    In the code below astype(int) will retuen the integer value of the last column of train data\n",
    "    (since Y_train = train_data.values[:,-1] )\"\"\"\n",
    "\n",
    "X_train = train_data.values[:,0:-1]\n",
    "Y_train = train_data.values[:,-1]\n",
    "Y_train = Y_train.astype(int)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "3b41af5e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(614, 11)"
      ]
     },
     "execution_count": 65,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 66,
   "id": "e593069b",
   "metadata": {},
   "outputs": [],
   "source": [
    "\"\"\"We are NOT splitting the test data here since we do not have any dependent column y present.\n",
    "    Here we are keeping the test data as it is\"\"\"\n",
    "\n",
    "X_test = test_data.values[:,:]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "6df743dc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(367, 11)"
      ]
     },
     "execution_count": 67,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_test.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b7a0dfc3",
   "metadata": {},
   "source": [
    "### **<font color = blue>5. Scaling the data</font>**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "e3f90a96",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import StandardScaler\n",
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train = scaler.transform(X_train)\n",
    "X_test = scaler.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "id": "05744265",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 0.47234264 -1.37208932 -0.73780632 ...  0.2732313   0.54095432\n",
      "   1.22329839]\n",
      " [ 0.47234264  0.72881553  0.25346957 ...  0.2732313   0.54095432\n",
      "  -1.31851281]\n",
      " [ 0.47234264  0.72881553 -0.73780632 ...  0.2732313   0.54095432\n",
      "   1.22329839]\n",
      " ...\n",
      " [ 0.47234264  0.72881553  0.25346957 ...  0.2732313   0.54095432\n",
      "   1.22329839]\n",
      " [ 0.47234264  0.72881553  1.24474546 ...  0.2732313   0.54095432\n",
      "   1.22329839]\n",
      " [-2.11710719 -1.37208932 -0.73780632 ...  0.2732313  -1.84858491\n",
      "  -0.04760721]]\n"
     ]
    }
   ],
   "source": [
    "print(X_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "6890c7c3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 0.47234264  0.72881553 -0.73780632 ...  0.2732313   0.54095432\n",
      "   1.22329839]\n",
      " [ 0.47234264  0.72881553  0.25346957 ...  0.2732313   0.54095432\n",
      "   1.22329839]\n",
      " [ 0.47234264  0.72881553  1.24474546 ...  0.2732313   0.54095432\n",
      "   1.22329839]\n",
      " ...\n",
      " [ 0.47234264 -1.37208932 -0.73780632 ...  0.2732313  -1.84858491\n",
      "  -0.04760721]\n",
      " [ 0.47234264  0.72881553 -0.73780632 ...  0.2732313   0.54095432\n",
      "  -1.31851281]\n",
      " [ 0.47234264 -1.37208932 -0.73780632 ... -2.52283563  0.54095432\n",
      "  -1.31851281]]\n"
     ]
    }
   ],
   "source": [
    "print(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "99139d53",
   "metadata": {},
   "source": [
    "### **<font color = blue>6. SVM fitting the model</font>**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "0f2bfa67",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-1 {color: black;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>SVC(C=20, gamma=0.01)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" checked><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">SVC</label><div class=\"sk-toggleable__content\"><pre>SVC(C=20, gamma=0.01)</pre></div></div></div></div></div>"
      ],
      "text/plain": [
       "SVC(C=20, gamma=0.01)"
      ]
     },
     "execution_count": 76,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\"\"\"In SVM we have 3 hyperparameters: C (Cost penalty), Kernel & gamma.\n",
    "    The values .... are base values of the model & if one is changing these base values, then one is performing\n",
    "    Hyperparameter tuning. Hyperparameter tuning improves the accuracy of the model.\"\"\"\n",
    "\n",
    "from sklearn.svm import SVC\n",
    "svc_model = SVC(kernel ='rbf',C=20, gamma =0.01)\n",
    "svc_model.fit(X_train,Y_train)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "df8379ec",
   "metadata": {},
   "source": [
    "### **<font color = blue>7. Predicting Y based on X_test</font>**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "id": "481d31af",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1]\n"
     ]
    }
   ],
   "source": [
    "Y_pred = svc_model.predict(X_test)\n",
    "print(list(Y_pred))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d005104a",
   "metadata": {},
   "source": [
    "### **<font color = blue>8. Evaluation of the model</font>**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "id": "5be3f9b1",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.7768729641693811"
      ]
     },
     "execution_count": 79,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "svc_model.score(X_train,Y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "id": "080ea43a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.70      0.51      0.59       192\n",
      "           1       0.80      0.90      0.85       422\n",
      "\n",
      "    accuracy                           0.78       614\n",
      "   macro avg       0.75      0.70      0.72       614\n",
      "weighted avg       0.77      0.78      0.77       614\n",
      "\n"
     ]
    }
   ],
   "source": [
    "from sklearn.metrics import confusion_matrix, classification_report\n",
    "\n",
    "Y_pred_new=svc_model.predict(X_train)\n",
    "confusion_matrix(Y_train, Y_pred_new)\n",
    "\n",
    "print (classification_report(Y_train, Y_pred_new) )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f95acddc",
   "metadata": {},
   "outputs": [],
   "source": [
    "\"\"\"\"Here we are not getting any accuracy score, because our test data do not consist of any Y-values\"\"\""
   ]
  },
  {
   "cell_type": "markdown",
   "id": "75f0cd10",
   "metadata": {},
   "source": [
    "### **<font color = blue>9. Adding Y-predictions column to the test data to display Eligibilty/Non-Eligbility</font>**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "id": "44ebd86f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Loan_ID</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Married</th>\n",
       "      <th>Dependents</th>\n",
       "      <th>Education</th>\n",
       "      <th>Self_Employed</th>\n",
       "      <th>ApplicantIncome</th>\n",
       "      <th>CoapplicantIncome</th>\n",
       "      <th>LoanAmount</th>\n",
       "      <th>Loan_Amount_Term</th>\n",
       "      <th>Credit_History</th>\n",
       "      <th>Property_Area</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>LP001015</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>5720</td>\n",
       "      <td>0</td>\n",
       "      <td>110.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>LP001022</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>3076</td>\n",
       "      <td>1500</td>\n",
       "      <td>126.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>LP001031</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>2.0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>5000</td>\n",
       "      <td>1800</td>\n",
       "      <td>208.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>LP001035</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>2.0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>2340</td>\n",
       "      <td>2546</td>\n",
       "      <td>100.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Urban</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>LP001051</td>\n",
       "      <td>Male</td>\n",
       "      <td>No</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Not Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>3276</td>\n",
       "      <td>0</td>\n",
       "      <td>78.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>362</th>\n",
       "      <td>LP002971</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>3.0</td>\n",
       "      <td>Not Graduate</td>\n",
       "      <td>Yes</td>\n",
       "      <td>4009</td>\n",
       "      <td>1777</td>\n",
       "      <td>113.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>363</th>\n",
       "      <td>LP002975</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>4158</td>\n",
       "      <td>709</td>\n",
       "      <td>115.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>364</th>\n",
       "      <td>LP002980</td>\n",
       "      <td>Male</td>\n",
       "      <td>No</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>3250</td>\n",
       "      <td>1993</td>\n",
       "      <td>126.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Semiurban</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>365</th>\n",
       "      <td>LP002986</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>5000</td>\n",
       "      <td>2393</td>\n",
       "      <td>158.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Rural</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>366</th>\n",
       "      <td>LP002989</td>\n",
       "      <td>Male</td>\n",
       "      <td>No</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>Yes</td>\n",
       "      <td>9200</td>\n",
       "      <td>0</td>\n",
       "      <td>98.0</td>\n",
       "      <td>180.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Rural</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>367 rows × 12 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      Loan_ID Gender Married  Dependents     Education Self_Employed  \\\n",
       "0    LP001015   Male     Yes         0.0      Graduate            No   \n",
       "1    LP001022   Male     Yes         1.0      Graduate            No   \n",
       "2    LP001031   Male     Yes         2.0      Graduate            No   \n",
       "3    LP001035   Male     Yes         2.0      Graduate            No   \n",
       "4    LP001051   Male      No         0.0  Not Graduate            No   \n",
       "..        ...    ...     ...         ...           ...           ...   \n",
       "362  LP002971   Male     Yes         3.0  Not Graduate           Yes   \n",
       "363  LP002975   Male     Yes         0.0      Graduate            No   \n",
       "364  LP002980   Male      No         0.0      Graduate            No   \n",
       "365  LP002986   Male     Yes         0.0      Graduate            No   \n",
       "366  LP002989   Male      No         0.0      Graduate           Yes   \n",
       "\n",
       "     ApplicantIncome  CoapplicantIncome  LoanAmount  Loan_Amount_Term  \\\n",
       "0               5720                  0       110.0             360.0   \n",
       "1               3076               1500       126.0             360.0   \n",
       "2               5000               1800       208.0             360.0   \n",
       "3               2340               2546       100.0             360.0   \n",
       "4               3276                  0        78.0             360.0   \n",
       "..               ...                ...         ...               ...   \n",
       "362             4009               1777       113.0             360.0   \n",
       "363             4158                709       115.0             360.0   \n",
       "364             3250               1993       126.0             360.0   \n",
       "365             5000               2393       158.0             360.0   \n",
       "366             9200                  0        98.0             180.0   \n",
       "\n",
       "     Credit_History Property_Area  \n",
       "0               1.0         Urban  \n",
       "1               1.0         Urban  \n",
       "2               1.0         Urban  \n",
       "3               NaN         Urban  \n",
       "4               1.0         Urban  \n",
       "..              ...           ...  \n",
       "362             1.0         Urban  \n",
       "363             1.0         Urban  \n",
       "364             NaN     Semiurban  \n",
       "365             1.0         Rural  \n",
       "366             1.0         Rural  \n",
       "\n",
       "[367 rows x 12 columns]"
      ]
     },
     "execution_count": 82,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_data=pd.read_csv(r'risk_analytics_test.csv', header=0)\n",
    "test_data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "id": "f04553ff",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Loan_ID</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Married</th>\n",
       "      <th>Dependents</th>\n",
       "      <th>Education</th>\n",
       "      <th>Self_Employed</th>\n",
       "      <th>ApplicantIncome</th>\n",
       "      <th>CoapplicantIncome</th>\n",
       "      <th>LoanAmount</th>\n",
       "      <th>Loan_Amount_Term</th>\n",
       "      <th>Credit_History</th>\n",
       "      <th>Property_Area</th>\n",
       "      <th>Y_predictions</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>LP001015</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>5720</td>\n",
       "      <td>0</td>\n",
       "      <td>110.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>LP001022</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>3076</td>\n",
       "      <td>1500</td>\n",
       "      <td>126.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>LP001031</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>2.0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>5000</td>\n",
       "      <td>1800</td>\n",
       "      <td>208.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>LP001035</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>2.0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>2340</td>\n",
       "      <td>2546</td>\n",
       "      <td>100.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Urban</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>LP001051</td>\n",
       "      <td>Male</td>\n",
       "      <td>No</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Not Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>3276</td>\n",
       "      <td>0</td>\n",
       "      <td>78.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    Loan_ID Gender Married  Dependents     Education Self_Employed  \\\n",
       "0  LP001015   Male     Yes         0.0      Graduate            No   \n",
       "1  LP001022   Male     Yes         1.0      Graduate            No   \n",
       "2  LP001031   Male     Yes         2.0      Graduate            No   \n",
       "3  LP001035   Male     Yes         2.0      Graduate            No   \n",
       "4  LP001051   Male      No         0.0  Not Graduate            No   \n",
       "\n",
       "   ApplicantIncome  CoapplicantIncome  LoanAmount  Loan_Amount_Term  \\\n",
       "0             5720                  0       110.0             360.0   \n",
       "1             3076               1500       126.0             360.0   \n",
       "2             5000               1800       208.0             360.0   \n",
       "3             2340               2546       100.0             360.0   \n",
       "4             3276                  0        78.0             360.0   \n",
       "\n",
       "   Credit_History Property_Area  Y_predictions  \n",
       "0             1.0         Urban              1  \n",
       "1             1.0         Urban              1  \n",
       "2             1.0         Urban              1  \n",
       "3             NaN         Urban              0  \n",
       "4             1.0         Urban              1  "
      ]
     },
     "execution_count": 83,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_data[\"Y_predictions\"]=Y_pred\n",
    "test_data.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b94a1215",
   "metadata": {},
   "source": [
    "### **<font color = blue>10. Mapping Eligibilty: \"1\"/Non-Eligbility: \"0\" in Y_predictions col</font>**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "id": "f1f462f9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Loan_ID</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Married</th>\n",
       "      <th>Dependents</th>\n",
       "      <th>Education</th>\n",
       "      <th>Self_Employed</th>\n",
       "      <th>ApplicantIncome</th>\n",
       "      <th>CoapplicantIncome</th>\n",
       "      <th>LoanAmount</th>\n",
       "      <th>Loan_Amount_Term</th>\n",
       "      <th>Credit_History</th>\n",
       "      <th>Property_Area</th>\n",
       "      <th>Y_predictions</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>LP001015</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>5720</td>\n",
       "      <td>0</td>\n",
       "      <td>110.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "      <td>Eligible</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>LP001022</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>3076</td>\n",
       "      <td>1500</td>\n",
       "      <td>126.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "      <td>Eligible</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>LP001031</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>2.0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>5000</td>\n",
       "      <td>1800</td>\n",
       "      <td>208.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "      <td>Eligible</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>LP001035</td>\n",
       "      <td>Male</td>\n",
       "      <td>Yes</td>\n",
       "      <td>2.0</td>\n",
       "      <td>Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>2340</td>\n",
       "      <td>2546</td>\n",
       "      <td>100.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Urban</td>\n",
       "      <td>Not Eligible</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>LP001051</td>\n",
       "      <td>Male</td>\n",
       "      <td>No</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Not Graduate</td>\n",
       "      <td>No</td>\n",
       "      <td>3276</td>\n",
       "      <td>0</td>\n",
       "      <td>78.0</td>\n",
       "      <td>360.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Urban</td>\n",
       "      <td>Eligible</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "    Loan_ID Gender Married  Dependents     Education Self_Employed  \\\n",
       "0  LP001015   Male     Yes         0.0      Graduate            No   \n",
       "1  LP001022   Male     Yes         1.0      Graduate            No   \n",
       "2  LP001031   Male     Yes         2.0      Graduate            No   \n",
       "3  LP001035   Male     Yes         2.0      Graduate            No   \n",
       "4  LP001051   Male      No         0.0  Not Graduate            No   \n",
       "\n",
       "   ApplicantIncome  CoapplicantIncome  LoanAmount  Loan_Amount_Term  \\\n",
       "0             5720                  0       110.0             360.0   \n",
       "1             3076               1500       126.0             360.0   \n",
       "2             5000               1800       208.0             360.0   \n",
       "3             2340               2546       100.0             360.0   \n",
       "4             3276                  0        78.0             360.0   \n",
       "\n",
       "   Credit_History Property_Area Y_predictions  \n",
       "0             1.0         Urban      Eligible  \n",
       "1             1.0         Urban      Eligible  \n",
       "2             1.0         Urban      Eligible  \n",
       "3             NaN         Urban  Not Eligible  \n",
       "4             1.0         Urban      Eligible  "
      ]
     },
     "execution_count": 84,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_data[\"Y_predictions\"]=test_data[\"Y_predictions\"]. map({1:\"Eligible\", 0:\"Not Eligible\"})\n",
    "test_data.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6c644f08",
   "metadata": {},
   "source": [
    "### **<font color = blue>11. Outsourcing the Loan elibilty test data to the client</font>**"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "id": "8f94f143",
   "metadata": {},
   "outputs": [],
   "source": [
    "test_data.to_csv(r'test_data_output.csv',index=False)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "id": "8d33ccfc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Y_predictions\n",
       "Eligible        280\n",
       "Not Eligible     87\n",
       "Name: count, dtype: int64"
      ]
     },
     "execution_count": 88,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_data.Y_predictions.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2b6ba1ee",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
