# EX-05-Feature-Generation


## AIM
To read the given data and perform Feature Generation process and save the data to a file. 

# Explanation
Feature Generation (also known as feature construction, feature extraction or feature engineering) is the process of transforming features into new features that better relate to the target.
 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature Generation techniques to all the feature of the data set
### STEP 4
Save the data to the file


# CODE
```

import pandas as pd
df=pd.read_csv("data.csv")
df

#feature generation
import category_encoders as ce
be=ce.BinaryEncoder()
ndf=be.fit_transform(df["bin_1"])
df["bin_1"] = be.fit_transform(df["bin_1"])
ndf

ndf2=be.fit_transform(df["bin_2"])
df["bin_2"] = be.fit_transform(df["bin_2"])
ndf2

df1=df.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
import category_encoders as ce
be=ce.BinaryEncoder()
ohe=OneHotEncoder(sparse=False)
le=LabelEncoder()
oe=OrdinalEncoder()


df1["City"] = ohe.fit_transform(df1[["City"]])

temp=['Cold','Warm','Hot','Very Hot']
oe1=OrdinalEncoder(categories=[temp])
df1['Ord_1'] = oe1.fit_transform(df1[["Ord_1"]])

edu=['High School','Diploma','Bachelors','Masters','PhD']
oe2=OrdinalEncoder(categories=[edu])
df1['Ord_2']= oe2.fit_transform(df1[["Ord_2"]])
df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df5
```
```
Titanic.csv :

import pandas as pd
df=pd.read_csv("titanic_dataset.csv")
df

# removing unwanted data
df.drop("Name",axis=1,inplace=True)
df.drop("Ticket",axis=1,inplace=True)
df.drop("Cabin",axis=1,inplace=True)

# data cleaning
df.isnull().sum()

df["Age"]=df["Age"].fillna(df["Age"].median())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])

df.isnull().sum()

df

# feature encoding
from category_encoders import BinaryEncoder
be=BinaryEncoder()
df["Sex"]=be.fit_transform(df[["Sex"]])
ndf=be.fit_transform(df["Sex"])
ndf

df1=df.copy()
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
embark=['S','C','Q']
e1=OrdinalEncoder(categories=[embark])
df1['Embarked'] = e1.fit_transform(df[['Embarked']])
df1

# feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df5
```
# OUPUT
## Data.csv:
## Initial Dataset:
![image](https://user-images.githubusercontent.com/119558093/234931783-ab93f4fa-3091-4d32-8288-ec207d658b47.png)
## Binary Encoding:
![image](https://user-images.githubusercontent.com/119558093/234931866-e0f56932-ae42-4154-a452-e9cf499e8d85.png)
![image](https://user-images.githubusercontent.com/119558093/234931905-fbf90c5a-3a95-4424-9778-a4752d3a5c93.png)

## Encoded Dataset:
![image](https://user-images.githubusercontent.com/119558093/234932012-2ab62029-f175-4682-8825-8985bc32eb27.png)

## Data Scaling using MinMaxScaler:
![image](https://user-images.githubusercontent.com/119558093/234932126-3225f770-a482-4c89-b71c-b13a4a7c0ec6.png)

## Data Scaling using StandardScaler:
![image](https://user-images.githubusercontent.com/119558093/234932223-e28a0044-aa4c-4381-8686-79408e32e28b.png)

## Data Scaling using MaxAbsScaler:
![image](https://user-images.githubusercontent.com/119558093/234932345-b651a8d3-6ec9-4672-bb92-7ff470633af5.png)

## Data Scaling using RobustScaler:
![image](https://user-images.githubusercontent.com/119558093/234935871-ef6cad57-a588-4354-b630-a19c538e4f2e.png)


## Encoding.csv :
## Initial Dataset:
![image](https://user-images.githubusercontent.com/119558093/234935914-448303ba-d247-46cb-b715-065e53f3357f.png)


## Binary Encoding:
![image](https://user-images.githubusercontent.com/119558093/234932819-04527224-3c4d-43c6-836c-f49fbf218f7f.png)
![image](https://user-images.githubusercontent.com/119558093/234935980-b84b7808-9b42-4444-892a-ebb570b6152a.png)

## Encoded Dataset:
![image](https://user-images.githubusercontent.com/119558093/234932986-764229a3-9a3b-4151-935a-7a83c2eaa59b.png)

![image](https://user-images.githubusercontent.com/119558093/234933014-378b5e71-c954-4c41-b2b7-e4a968e0b9ca.png)
## Data Scaling using MinMaxScaler:
![image](https://user-images.githubusercontent.com/119558093/234933859-62060a46-c4dc-4960-a078-4c6638b253fb.png)
## Data Scaling using StandardScaler:
![image](https://user-images.githubusercontent.com/119558093/234933943-f45869b4-6536-4b67-a7b6-216e3959a9d8.png)

## Data Scaling using MaxAbsScaler:
![image](https://user-images.githubusercontent.com/119558093/234934062-0b9d7e8c-0dfb-44c4-a70f-b5e3fa88ea35.png)
## Data Scaling using RobustScaler:
![image](https://user-images.githubusercontent.com/119558093/234934314-1534956f-f84b-4f5c-b9df-c1e6f7bafd42.png)

## Titanic.csv : Initial Dataset:
![image](https://user-images.githubusercontent.com/119558093/234934403-edc5a493-7bfb-4865-9aa8-5f3bda3f3643.png)

## Data cleaning before encoding:
![image](https://user-images.githubusercontent.com/119558093/234934575-2f77f738-a39a-4c66-9855-180aed6d2121.png)

![image](https://user-images.githubusercontent.com/119558093/234934519-9d64fb8d-2f3d-4067-8f71-eef89ca0c0f4.png)
![image](https://user-images.githubusercontent.com/119558093/234934626-84b20e70-7919-4f7b-a3c0-6064469dbdc6.png)

## Cleaned Dataset:
![image](https://user-images.githubusercontent.com/119558093/234934748-6d693017-ae0d-4fb7-aeeb-2d28ce4295ad.png)

## Binary Encoding:
![image](https://user-images.githubusercontent.com/119558093/234934827-37707c00-f31a-4789-9a21-3566933c511a.png)

## Encoded Dataset:
![image](https://user-images.githubusercontent.com/119558093/234934928-acd4b884-555c-4b7f-a72c-6a92ec9e2f85.png)

## Data Scaling using MinMaxScaler:
![image](https://user-images.githubusercontent.com/119558093/234935010-d166e639-f9fa-4b12-98bc-d8168825ee20.png)

## Data Scaling using StandardScaler:
![image](https://user-images.githubusercontent.com/119558093/234935096-57529ce7-ae51-4f32-83b6-95713ea04801.png)

## Data Scaling using RobustScaler:
![image](https://user-images.githubusercontent.com/119558093/234935184-0ffc3fb6-2989-4690-bf26-f23a3deca402.png)

# Result:

Feature Generation process and Feature Scaling process is applied to the given data frames sucessfully.
