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
DEVELOPED BY:Priyadharshini P
REGISTER NO:212222100039

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
# Dataset-1 (data.csv)
![image](https://github.com/Priyadharshini-Er/EX-05-Feature-Generation/assets/119558093/069d2036-818e-48f8-89a1-f8f90795096d)
![image](https://github.com/Priyadharshini-Er/EX-05-Feature-Generation/assets/119558093/d3383766-38ef-4b34-a542-0a3d3d127c58)

![image](https://github.com/Priyadharshini-Er/EX-05-Feature-Generation/assets/119558093/a9281790-02f9-48cd-8880-e1423facdbf2)
![image](https://github.com/Priyadharshini-Er/EX-05-Feature-Generation/assets/119558093/874c4243-04c7-4c71-97be-84afca7262ba)
![image](https://github.com/Priyadharshini-Er/EX-05-Feature-Generation/assets/119558093/53faec9e-2482-4f46-b026-65f09a8f170f)
![image](https://github.com/Priyadharshini-Er/EX-05-Feature-Generation/assets/119558093/9b99e5ae-a26f-4753-8fa4-985d27ec11ba)
![image](https://github.com/Priyadharshini-Er/EX-05-Feature-Generation/assets/119558093/15ac13aa-7c9e-4b41-8a5c-c7e5c6e05adb)
# Dataset-2 (Encoding data.csv)
![image](https://github.com/Priyadharshini-Er/EX-05-Feature-Generation/assets/119558093/55ac27b3-2cac-45b2-8689-7c34e34ebe62)

![image](https://github.com/Priyadharshini-Er/EX-05-Feature-Generation/assets/119558093/a98f1686-cb3f-47d5-a188-54f664c83b35)
![image](https://github.com/Priyadharshini-Er/EX-05-Feature-Generation/assets/119558093/1b02c7d3-2ffb-4516-85f3-095503c2f4f9)

![image](https://github.com/Priyadharshini-Er/EX-05-Feature-Generation/assets/119558093/37112ad5-abb4-4138-a057-5874962da7ba)
![image](https://github.com/Priyadharshini-Er/EX-05-Feature-Generation/assets/119558093/5355cbb5-8459-4106-af78-48a9630f0077)
![image](https://github.com/Priyadharshini-Er/EX-05-Feature-Generation/assets/119558093/d6acb6e7-6461-40e5-97e5-0342cc078b39)

![image](https://github.com/Priyadharshini-Er/EX-05-Feature-Generation/assets/119558093/9de10b7f-512b-4f29-a096-ff51a9eedd64)

# Dataset-3 (titanic_dataset.csv)
![image](https://github.com/Priyadharshini-Er/EX-05-Feature-Generation/assets/119558093/12fcb10a-fc89-487f-a6aa-839a3d5b23c9)
![image](https://github.com/Priyadharshini-Er/EX-05-Feature-Generation/assets/119558093/13f7edc9-5351-499e-a5c7-1d627662ffc9)
![image](https://github.com/Priyadharshini-Er/EX-05-Feature-Generation/assets/119558093/2c8cdf0d-af70-41dc-8683-aa5b21339c10)
![image](https://github.com/Priyadharshini-Er/EX-05-Feature-Generation/assets/119558093/cf9a30ac-ec3d-4673-8bb3-bd4335164e8e)
![image](https://github.com/Priyadharshini-Er/EX-05-Feature-Generation/assets/119558093/c5729500-93d3-49fc-9547-135b6cecfde6)
![image](https://github.com/Priyadharshini-Er/EX-05-Feature-Generation/assets/119558093/dc64b6f4-f078-4bce-ab52-c56d045e2b17)
![image](https://github.com/Priyadharshini-Er/EX-05-Feature-Generation/assets/119558093/f8c45216-d6bb-4df2-a28e-ac216c95b4f5)



# Result:

Feature Generation process and Feature Scaling process is applied to the given data frames sucessfully.
