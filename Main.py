import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

#-------------------------Importing Datasets

#_____________Training Data

FileName = 'water_potability.csv'
# FileName = 'Train2.csv'

Train_Data = pd.read_csv(FileName) # Training Data with null
# Train_Data.columns # get columns name

Train = Train_Data.dropna() # Not null Training Data(filter)

# Extra Information
# here Data = Train_Data or Train
# Data.isnull().sum() to get null data info
# Data.info() to get all cloumn name with no of non null info
# Data.shape to get (m*n) data info

#------------Variables of Training

X_Train =  Train[['ph','Hardness','Solids','Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes','Turbidity']]
# X_Train = Train[list(Train_Data)[:-1]]
Y_Train = Train.Potability

#__________Test Data
Test_Data = pd.read_csv('Test2.csv')
Test = Test_Data.dropna() #Test Data


#---------------Variables of Testing

X_Test =  Test[['ph','Hardness','Solids','Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes','Turbidity']]
Y_Test =  Test.Potability

# Modal_1 DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

Model = DecisionTreeClassifier()
Model.fit(X_Train,Y_Train)

Prediction = Model.predict(X_Test)
# print(Prediction)
Accuracy = Model.score(X_Test,Y_Test)
print("model 1")
print(Accuracy)

#Model 2 KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

Model = KNeighborsClassifier()
Model.fit(X_Train,Y_Train)

Prediction = Model.predict(X_Test)
# print(Prediction)
Accuracy = Model.score(X_Test,Y_Test)
print("model 2")
print(Accuracy)

# print(X_Train.describe())
# print(X_Train.info())

