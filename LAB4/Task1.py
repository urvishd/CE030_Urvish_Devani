# -*- coding: utf-8 -*-
"""lab4_0.ipynb


Task-1:Try the algo on Same Whether dataset - LabelEncoding of features: and Train test Division 80%-20% with Gini Index as attribute selection measure
"""

#Import library
from sklearn import preprocessing
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np

#Predictor variables
Outlook = ['Rainy', 'Rainy', 'Overcast', 'Sunny', 'Sunny', 'Sunny', 'Overcast',
            'Rainy', 'Rainy', 'Sunny', 'Rainy','Overcast', 'Overcast', 'Sunny']
Temperature = ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 'Cool',
                'Mild', 'Cool', 'Mild', 'Mild', 'Mild', 'Hot', 'Mild']
Humidity = ['High', 'High', 'High', 'High', 'Normal', 'Normal', 'Normal',
            'High', 'Normal', 'Normal', 'Normal', 'High', 'Normal', 'High']
Wind = ['False', 'True', 'False', 'False', 'False', 'True', 'True',
            'False', 'False', 'False', 'True', 'True', 'False', 'True']

#Class Label:
Play = ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No',
'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No']

#creating labelEncoder
le = preprocessing.LabelEncoder()
# Converting string labels into numbers.
Outlook_encoded = le.fit_transform(Outlook)
Outlook_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Outllok mapping:",Outlook_name_mapping)

Temperature_encoded = le.fit_transform(Temperature)
Temperature_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Temperature mapping:",Temperature_name_mapping)

Humidity_encoded = le.fit_transform(Humidity)
Humidity_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Humidity mapping:",Humidity_name_mapping)

Wind_encoded = le.fit_transform(Wind)
Wind_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Wind mapping:",Wind_name_mapping)

Play_encoded = le.fit_transform(Play)
Play_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Play mapping:",Play_name_mapping)

print("\n\n")
print("Weather:" ,Outlook_encoded)
print("Temerature:" ,Temperature_encoded)
print("Humidity:" ,Humidity_encoded)
print("Wind:" ,Wind_encoded)
print("Play:" ,Play_encoded)

weather=pd.DataFrame(Outlook_encoded, columns=['Outlook'])
weather['Temperature']=pd.DataFrame(Temperature_encoded, columns=['Temperature'])
weather['Humidity']=pd.DataFrame(Humidity_encoded, columns=['Humidity'])
weather['Wind']=pd.DataFrame(Wind_encoded, columns=['Wind'])
weather['Play']=pd.DataFrame(Play_encoded, columns=['Play'])
print(weather)

from sklearn.model_selection import train_test_split
X=weather.values[:,0:4]
Y=weather.values[:,-1]
X_train, X_test, y_train, y_test = train_test_split( 
    X, Y, test_size = 0.2, random_state = 100)

#Create a Decision Tree Classifier (using Gini)
clf_gini = DecisionTreeClassifier(criterion = "gini",
            random_state = 26,max_depth=7, min_samples_leaf=26)

clf_gini.fit(X_train, y_train)

# Predict the classes of test data
y_pred = clf_gini.predict(X_test)
print("Predicted values:")
print(y_pred)

# Model Accuracy
from sklearn import metrics
print("Confusion Matrix: ",
        metrics.confusion_matrix(y_test, y_pred))
print ("Accuracy : ",
    metrics.accuracy_score(y_test,y_pred)*100)
print("Report : ",
    metrics.classification_report(y_test, y_pred))
