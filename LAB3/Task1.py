

#importing libraries
import numpy as np 
import pandas as pd
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB, MultinomialNB

"""Step 2: Prepare dataset."""

dataset = pd.read_csv('Dataset1.csv')
print("Data :- \n",dataset)
print("Data Statistics :- \n",dataset.describe())

"""Step 3: Digitize the data set using encoding"""

#creating labelEncoder
le = preprocessing.LabelEncoder()

# Converting string labels into numbers.
outlook_encoded=le.fit_transform(dataset['Outlook'])
print("Outlook:" ,outlook_encoded)
temp_encoded=le.fit_transform(dataset['Temp'])
print("Temp:" ,temp_encoded)
humidity_encoded=le.fit_transform(dataset['Humidity'])
print("Humidity:" ,humidity_encoded)
wind_encoded=le.fit_transform(dataset['Wind'])
print("Wind:" ,wind_encoded)
play_encoded=le.fit_transform(dataset['Play'])
print("Play:" ,play_encoded)

"""Step 4: Merge different features to prepare dataset"""

#Combinig Outlook,Temp,Humidity and Wind into single listof tuples
features=tuple(zip(outlook_encoded,temp_encoded,humidity_encoded,wind_encoded))
print("Features:",features)

"""Step 5: Train ’Naive Bayes Classifier’"""

#Create a Classifier
model=MultinomialNB()
# Train the model using the training sets
model.fit(features,play_encoded)

"""Step 6: Predict Output for new data"""

#Predict Output
predicted= model.predict([[1,1,2,0]]) # 1:Overcast, 2:Mild
print("Predicted Value:", predicted)

#Predict Output
predicted= model.predict([[1,1,0,1]]) # 0:Overcast, 2:Mild
print("Predicted Value:", predicted)

#Predict Output
predicted= model.predict([[0,1,2,0]]) # 0:Overcast, 2:Mild
print("Predicted Value:", predicted)