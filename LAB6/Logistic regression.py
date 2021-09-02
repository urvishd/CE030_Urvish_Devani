

#Importing libraries
import numpy as np 
import pandas as pd 
import io
import matplotlib.pyplot as plt

# reading the csv file, del 2 columns from the file, checking first few rows of the file


data = pd.read_csv('BuyComputer.csv')

data.drop(columns=['User ID',],axis=1,inplace=True)
data.head()

#Declare label as last column in the source file
label=data['Purchased']
label

#Declaring X as all columns excluding last
X=data[['Age','EstimatedSalary']]
X

# Splitting data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,label,test_size=0.3,random_state=30)
X_train, X_test, y_train, y_test

# Sacaling data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Variabes to calculate sigmoid function
y_pred = []
len_x = len(X_train[0])
w = []
b = 0.2
print(len_x)

entries = len(X_train[:,0])
entries

for weights in range(len_x):
    w.append(0)
w

# Sigmoid function
def sigmoid(z):
  return 1/(1+np.exp(-z))

def predict(inputs,w):
    z=np.dot(inputs,w)
    return sigmoid(z)

#Loss function
def loss_func(features,labels,w):
    observations = len(labels)
    predictions = predict(features, w)
    class1_cost = -labels*np.log(predictions)
    class2_cost = (1-labels)*np.log(1-predictions)
    cost = class1_cost - class2_cost
    cost = cost.sum() / observations
    return cost

dw = []
db = 0
J = 0
alpha = 0.1
for x in range(len_x):
    dw.append(0)

def update_weights(features, labels, weights, lr):    
    
    N = len(features)    
    predictions = predict(features, weights)    
    gradient = np.dot(features.T,  predictions - labels)    
    gradient /= N    
    gradient *= lr    
    weights -= gradient

    return weights

#Repeating the process 3000 times
cost_history = []
for i in range(3000):
    w = update_weights(X, label, w, alpha)

    #Calculate error for auditing purposes
    cost = loss_func(X, label, w)
    cost_history.append(cost)

    # Log Progress
    if i % 1000 == 0:
        print("iter: "+ str(i) + " cost: "+str(cost))

#Print weight
w

#print bias
b

#predicting the label
predicted_labels=predict(X_test,w)

#print actual and predicted values in a table
predicted_labels,label

# Calculating accuracy of prediction
diff = predicted_labels - y_test
acc=1.0 - (float(np.count_nonzero(diff)) / len(diff))
acc

"""#Using sklearn LogisticRegression model"""

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
LR = LogisticRegression(random_state = 26)

#Fit
LR=LR.fit(X_train,y_train)
#predicting the test label with LR. Predict always takes X as input
y_pred=LR.predict(X_test)
y_pred