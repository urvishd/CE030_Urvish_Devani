# -*- coding: utf-8 -*-


import numpy as np 
import pandas as pd 
from sklearn import datasets
from sklearn.cluster import KMeans

"""Preapre Data

"""

dataset=sklearn.datasets.load_breast_cancer()
dataset

print(dataset.data.shape)
print(dataset.target.shape)

"""K-Mean Model"""

kmeans = KMeans(n_clusters=2, random_state=0)
prediction = kmeans.fit_predict(dataset.data)
prediction

kmeans.cluster_centers_.shape

# Scatter plot of the data points
import matplotlib.pyplot as plt
fig, ax = plt.subplots(2, 5, figsize=(8, 3))
centers = kmeans.cluster_centers_.reshape(10, 6, 5)
for axi, center in zip(ax.flat, centers):
  axi.set(xticks=[], yticks=[])
  axi.imshow(center, interpolation='nearest', cmap=plt.cm.binary)

import numpy as np
from scipy.stats import mode
labels = np.zeros_like(prediction)
for i in range(10):
  mask = (prediction == i)
  labels[mask] = mode(dataset.target[mask])[0]

from sklearn.metrics import accuracy_score
accuracy_score(dataset.target, labels)

from sklearn.metrics import confusion_matrix
import seaborn as sns
mat = confusion_matrix(dataset.target, labels)
ax = sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,xticklabels=dataset.target_names,yticklabels=dataset.target_names)

#ax.set_ylim(10,10)
plt.xlabel('true label')
plt.ylabel('predicted label');