import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

# Height---Weight---Shoe Size
X = [[190,70,44], [166,65,45], [190,90,47], [175,64,39], [171,75,40], [177,80,42], [160,60,38], [144,54,37]]
Y = ['male','male','male','male','female','female','female','female']

# Predict for this vector (Height, Weight, Shoe Size)
P = [[190,80,64]]     # this is the data, i want to predict is male or female

# Decision Tree Algorithm
clf = DecisionTreeClassifier()
clf = clf.fit(X,Y)
print("Using Decision Tree Prediction is:  " + str(clf.predict(P)))

# K Neighbors Algorithm
knn = KNeighborsClassifier()
knn = knn.fit(X,Y)
print("Using K Neighbors Classifier is:  " + str(knn.predict(P)))

# MLPC Algorithm
kmlpc = MLPClassifier()
kmlpc = kmlpc.fit(X,Y)
print("Using MLPC Classifier is:  " + str(kmlpc.predict(P)))

# RandomForest Algorithm
rfor = RandomForestClassifier()
rfor = rfor.fit(X,Y)
print("Using RandomForest Classifier is:  " + str(rfor.predict(P)))
