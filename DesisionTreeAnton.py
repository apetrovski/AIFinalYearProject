import numpy as np
import sys
import matplotlib
import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#from sklearn.model_selection import


df = pandas.read_csv("DatasetAnton.csv") #Reads the dataset.csv file

features = ["Air_Temperature","Process_Temperature","Rotational_Speed","Torque","Tool_wear"] #Splits the data set into features of the machine and operation values
y = df["Operation"]

x = df[features]
trainingvalues = int(len(x)*0.8) #splits the data into training set and testing set with a ratio of 80:20 split
Xlearn = x[:trainingvalues]
Ylearn = y[:trainingvalues]
XTest = x[trainingvalues:]
YTest = y[trainingvalues:]

tree_model = DecisionTreeClassifier(
    criterion = "entropy",
    max_depth = 3,
    class_weight = "balanced",



)