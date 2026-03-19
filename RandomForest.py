import numpy as np
import sys
import matplotlib
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

df = pd.read_csv("dataset_output.csv") #Reads the dataset.csv file

features = ["Air_Temperature","Process_Temperature","Rotational_Speed","Torque","Tool_wear"] #Splits the data set into features of the machine and operation values
y = df["Operation"]

x = df[features]
trainingvalues = int(len(x)*0.8) #splits the data into training set and testing set with a ratio of 80:20 split
Xlearn = x[:trainingvalues]
Ylearn = y[:trainingvalues]
XTest = x[trainingvalues:]
YTest = y[trainingvalues:]

rf_model = RandomForestClassifier(n_estimators=100, random_state=42) #n_estimators → number of decision trees. random_state → ensures reproducible results

rf_model.fit(Xlearn, Ylearn)

y_pred = rf_model.predict(XTest)

#Everything bellow is for results.

accuracy = accuracy_score(YTest, y_pred)
precision = precision_score(YTest, y_pred)
recall = recall_score(YTest, y_pred)
f1 = f1_score(YTest, y_pred)
cm = confusion_matrix(YTest, y_pred)

y_scores = rf_model.predict_proba(XTest)[:,1]
fpr, tpr, thresholds = roc_curve(YTest, y_scores)
roc_auc = auc(fpr, tpr)
print("AUC:", roc_auc)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
print("Confusion Matrix:\n", cm)

plt.figure()

plt.plot(fpr, tpr, label="ROC curve (AUC = %0.3f)" % roc_auc)
plt.plot([0,1], [0,1], linestyle="--")   # Random classifier line

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")

plt.show()

plt.figure(figsize=(6, 5))
seaborn.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Predicted: No Failure", "Predicted: Failure"],
            yticklabels=["Actual: No Failure", "Actual: Failure"])

plt.title("Confusion Matrix for Predictive Maintenance Model")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()