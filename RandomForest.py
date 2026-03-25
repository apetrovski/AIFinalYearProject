import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

import os
os.makedirs("results", exist_ok=True)

df = pd.read_csv("dataset_output.csv") #Reads the dataset.csv file

features = ["Air_Temperature","Process_Temperature","Rotational_Speed","Torque","Tool_wear"] #Splits the data set into features of the machine and operation values
y = df["Operation"]

x = df[features]
#trainingvalues = int(len(x)*0.8) #splits the data into training set and testing set with a ratio of 80:20 split
#Xlearn = x[:trainingvalues]
#Ylearn = y[:trainingvalues]
#XTest = x[trainingvalues:]
#YTest = y[trainingvalues:]

Xlearn, XTest, Ylearn, YTest = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

parameter_test = {
    "criterion": ["gini", "entropy", "log_loss"],
    'n_estimators': [100, 200, 300, 400, 500], #number of decision trees
    "max_depth": [3, 5, 7, 10, 15, 20],
    'min_samples_split': [2, 5, 10],           #minimum samples needed to split a node
    'min_samples_leaf': [1, 2, 4],             #minimum samples required in a leaf node after a split
    'max_features': ['sqrt', 'log2'],
    'class_weight': [None, 'balanced']
}


rf_model = RandomForestClassifier(random_state=3) 


random_search = RandomizedSearchCV(
    rf_model, 
    param_distributions = parameter_test,
    n_iter = 33,
    cv = 3,
    scoring = 'recall',
    n_jobs=-1,  #number of cpus - all
    random_state= 3
    )

random_search.fit(Xlearn, Ylearn)

best_model = random_search.best_estimator_
#print(type(best_model))

y_pred = best_model.predict(XTest)
# Best parameters

print("Best Parameters:", random_search.best_params_)

#Everything bellow is for results.

accuracy = accuracy_score(YTest, y_pred)
precision = precision_score(YTest, y_pred)
recall = recall_score(YTest, y_pred)
f1 = f1_score(YTest, y_pred)
cm = confusion_matrix(YTest, y_pred)

y_scores = best_model.predict_proba(XTest)[:,1]
fpr, tpr, thresholds = roc_curve(YTest, y_scores)
roc_auc = auc(fpr, tpr)
print("AUC:", roc_auc)

train_score = best_model.score(Xlearn, Ylearn)
test_score = best_model.score(XTest, YTest)

print("Train:", train_score)
print("Test:", test_score)
print("\n")

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
plt.savefig("results/ROC_curve_RandomForest.png")
plt.show()
plt.close()


plt.figure(figsize=(6, 5))
seaborn.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Predicted: No Failure", "Predicted: Failure"],
            yticklabels=["Actual: No Failure", "Actual: Failure"])

plt.title("Confusion Matrix for Predictive Maintenance Model")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("results/confusion_matrix_RandomForest.png")
plt.show()
plt.close()

with open("results/results_RandomForest.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1 Score: {f1}\n")
    f.write(f"Best Parameters: {random_search.best_params_}\n")
    f.write(f"Best Cross-Validation Score: {random_search.best_score_}\n")
    f.write("Feature Importance:\n")
    for feature, importance in zip(features, best_model.feature_importances_):
        f.write(f"{feature}: {importance:.4f}\n")