import pandas as pd
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

import os
os.makedirs("results", exist_ok=True)

df = pd.read_csv("dataset_output.csv") #Reads the dataset.csv file

features = ["Air_Temperature","Process_Temperature","Rotational_Speed","Torque","Tool_wear"] #Splits the data set into features of the machine and operation values
y = df["Operation"]

x = df[features]
trainingvalues = int(len(x)*0.8) #splits the data into training set and testing set with a ratio of 80:20 split
Xlearn = x[:trainingvalues]
Ylearn = y[:trainingvalues]
XTest = x[trainingvalues:]
YTest = y[trainingvalues:]

Xlearn, XTest, Ylearn, YTest = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


tree_model = DecisionTreeClassifier(random_state=42)

# Hyperparameter grid
parameter_test = {
    "criterion": ["gini", "entropy", "log_loss"],
    "max_depth": [3, 5, 7, 10, 15, 20],
    "min_samples_split": [2, 5, 10, 20],
    "min_samples_leaf": [1, 2, 5, 10],
    "max_features": [None, "sqrt", "log2"],
    "ccp_alpha": [0.0, 0.001, 0.01]
}

# Grid search
grid_search = GridSearchCV(
    estimator=tree_model,
    param_grid=parameter_test,
    scoring="recall",   
    cv=5,
    n_jobs=-1,
    verbose=2
)

# Fit on training data
grid_search.fit(Xlearn, Ylearn)

# Best model
best_model = grid_search.best_estimator_

print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

plt.figure(figsize=(20,10))

tree.plot_tree(
    best_model,
    feature_names = x.columns,
    class_names = ["Normal", "Failure"],
    filled = True
)

plt.show()

probabilities = best_model.predict_proba(XTest)[:,1]

train_score = best_model.score(Xlearn, Ylearn)
test_score = best_model.score(XTest, YTest)

print("Train:", train_score)
print("Test:", test_score)

y_pred = (probabilities > 0.5).astype(int)
accuracy = accuracy_score(YTest, y_pred)
precision = precision_score(YTest, y_pred)
recall = recall_score(YTest, y_pred)
f1 = f1_score(YTest, y_pred)
cm = confusion_matrix(YTest, y_pred)

y_scores = best_model.predict_proba(XTest)[:,1]
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
plt.savefig("results/ROC_curve_DecisionTree.png")
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
plt.savefig("results/confusion_matrix_DecisionTree.png")
plt.show()
plt.close()

with open("results/results_DecisionTree.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1 Score: {f1}\n")
    f.write(f"AUC: {roc_auc}\n")
    f.write(f"Best Parameters: {grid_search.best_params_}\n")
    f.write(f"Best Cross-Validation Score: {grid_search.best_score_}\n")
    f.write(f"Training score:{train_score}\n")
    f.write(f"Testing score:{test_score}\n")
    f.write("Feature Importance:\n")
    for feature, importance in zip(features, best_model.feature_importances_):
        f.write(f"{feature}: {importance:.4f}\n")