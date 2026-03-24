import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc


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
    x, y, test_size=0.2, random_state=42,
)

model = GradientBoostingClassifier()
#Model training
#model.fit(Xlearn, Ylearn)
#Model testing

param_grid = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [2, 3, 5],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5],
    "subsample": [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring="f1",          # use "recall" if catching failures matters most
    cv=5,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(Xlearn, Ylearn)

best_model = grid_search.best_estimator_

print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

y_pred = best_model.predict(XTest)
#Results
#probabilities = model.predict_proba(XTest)[:,1]

#y_pred = (probabilities > 0.5).astype(int)
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
