import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc

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
    x, y, test_size=0.2, random_state=42,
)

model = GradientBoostingClassifier()
#Model training
#model.fit(Xlearn, Ylearn)
#Model testing

param_random = {
    "n_estimators": [50, 100, 200],
    "learning_rate": [0.01, 0.05, 0.1],
    "max_depth": [2, 3, 5],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5],
    "subsample": [0.8, 1.0]
}

random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_random,
    n_iter=20,
    cv=5,
    n_jobs=-1,
    verbose=2,
    random_state=42
)

random_search.fit(Xlearn, Ylearn)

best_model = random_search.best_estimator_

print("Best Parameters:", random_search.best_params_)
print("Best Cross-Validation Score:", random_search.best_score_)

train_score = best_model.score(Xlearn, Ylearn)
test_score = best_model.score(XTest, YTest)

print("Train:", train_score)
print("Test:", test_score)
print("\n")

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
plt.plot([0,1], [0,1], linestyle="--") 
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.savefig("results/ROC_curve_gradient_boosting.png")
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
plt.savefig("results/confusion_matrix_gradient_boosting.png")
plt.show()
plt.close()

with open("results/results_gradient_boosting.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1 Score: {f1}\n")
    f.write(f"AUC: {roc_auc}\n")
    f.write(f"Best Parameters: {random_search.best_params_}\n")
    f.write(f"Best Cross-Validation Score: {random_search.best_score_}\n")
    f.write(f"Training score:{train_score}\n")
    f.write(f"Testing score:{test_score}\n")
    f.write("Feature Importance:\n")
    for feature, importance in zip(features, model.feature_importances_):
        f.write(f"{feature}: {importance:.4f}\n")