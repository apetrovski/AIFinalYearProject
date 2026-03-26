import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn
from sklearn.model_selection import train_test_split

import os
os.makedirs("results", exist_ok=True)

scaler = StandardScaler()

df = pd.read_csv("dataset_output2.csv")

features = ["Air_Temperature","Process_Temperature","Rotational_Speed","Torque","Tool_wear"]
y = df["Operation"]

# Create model
log_model = LogisticRegression(max_iter=1000)

# Split data
x = df[features]
#trainingvalues = int(len(x)*0.8)
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

Xlearn_scaled = scaler.fit_transform(Xlearn)
XTest_scaled = scaler.transform(XTest)

parameter_test = [
    {
        "penalty": ["l1"],
        "solver": ["liblinear", "saga"],
        "C": [0.01, 0.1, 1, 10],
        "class_weight": [None, "balanced"]
    },
    {
        "penalty": ["l2"],
        "solver": ["lbfgs", "liblinear"],
        "C": [0.01, 0.1, 1, 10],
        "class_weight": [None, "balanced"]
    },
    {
        "penalty": ["elasticnet"],
        "solver": ["saga"],
        "C": [0.01, 0.1, 1, 10],
        "l1_ratio": [0.2, 0.5, 0.8],
        "class_weight": [None, "balanced"]
    }
]



grid_search = GridSearchCV(
    log_model, 
    parameter_test,
    cv = 10,
    scoring = 'accuracy',
    n_jobs=-1,  #number of cpus - all
    verbose=2
    )

Xlearn_scaled = pd.DataFrame(Xlearn_scaled, columns=Xlearn.columns)
XTest_scaled = pd.DataFrame(XTest_scaled, columns=XTest.columns)

grid_search.fit(Xlearn_scaled, Ylearn)

best_model = grid_search.best_estimator_
#print(type(best_model))

y_pred = best_model.predict(XTest_scaled)
# Best parameters

print("Best Parameters:", grid_search.best_params_)

# Get weights
print("Intercept LogisticRegression (β0):", best_model.intercept_)
print("Coefficients LogisticRegression (β1, β2, β3):", best_model.coef_)
print("\n")

train_score = best_model.score(Xlearn_scaled, Ylearn)
test_score = best_model.score(XTest_scaled, YTest)

print("Train:", train_score)
print("Test:", test_score)
print("\n")
# Predict probabilities
y_prob = best_model.predict_proba(XTest_scaled)[:,1]  # Probability of failure

# Predict classes (0 = normal, 1 = failure)
#y_pred = log_model.predict(XTest)

beta = best_model.coef_[0]  # Convert (1,5) → (5,)
intercept = best_model.intercept_[0]

# probabilities is an array of shape (num_samples,)
y_pred = best_model.predict(XTest_scaled)

accuracy = accuracy_score(YTest, y_pred)
precision = precision_score(YTest, y_pred)
recall = recall_score(YTest, y_pred)
f1 = f1_score(YTest, y_pred)
cm = confusion_matrix(YTest, y_pred)

y_scores = best_model.predict_proba(XTest_scaled)[:,1]
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
plt.savefig("results/ROC_curve_LogisticalRegression.png")
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
plt.savefig("results/confusion_matrix_LogisticalRegression.png")
plt.show()
plt.close()

with open("results/results_LogsiticalRegression.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Precision: {precision}\n")
    f.write(f"Recall: {recall}\n")
    f.write(f"F1 Score: {f1}\n")
    f.write(f"Best Parameters: {grid_search.best_params_}\n")
    f.write(f"Best Cross-Validation Score: {grid_search.best_score_}\n")
    f.write("Coefficients:\n")
    for feature, coef in zip(features, best_model.coef_[0]):
        f.write(f"{feature}: {coef:.4f}\n")