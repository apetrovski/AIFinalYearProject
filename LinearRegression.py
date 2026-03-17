import pandas
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn

df = pandas.read_csv("dataset_output.csv")

features = ["Air_Temperature","Process_Temperature","Rotational_Speed","Torque","Tool_wear"]
y = df["Operation"]

# Split data
x = df[features]
trainingvalues = int(len(x)*0.8)
Xlearn = x[:trainingvalues]
Ylearn = y[:trainingvalues]
XTest = x[trainingvalues:]
YTest = y[trainingvalues:]

# Create model
#model = LinearRegression()

# Train model (THIS calculates the weights)
#model.fit(Xlearn, Ylearn)

# Get coefficients (weights)
#print("\n")
#print("Intercept LinearRegression (β0):", model.intercept_)
#print("Coefficients LinearRegression (β1, β2, β3):", model.coef_)
#print("\n")

# Create model
log_model = LogisticRegression()

# Train model (calculates weights)
log_model.fit(Xlearn, Ylearn)

# Get weights
print("Intercept LogisticRegression (β0):", log_model.intercept_)
print("Coefficients LogisticRegression (β1, β2, β3):", log_model.coef_)
print("\n")

# Predict probabilities
y_prob = log_model.predict_proba(XTest)[:,1]  # Probability of failure

# Predict classes (0 = normal, 1 = failure)
y_pred = log_model.predict(XTest)

beta = log_model.coef_[0]  # Convert (1,5) → (5,)
intercept = log_model.intercept_[0]

# Loop over each test sample
probabilities = []
for i in range(XTest.shape[0]):
    x = XTest.iloc[i].values  # or X_test[i] if numpy array
    z = intercept + np.dot(beta, x)
    prob = 1 / (1 + np.exp(-z))
    probabilities.append(prob)

probabilities = np.array(probabilities)

# probabilities is an array of shape (num_samples,)
y_pred = (probabilities > 0.5).astype(int)

accuracy = accuracy_score(YTest, y_pred)
precision = precision_score(YTest, y_pred)
recall = recall_score(YTest, y_pred)
f1 = f1_score(YTest, y_pred)
cm = confusion_matrix(YTest, y_pred)

y_scores = log_model.predict_proba(XTest)[:,1]
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