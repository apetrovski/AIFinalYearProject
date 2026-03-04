import numpy
import sys
import matplotlib
#matplotlib.use('Agg')

import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


df = pandas.read_csv("DatasetAnton.csv")

features = ["Air_Temperature","Process_Temperature","Rotational_Speed","Torque","Tool_wear"]
y = df["Operation"]

x = df[features]
trainingvalues = int(len(x)*0.8)
Xlearn = x[:trainingvalues]
Ylearn = y[:trainingvalues]
XTest = x[trainingvalues:]
YTest = y[trainingvalues:]

# Train decision tree
dtree = DecisionTreeClassifier(max_depth=10, random_state=0)  # Limit depth for clarity
dtree.fit(Xlearn, Ylearn)

for feature in features:

    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]

    print("\nFeature:", feature)
    print("Number of outliers:", len(outliers))

    if len(outliers) != 0:
        print("Failure breakdown:")
        print(outliers["Operation"].value_counts())

# Evaluate accuracy
pred = dtree.predict(XTest)
print("Test Accuracy:", accuracy_score(YTest, pred))
results = dtree.predict(XTest)

# Convert YTest to a NumPy array for consistency
YTest_array = YTest.to_numpy()

# Compute element-wise difference (0 = correct, ±1 = incorrect)
difference = results - YTest_array
print(difference)

numberOfFailureLearn = (Ylearn == 1).sum()
print("Number of failures in training dataset: ",numberOfFailureLearn)

numberOfFailureTest = (YTest == 1).sum()
print("Number of failures in Test dataset: ",numberOfFailureTest)

print("Accuracy:", accuracy_score(YTest, results))
print("\nConfusion matrix:\n", confusion_matrix(YTest, results))
print("\nClassification report:\n", classification_report(YTest, results))
accuracy = (difference == 0).sum() / len(difference)
print("Manual accuracy:", accuracy)
print("\n")
for feature, importance in zip(features, dtree.feature_importances_):
    print(f"{feature}: {importance*100:.3f}")
# Confusion matrix
cm = confusion_matrix(YTest, results)

# Visualize as heatmap
plt.figure(figsize=(6, 5))
seaborn.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Predicted: No Failure", "Predicted: Failure"],
            yticklabels=["Actual: No Failure", "Actual: Failure"])

plt.title("Confusion Matrix for Predictive Maintenance Model")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.show()

#air_temp = float(input("Air temperature [K]: "))
#proc_temp = float(input("Process temperature [K]: "))
#speed = float(input("Rotational speed [rpm]: "))
#torque = float(input("Torque [Nm]: "))
#wear = float(input("Tool wear [min]: "))

#user_input = pandas.DataFrame([{
#    "Air temperature [K]": air_temp,
#    "Process temperature [K]": proc_temp,
#    "Rotational speed [rpm]": speed,
#    "Torque [Nm]": torque,
#    "Tool wear [min]": wear
#}])

#print("Prediction:", dtree.predict(user_input))

#dtree = DecisionTreeClassifier()
#dtree = dtree.fit(x, y)

#tree.plot_tree(dtree, feature_names=features)
#plt.show
#plt.scatter(
#    df['Torque [Nm]'], 
#    df['Rotational speed [rpm]'], 
#    c=(df['Failure Type'] != 'No Failure').astype(int),  # color = failure
#    s=df['Tool wear [min]'] * 0.5,                      # size = tool wear
#    alpha=0.7, cmap='coolwarm'
#)
#plt.xlabel('Torque [Nm]')
#plt.ylabel('Rotational speed [rpm]')
#plt.title('4D Scatter Plot: Torque vs Rotational Speed')
#plt.colorbar(label='Failure (1=Failure, 0=No Failure)')
#plt.show()

