import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
os.makedirs("results", exist_ok=True)

df = pd.read_csv("dataset_output.csv") #Reads the dataset.csv file

features = ["Air_Temperature","Process_Temperature","Rotational_Speed","Torque","Tool_wear"] #Splits the data set into features of the machine and operation values
y = df["Operation"]

#x = df[features]

for feature in features:
    plt.figure()
    sns.histplot(df[feature], kde=True)
    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)                                                 #histograms for the different features.
    plt.ylabel("Frequency")
    plt.savefig(f"results/{feature}_distribution.png")  # saves as PNG
    plt.close()

for feature in features:
    plt.figure()
    sns.boxplot(x=df[feature])
    plt.title(f"Boxplot of {feature}")                                  #Boxplots
    plt.savefig(f"results/{feature}_boxplot.png")
    plt.close()

plt.figure(figsize=(8,6))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")                          
plt.title("Correlation Matrix")                                        #Correlation matrix         
plt.savefig("results/correlation_heatmap.png")
plt.close()

sns.scatterplot(x="Torque", y="Tool_wear", hue="Operation", data=df)
plt.title("Torque vs Tool Wear")
plt.savefig("results/scatter_torque_toolwear.png")
plt.close()

sns.countplot(x="Operation", data=df)
plt.title("Class Distribution")                                         #Class distribution
plt.savefig("results/class_distribution.png")   
plt.close()