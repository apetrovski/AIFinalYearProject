import numpy as np
import pandas as pd
from decision_rules import rules


np.random.seed(3) #makes sure the random number generator produces the same values everytime. Makes sure that the dataset is reproducible.

no_samples = 100000 #Dataset size. Allows it to be modified to get more data for testing.


#air_temp (ambient)
air_temp = np.random.normal(loc = 300,scale = 2,size = no_samples) #loc = 300 produces the random values to be around the mean value of 300. 300 being the temp in kelvin(27 degrees celcius). Scale is the standard deviation of the random numbers 2 being a realistic value to be in a factory environment that is temperature controlled. no_samples produces a data value for the whole size of the dataset. 
    #print((air_temperature > 310).sum()) #Test to check how many values of air temp are higher than 310k

#process_temp related to ambient air temp
process_temp = air_temp + np.random.normal(loc = 10,scale = 1.5, size = no_samples)
    #print((((process_temp > 315).sum())/no_samples)*100) #For testing produces the percentage of elvated errors occuring.
    #print((((process_temp > 318).sum())/no_samples)*100) #Likely would cause failures at this temperature.

#Rotational speed(RPM)
rota_speed = np.random.normal(loc = 1500, scale  = 100, size = no_samples) #Based around 1500rpm, standard deviation of 300
#print(((rota_speed > 1800) | (rota_speed < 1200)).sum()) #checks how many values are outside the range showing how many values are outside the range likely showing issues

#Torque (increases when speed decreases)
torque = 40 - (rota_speed/100) + np.random.normal(loc = 0, scale = 2, size = no_samples)
#print(torque)

#Tool wear
tool_wear = np.cumsum(np.random.normal(loc = 0.05, scale = 0.01,size = no_samples)) #produces a tool wear that increases incrementally with a little variation with each iteration

#Rules for failure and operation
#risk_score = (
#    0.04*(process_temp - 310) +
#    0.03*(torque - 40) +
#    0.002*(1500 - rota_speed) +
#    0.01*((tool_wear > 200))   
#) #Risk score created by summing the difference between the variables and their normal operating values and then multiplying by the weighting that each variable will have an effect.  

#print(risk_score)

#failure_prob =1/(1+np.exp(-risk_score))  #sigmoid function converts risk_score into a probability between 0-1. high risk_score produces a postive number so sigmoid produces a number close to 1. low risk is negative so sigmoid produces number close to 0
#operation = np.random.binomial(1,failure_prob) #Simulates 1 test with the probability of using failure_prob this then produces a 1 or 0 for failure or non-failure.

#print("Percentage of failure in dataset:",((operation.sum())/no_samples)*100,"%") #Used for checking how many failures are created in the dataset.

#Rules doesnt work as it is random. so model can't predict.

#operation = rules(process_temp, rota_speed,torque,tool_wear)

df = pd.DataFrame({
    "Air_Temperature": air_temp,
    "Process_Temperature": process_temp,
    "Rotational_Speed": rota_speed,
    "Torque": torque,
    "Tool_wear": tool_wear

}) #Combines the different arrays that 

# Apply the rule row by row
df["Operation"] = df.apply(
    lambda row: rules(
        row["Process_Temperature"],
        row["Rotational_Speed"],
        row["Torque"],
        row["Tool_wear"]
    ),
    axis=1
)

print(df)
print("Total failures:", df["Operation"].sum())

df.head()
df = df.round(2)
print(df)

df.to_csv("DatasetAnton.csv",index = False) #Saves the dataset into a csv file for machine learning models 