import numpy as np

def rules(Process_Temperature, Rotational_Speed, Torque, Tool_wear):
    if Process_Temperature > 318:
        return 1
    
    if Rotational_Speed > 1800:
        return 1
    
    if Torque > 55:
        return 1
    
    if Tool_wear > 200:
        return 1

    return 0





    return rules

