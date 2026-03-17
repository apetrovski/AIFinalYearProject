import numpy as np

def rules(air_temp,process_temp, rota_speed, torque, tool_wear):
    if (process_temp >= 318):
        return 1
    
    if (rota_speed >= 1800 or rota_speed <= 1200):
        return 1
    
    if (torque >= 65):
        if (torque < 13.2):
            return 1
        if (rota_speed >= 1228):
            return 1
        elif (rota_speed <1228):
             if (tool_wear > 141.5):
                 return 1
             else:
                 return 0   
    if (process_temp - air_temp < 8.6 and rota_speed):
        return 1
    else:
        return 0

