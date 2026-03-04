import numpy as np

def rules(process_temp, rota_speed, torque, tool_wear):
    if process_temp > 318:
        return 1
    
    if rota_speed > 1800:
        return 1
    
    if torque > 55:
        return 1
    
    if tool_wear > 200:
        return 1

    return 0





    return rules

