import numpy as np

def rules(air_temp, process_temp, rota_speed, torque, tool_wear):
    risk = -5.0   # baseline keeps most normal cases as non-failures

    # 1. Air temperature
    # Ambient temperature alone is not usually a direct failure cause,
    # so it gets a smaller weighting.
    if air_temp > 303:
        risk += 0.4 * (air_temp - 303)
    elif air_temp < 297:
        risk += 0.2 * (297 - air_temp)

    # 2. Process temperature
    # More important than ambient because overheating directly affects the machine.
    if process_temp > 313:
        risk += 0.8 * (process_temp - 313)

    # 3. Temperature difference
    # The normal difference is about 10 K. If the gap becomes too small,
    # cooling may be poor or heat transfer abnormal.
    temp_diff = process_temp - air_temp
    if temp_diff < 8.5:
        risk += 1.2 * (8.5 - temp_diff)
    elif temp_diff > 12.5:
        risk += 0.4 * (temp_diff - 12.5)

    # 4. Rotational speed
    # Speeds far from nominal raise failure risk.
    if rota_speed > 1700:
        risk += 0.006 * (rota_speed - 1700)
    elif rota_speed < 1200:
        risk += 0.008 * (1200 - rota_speed)

    # 5. Torque
    # High torque indicates overload; very low torque may indicate abnormal operation.
    if torque > 45:
        risk += 0.18 * (torque - 45)
    elif torque < 15:
        risk += 0.12 * (15 - torque)

    # 6. Tool wear
    # Wear accumulates over time, so this should matter more at higher values.
    if tool_wear > 120:
        risk += 0.025 * (tool_wear - 120)
    if tool_wear > 200:
        risk += 0.04 * (tool_wear - 200)

    # 7. Interaction terms
    # Failures usually come from combinations, not just single variables.
    if torque > 50 and rota_speed < 1250:
        risk += 2.0

    if process_temp > 315 and tool_wear > 180:
        risk += 2.0

    if temp_diff < 8.5 and rota_speed > 1650:
        risk += 1.5

    # Convert risk score to probability
    failure_prob = 1 / (1 + np.exp(-risk))

    # Generate failure outcome
    if failure_prob >= 0.5:
        return 1
    else: 
        return 0