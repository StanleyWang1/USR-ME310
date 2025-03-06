import json

def espStatus(ser):
    # Read a line from the serial and decode it
    status_line = ser.readline().decode('utf-8').strip()
    
    try:
        json_data = json.loads(status_line)
    except json.JSONDecodeError:
        print("Error: The response is not valid JSON:", status_line)
        json_data = {  # Default values if parsing fails
            "Pedal 1": 0,
            "touch": 0,
            "pot": 0
        }
    
    return json_data
