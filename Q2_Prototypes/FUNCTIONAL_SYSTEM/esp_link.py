import json

def espStatus(ser):
    #Establish the serial connection
    Status = ser.readline().decode('utf-8') #Read buffer for JSON file 

    try:
        json_data = json.loads(Status)   #Create Python dictionary to store data for access later on 
    except json.JSONDecodeError:
        print("Error: The response is not valid JSON")
        json_data = {     #turn off all system green lights 
            "Pedal 1": 0,
            "touch": 0,
            "pot": 0
        }

    return json_data