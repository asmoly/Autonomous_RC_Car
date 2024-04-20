import serial

class Car_Controller:
    def __init__(self, port="/dev/ttyACM", baudrate=115200) -> None:
        port_counter = 0
        while port_counter <= 4:
            try:
                self.arduino = serial.Serial(port=f"{port}{port_counter}", baudrate=baudrate)
                break
            except:
                port_counter += 1 

    def set_speed_and_steering(self, target_speed, target_steering, translate_values=False): # speed is value from 0 - 1, steering is value from -1 - 1
        target_speed_translated = target_speed
        target_steering_translated = target_steering
        if translate_values == True:
            target_speed_translated = int(target_speed*100) # convert to sclae of 0 - 100 for arduino
            target_steering_translated = int(target_steering*50 + 50) # convert to scale of 0 - 100 with 50 being center for arduino
        
        self.arduino.write(f"{target_speed_translated},{target_steering_translated}\n".encode()) 
        self.arduino.flush()