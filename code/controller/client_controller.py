import socket
import pickle
from Car_Controller import Car_Controller

PORT = 9998

car_controller = Car_Controller(33, 32)

socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
socket.setblocking(False)
socket.bind(("", PORT))

missing_data = 0
while missing_data < 100000:
    try:
        data, address = socket.recvfrom(1024)
        data = pickle.loads(data)

        car_controller.set_speed(data[1])
        car_controller.set_steering(data[0])

        missing_data = 0
    except:
        missing_data += 1

del(car_controller)
print("disconnected")