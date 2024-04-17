import socket
import pickle
import Car_Controller

PORT = 9998

car_controller = Car_Controller.Car_Controller()

socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
socket.setblocking(False)
socket.bind(("", PORT))

missing_data = 0
while missing_data < 100000:
    try:
        data, address = socket.recvfrom(1024)
        data = data.decode("utf-8")
        data = data.split(",")
        #data = pickle.loads(data)

        car_controller.set_speed_and_steering(int(data[0]), int(data[1]))

        missing_data = 0
    except:
        missing_data += 1

print("disconnected")