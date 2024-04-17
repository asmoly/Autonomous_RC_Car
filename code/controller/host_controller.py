import pickle
import socket
from Gamepad import Gamepad

PORT = 9998
RECIPIENT_ADDRESS = "10.42.0.1"

gamepad = Gamepad()

socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

while gamepad.a == 0:
    data_to_send = pickle.dumps([gamepad.left_joystick[0], gamepad.right_trigger]) 
    socket.sendto(data_to_send, (RECIPIENT_ADDRESS, PORT))