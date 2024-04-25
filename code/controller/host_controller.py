import pickle
import socket
from Gamepad import Gamepad

PORT = 6001
RECIPIENT_ADDRESS = "10.42.0.1"

gamepad = Gamepad()

socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

while gamepad.a == 0:
    # data_to_send = pickle.dumps([gamepad.left_joystick[0], gamepad.right_trigger]) 
    # print([gamepad.left_joystick[0], gamepad.right_trigger])
    trigger = int(gamepad.right_trigger*100)
    joystick = int(gamepad.left_joystick[0]*-50 + 50)
    print(trigger, joystick)

    socket.sendto(bytes(f"{trigger},{joystick}", "utf-8"), (RECIPIENT_ADDRESS, PORT))

socket.close()