import math
import threading
from inputs import get_gamepad

class Gamepad:
    MAX_TRIG_VALUE = math.pow(2, 8)
    MAX_JOY_VALUE = math.pow(2, 15)
    
    def __init__(self) -> None:
        self.left_joystick = [0, 0]
        self.right_joystick = [0, 0]

        self.left_trigger = 0
        self.right_trigger = 0

        self.left_button = 0
        self.right_button = 0

        self.a = 0
        self.b = 0
        self.x = 0
        self.y = 0

        self.monitor_thread = threading.Thread(target=self.monitor_values, args=(), daemon=True)
        self.monitor_thread.start()

    def monitor_values(self):
        while True:
            events = get_gamepad()
            for event in events:
                if event.code == 'ABS_Y':
                    self.left_joystick[1] = event.state / Gamepad.MAX_JOY_VALUE
                elif event.code == 'ABS_X':
                    self.left_joystick[0] = event.state / Gamepad.MAX_JOY_VALUE
                elif event.code == 'ABS_RY':
                    self.right_joystick[1] = event.state / Gamepad.MAX_JOY_VALUE
                elif event.code == 'ABS_RX':
                    self.right_joystick[0] = event.state / Gamepad.MAX_JOY_VALUE
                elif event.code == 'ABS_Z':
                    self.left_trigger = event.state / Gamepad.MAX_TRIG_VALUE
                elif event.code == 'ABS_RZ':
                    self.right_trigger = event.state / Gamepad.MAX_TRIG_VALUE
                elif event.code == 'BTN_TL':
                    self.left_button = event.state
                elif event.code == 'BTN_TR':
                    self.right_button = event.state
                elif event.code == 'BTN_SOUTH':
                    self.a = event.state
                elif event.code == 'BTN_NORTH':
                    self.y = event.state #previously switched with X
                elif event.code == 'BTN_WEST':
                    self.x = event.state #previously switched with Y
                elif event.code == 'BTN_EAST':
                    self.b = event.state
