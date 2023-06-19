import keyboard
import car_controlls

car_controller = car_controlls.Car_Controller(33, 32)

while True:
    if keyboard.is_pressed("a"):
        car_controller.set_steering(-0.5)
    elif keyboard.is_pressed("d"):
        car_controller.set_steering(0.5)
    else:
        car_controller.set_steering(0)