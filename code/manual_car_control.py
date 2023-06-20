import keyboard
import Car_Controller

car_controller = Car_Controller.Car_Controller(33, 32)

while True:
    if keyboard.is_pressed("a"):
        car_controller.set_steering(-1)
    elif keyboard.is_pressed("d"):
        car_controller.set_steering(1)
    else:
        car_controller.set_steering(0)