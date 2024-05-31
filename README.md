# Autonomous_RC_Car
This is an autonomous car which I built and programed. It can drive around a track made of cones all by itself. It uses a convolutional neural network (CNN) for perception and a PID controller for driving.

## Platform Components
These are the components and steps I used to build my platform:
* I used the following RC car platform as the base [traxass 4-Tec 2.0 VXL AWD chassis](https://traxxas.com/products/models/electric/4-tec-2-vxl-chassis)
* I removed the reciever and speed controller and replaced the motor with the [QuicRun Fusion Pro Motor](https://www.hobbywingdirect.com/products/quicrun-fusion-pro?variant=40105176760435) because it allows the car to drive much more steady at slower speeds
* I also replaced the servo with another one that has the same size and specifications as the original one
* I use one LiPo battery to power the entire car including its computer and motor [11.1v 5200mah 80c lipo battery](https://www.amazon.com/HOOVO-Battery-Softcase-Airplane-Helicopter/dp/B08YR15KQ8/ref=sr_1_26?crid=3T1CQWZ9E0ZQR&dib=eyJ2IjoiMSJ9.6ZjyZbfEiWJ7tdVpfAVvuvjt8XudmCtCPzSYpTleNc05mh2yo6mfVAMZlAF3hvNmckC2ep7nfSi4SJ74hudXTMdZZkmM9k-76faLNpIbUFBnAWACijy51pD85f4hKgeTCn34ava9G3eAQqr2qH7FEA7Cg1qoIk1BzBP1DYS2NZkXySz4zo3nr5_EIHcrQxVgAoznscEDhCsp7-KeBFDmaeiJQlVJl3_DbH6vAjIuQZyXx4YL04KSaSCYucGg5N-02sER1Vz146WEheqc_nmL9LXG3RgVCxjw9MLpna-Cvyk.6MwmgP6LzbD1DifoUhehMecCohP2Bj-yqL6HqxLd3QU&dib_tag=se&keywords=hoovo+battery+5200+mah+80c&qid=1717099192&sprefix=hoovo+battery+5200+mah+80%2Caps%2C218&sr=8-26)
* The battery is soldered straight to the motor power and connected to the computer via a manual switch
* I added a PCB board to the top of the car for mounting the on-board computer
* I use [NVIDIA Jetson Orin Nano](https://www.amazon.com/NVIDIA-Jetson-Orin-Nano-Developer/dp/B0BZJTQ5YP/ref=sr_1_3?crid=10A6DHVZOMHYS&dib=eyJ2IjoiMSJ9.EY0iLDd0M9dkGkWsLUJY8GyM5_RC274wWAuZvuHJK3awsYoCOS7ApMqgCFxnREWCAg487swCnHqQlOT3l6tR1k6h4vm1iKd-lu0q7GZf3BnHLHl2MwSn6SY8o_craQPrrWwd4YmGHAbw27qIeQZoBN3mC43_aI9_udkkIgCGUnczk9KcKDmB5AG7x9ctKI_02A6jp643ird4vcV8CU88zz55-YsViHnQ7JmzxYWkXCk.ZX_zvPC0fUd2RQG6fz2tD6Ggt27hwGcSPr760RlVMLA&dib_tag=se&keywords=jetson+orin+nano&qid=1717099447&sprefix=jetson+orin+nano%2Caps%2C177&sr=8-3) as the on-board computer for all the proccessing. I also soldered a connection from the switch to the barrel jack of Jetson.
* I also use Arduino Uno for motor control. All the wire connections will be specified later.
* Another small raised platform is added on top of the main platform to hold the camera. I mounted [ZED Mini Stereo Camera](https://store.stereolabs.com/products/zed-mini) onto this small platform.
* Both the camera and the Arduino board are connected to the Jetson via USB

### Wiring
* Both the servo and the motor should be connected to 5V and ground on the Arduino, the third wire on the motor should be connected to pin 9, and the 3rd pin on the servo should be connected to pin 10.
* To make the Arduino board control the motor upload the file `code/controller/arduino_ESC_servo_controller/arduino_ESC_servo_controller.ino` to the Arduino board.
* To control the motor speed and steering from Jetson download the file `code/controller/Car_Controller.py` to it. This is a class which contains functions for controlling motor speed and servo angle.

## How it Works
All code for autonomous driving is located in `code/driving/`. To run self-driving runtime, download all the files in that directory and run `main.py`. The file contains all of the parameters for the runtime like the PID controller gains, cone height, etc. 
The following is a detailed description of how my system works:

### Convolutional Neural Network
First the car runtime uses a convolutional neural network (CNN), which I trained, to find the cones in the image. The CNN is able of differentiating between green and orange cones. All the code for the CNN is located in the directory `code/cone_detection/`. The file `cone_finder_AI.py` is the program that trains the CNN. There are a few programs I used to create the dataset that are located in `code/dataset`. I made the dataset by walking around with a camera at car height and recording videos using the `collect_raw_images.py`. I then automatically pasted prepared images of cones into video frames with another program. This is done by `create_dataset.py`. The resulting frames are used as the training dataset. The model checkpoint file for the already trained CNN is in the `code/driving/` directory.

### Driving algorithm
The driving algorithm is located in the `main.py`.
* First, it uses the CNN to detect all the cones. It then finds the closest green and orange cone using their mask areas. It assumes that the green cones should always be on the left and the orange cones should always be on the right.
* After that, it finds the center point between the two cones it found.
* If the camera sees only one cone (for example when the car is turning it doesn't see the inside cone, but can still see the outside cone) it will calculate the distance to the outside cone using inverse projection and then use that distance to find a center point some distance away from the outside cone. This distance can be specified with the `NON_VISIBLE_CONE_OFFSET` parameter in the `main.py`.
* Finally, it calculates the angle to the center point and uses two PI controllers (a PID controller without the differential part) to calculate the necessary steering angle and to compute the car speed to drive to that center point. The proportional and integral gains for the PI controllers can be adjusted in `main.py`.

## Demo Videos
To see videos of this system in action download the [demo videos](https://drive.google.com/file/d/1EFXXdTS90mtLiUYapqNxHuzPokW_0hlM/view?usp=drive_link)
