# Autonomous_RC_Car
This is an autonomous car which I built and programed that can drive around a track made of cones. It uses a CNN for perception and a PID controller for the car's controls.

## The platform
These are the components and steps I used to build the platform:
* The base I used for the RC car was the [traxass 4-Tec 2.0 VXL AWD chassis](https://traxxas.com/products/models/electric/4-tec-2-vxl-chassis)
* First I removed the reciever and speed controller
* I replaced the motor with a rock crawler motor to achieve lower speeds [motor](fefe)
* I also replaced the servo with one of the same size and specifications but it shouldn't matter
* The battery I use to power the entire car is a [11.1v 5200mah 80c lipo battery](https://www.amazon.com/HOOVO-Battery-Softcase-Airplane-Helicopter/dp/B08YR15KQ8/ref=sr_1_26?crid=3T1CQWZ9E0ZQR&dib=eyJ2IjoiMSJ9.6ZjyZbfEiWJ7tdVpfAVvuvjt8XudmCtCPzSYpTleNc05mh2yo6mfVAMZlAF3hvNmckC2ep7nfSi4SJ74hudXTMdZZkmM9k-76faLNpIbUFBnAWACijy51pD85f4hKgeTCn34ava9G3eAQqr2qH7FEA7Cg1qoIk1BzBP1DYS2NZkXySz4zo3nr5_EIHcrQxVgAoznscEDhCsp7-KeBFDmaeiJQlVJl3_DbH6vAjIuQZyXx4YL04KSaSCYucGg5N-02sER1Vz146WEheqc_nmL9LXG3RgVCxjw9MLpna-Cvyk.6MwmgP6LzbD1DifoUhehMecCohP2Bj-yqL6HqxLd3QU&dib_tag=se&keywords=hoovo+battery+5200+mah+80c&qid=1717099192&sprefix=hoovo+battery+5200+mah+80%2Caps%2C218&sr=8-26)
* The battery should be soldered straight to the motor power and also to a switch which will provide power to the computer
* After I created a platform that sits on top of the RC car using a large pcb board
* Towards the back of the platform I put a [Jetson Orin Nano](https://www.amazon.com/NVIDIA-Jetson-Orin-Nano-Developer/dp/B0BZJTQ5YP/ref=sr_1_3?crid=10A6DHVZOMHYS&dib=eyJ2IjoiMSJ9.EY0iLDd0M9dkGkWsLUJY8GyM5_RC274wWAuZvuHJK3awsYoCOS7ApMqgCFxnREWCAg487swCnHqQlOT3l6tR1k6h4vm1iKd-lu0q7GZf3BnHLHl2MwSn6SY8o_craQPrrWwd4YmGHAbw27qIeQZoBN3mC43_aI9_udkkIgCGUnczk9KcKDmB5AG7x9ctKI_02A6jp643ird4vcV8CU88zz55-YsViHnQ7JmzxYWkXCk.ZX_zvPC0fUd2RQG6fz2tD6Ggt27hwGcSPr760RlVMLA&dib_tag=se&keywords=jetson+orin+nano&qid=1717099447&sprefix=jetson+orin+nano%2Caps%2C177&sr=8-3) this is the on board computer used for all the proccessing. I also soldered a connection from the switch to the barrel jack of the jetson.
* I also put an Arduino Uno on the platform. All the wire connections will be specified later.
* I created another small raised platform that is put on top of the main platform. I then glued a [ZED Mini Stereo Camera](https://store.stereolabs.com/products/zed-mini) ont he small platform.
* Both the camera and the Arduino should be connected to the Jetson via USB
### Wiring
* Both the servo and motor should be connected to 5v and ground on the Arduino, the third wire on the motor should be connected to pin 9, and the third pin on the servo should be connected to pin 10.
* Finally for the arduino to work upload the file `code/controller/arduino_ESC_servo_controller/arduino_ESC_servo_controller.ino` to the arduino.
* To control the motor speed and steering from the Jetson download the file `code/controller/Car_Controller.py` to the Jetson. This is a class which contrains functions for controlling motor speed and servo angle

## How it Works
The files actually responsible for autonomously driving are located in `code/driving/` to run the self driving download all the files in that directory and run `main.py`. The file contains all of the parameters for the program like the PID controller values and cone height. THis is a detailed proccess of how the car works.
### Convolutions Neural Network
First the car uses a CNN to find the cones in the image. The CNN is able to differentiate between green and orange cones. All the code for the CNN is located in the directory `code/cone_detection`. The file `cone_finder_AI.py` is the program that trians the CNN. There are a few programs I used to create the dataset that are located in `code/dataset`. I made the dataset by recording videos of me walking around with a camera. I then automatically pasted images of cones with another program into the video frames and used that as the training dataset. The file for the already trained CNN is in the `code/driving` directory.
### Driving algorithm
The entire driving algorithm is located in the `main.py` file.
* First it uses the CNN to detect all the cones. It then finds the closest green and orange cone using their mask areas. (In this algorithm the green cones should always be on the left and the orange cones should always be on the right)
* After that it finds the center point between the two cones it found.
* In case the camera can only see one cone (for example when the car is turning it doesn't see the inside cone but can still see the outside cone) it will calculate the distance to the outside cone using inverse projection and then use that distance to find a center point a certain distance from the outside cone. This distance can be specified wiht the `NON_VISIBLE_CONE_OFFSET` parameter in the `main.py` file.
* Finally it calculates the angle to the center point and uses two PI controllers (a PID controller but without the differential part) to calculate the necessary steering angle and speed to drive to that center point. The proportional and integral gains for both the PI controllers can be adjusted in `main.py`

## Demo Videos
To see videos of this system in action download the [demo videos](https://drive.google.com/file/d/1EFXXdTS90mtLiUYapqNxHuzPokW_0hlM/view?usp=drive_link)
