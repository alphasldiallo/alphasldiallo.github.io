---
title: ANN Gesture recognition with Nordic Thingy:52
layout: post
description: Gesture recognition using a set of 3-axis IMU sensors
img: gesture.jpg
---

Gesture recognition is a fascinating area widely explored in the literature. Its goal is to infer users' actions based on their movement and it is mainly used in Human Computer Interaction. The general structure of gesture recognition consists of three main steps:
- Data acquisition. In the first step, we list all the mecanisms used for collecting movements data and saving in a usable shape.
- Pattern recognition
- Interpretation

In this article, we will build a simple model that collects data from an Inertial Measurement Unit (IMU) sensors and identifies known gestures. 
The rest of the article is specially dedicated to those of you who are born in the 90's and/or have a culture of arcade gaming. If it applies to you, then you probably heard about Street Fighter II. If words like _SNES_, _Street Fighter_ or _arcade_ sound unfamiliar to you, please read at least [this article](https://en.wikipedia.org/wiki/Street_Fighter_II).

## Data acquisition
To collect data, we can use two approaches, one with a 3-Axis accelerometer and gyroscope IMU - MPU6050  and a Nordic Thingy:52 which is built on a Bluetooth 5 ready nRF52832 SoC and embeds a 9-axis IMU (Accelerometer, compass and Gyroscope).

### Collecting data with an MPU6050
If you choose to use this approach, make sure to order the correct IMU sensor. You can use this link as reference: [MPU-6050 on Amazon](https://www.amazon.com/HiLetgo-MPU-6050-Accelerometer-Gyroscope-Converter/dp/B00LP25V1A/ref=sr_1_1?dchild=1&keywords=MPU-6050&qid=1625759114&sr=8-1).
As you can see, it is quite cheap so do not hesitate to order several items just to be on the safe side. In this article, we will collect data from the IMU sensor through a Raspberry Pi. Make sure to have a soldering iron and some wires to connect to the GPIO port on your Raspberry as presented below.
<img src="../assets/img/RPI_mpu6050.jpg" style="height=500px; text-align:'center'" />

Once the hardware part done, we can focus on the software side. To manipulate the data, we will create a simple script to make a link between the sensor and the raspberry, collect the data and store in a file.

```python
from board import SDA, SCL
from imu_mpu6050 import MPU6050
import busio
import time
import csv

i2c = busio.I2C(SCL, SDA)
IMU=MPU6050(i2c)
print(IMU.whoami)

data = []
SAMPLES = 500
  
while(len(data)<SAMPLES):
    acc =IMU.get_accel_data(g=True)
    gyro = IMU.get_gyro_data()
    tmp_dict = {"acc_x":acc["x"], "acc_y":acc["y"], "acc_z":acc["z"], "gyro_x":gyro["x"], "gyro_y":gyro["y"], "gyro_z":gyro["z"] }
    data.append(tmp_dict)
    print(str(len(data)/SAMPLES*100)+"%")
    time.sleep(0.1)
    
    
with open("idle.csv", mode="w") as f:
    w = csv.DictWriter(f, data[0].keys())
    w.writeheader()
    for i in data:
        w.writerow(i)

```

The data collected will be used to train our Neural Network. By default, we will collect 10 samples per second or each action and save in a `CSV` file that should look as follows:
```csv
acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z
0.414794921875,-0.2760009765625,0.800048828125,-2.450381679389313,15.49618320610687,0.2366412213740458
0.4761962890625,-0.241943359375,0.863037109375,-0.08396946564885496,25.427480916030536,-5.419847328244275
0.5587158203125,-0.453125,0.8902587890625,9.969465648854962,18.244274809160306,37.36641221374046
0.731201171875,-0.578125,0.932373046875,22.557251908396946,45.74809160305343,85.35114503816794
```

The action that we will try to detect are:
- leftright: action of moving a hand (while holding the sensor) from left to right repeatedly.
- updown: same as the previous but we should move from up to down repeatedly.
- Hadoken: coolest part of this project. We just have to make (a lot of) hadokens.
- Idle: We will just collect data on a flat surface. In our example, we will consider unknown actions as idle.

Collecting data is annoying, that why I am sharing with you all the data you will need: [leftright](https://raw.githubusercontent.com/alphasldiallo/IMU_data/main/leftright.csv), [updown](https://raw.githubusercontent.com/alphasldiallo/IMU_data/main/updown.csv), [hadoken](https://raw.githubusercontent.com/alphasldiallo/IMU_data/main/hadoken.csv), [idle](https://raw.githubusercontent.com/alphasldiallo/IMU_data/main/idle.csv).

### Collecting data using a Thingy:52

To collect data from the Thingy:52, we can use this nice open source library available here: https://github.com/hbldh/bleak. To get the measurements, we will first make a scan in order to get our device's UUID: 
```python
import asyncio
from bleak import BleakScanner


async def run():
    devices = await BleakScanner.discover()
    for d in devices:
        if "Thingy" in d.name:
            print(d)

loop = asyncio.get_event_loop()
loop.run_until_complete(run())
```
Hint: Make sure to install the dependencies before running the scripts.

If everything goes as expected, we should get a list containing all the Thingy in the surroundings. In my case, I only have one with the following UUID: `3F706B28-2A00-4C26-9C1F-AF1C2F01299B`. Copy your UUID and store it somewhere, we will need it later.


## Pattern recognition

## Interpretation
