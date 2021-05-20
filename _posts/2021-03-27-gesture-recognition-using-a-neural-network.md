---
title: ANN Gesture recognition with Nordic Thingy:52
layout: post
description: Gesture recognition using a set of 3-axis IMU sensors
---

Gesture recognition is a fascinating area widely explored in the literature. Its goal is to infer users' actions based on their movement and it is mainly used in Human Computer Interaction. The general structure of gesture recognition consists of three main steps:
- Data acquisition. In the first step, we list all the mecanisms used for collecting movements data and saving in a usable shape.
- Pattern recognition
- Interpretation

In this article, we will build a simple model that collects data from an Inertial Measurement Unit (IMU) sensors and identifies known gestures. 
The rest of the article is specially dedicated to those of you who are born in the 90's and/or have a culture of arcade gaming. If it applies to you, then you probably heard about Street Fighter II. If words like _SNES_, _Street Fighter_ or _arcade_ sound unfamiliar to you, please read at least [this article](https://en.wikipedia.org/wiki/Street_Fighter_II).

## Data acquisition
To collect data, I used two approaches, one with a 3-Axis accelerometer and gyroscope IMU - MPU6050  and a Nordic Thingy:52 which is built on a Bluetooth 5 ready nRF52832 SoC and embeds a 9-axis IMU (Accelerometer, compass and Gyroscope).

## Pattern recognition

## Interpretation
