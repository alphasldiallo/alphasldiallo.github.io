---
title: "[Paper review] CrowdInside: Automatic Construction of Indoor Floor plans"
layout: post
img: crowdplan.jpg
tags:
- review
- indoor
- plan
- reconstruction
- IMU
---

*M. Alzantot and M. Youssef, “Crowdinside: automatic construction of indoor floorplans”, in Proceedings of the 20th International Conference on Advances in Geographic Information Sys- tems, pp. 99–108, 2012.*

Context
----

The ubiquity of mobile phones has opened the door to researchers and industries in implementing indoor location solutions. The reason justifying this interest is the poor quality of the GPS signal inside buildings. This has led many researchers to turn to mobile phone sensors such as sound and electromagnetic waves, ultra-sensitive GNSS sensors, magnetic or infrared systems, etc.$$^1$$. All these systems, although offering many advantages, come up against the essential question of building indoor maps. Indeed, according to Shen et al., few indoor location system had been deployed at scale mainly because of the difficulty of finding suitable localization maps$$^2$$. The reason behind this is the fact that each building is unique and unlike GPS, it is not possible to know in advance the plans of a building from the outside. This task remains an arduous step in the quest for a system as efficient as GPS that can be used universally.


Summary
----

The goal of the system presented in this paper is to automatically construct overall floorplan shapes, identify points of interest (such as elevators, stairs or escalators) and finally get the room/corridors shapes. In this paper, Alzantot and Youssef present an innovative approach which consist in collecting data from smartphone sensors (accelerometer, gyroscope, magne- tometer, Wi-Fi) to detect a unique footprint specific to each room in a building. Based on the data collected by the sensors, they also manage to detect points of interest and entry points with a high accuracy. To associate a user to a specific point of interest, the authors use an approach which consist in matching data continuously collected by the user to known patterns of point of interests. By analyzing the collected traces, the authors also draw the shape of each room of a building. The shapes of corridors, rooms and halls are recovered by breaking down continuous motion traces into segments separated by significant changes in the direction (with a threshold set to 45à). To obtain the entire floorplan, the authors propose a clustering approach of the collected traces. This approach consists in building point clouds of the collected traces and in clustering them using α-shapes which generate a set of boundary points such that together they form a smaller surface area representing the floorplan.


Contribution
----

The main contribution made by the authors in the area of automatic construction of building plans is to provide a solution that is easy to implement with widely available sensors. By using the notion of crowdsensing, the authors propose an approach that reduces the time required to fully map a floorplan. Indeed, the concept of crowdsensing consists in collecting data extracted from sensors of multiple users in order to measure, map, analyze, estimate or infer processes of common interest. In this paper, the authors use this approach to construct floorplans by collecting data from multiple sensors such as accelerometers, gyroscopes, magnetometers and the surrounding Wifi signals. This novel approach is a step towards finding a ubiquitous system for indoor location such as GPS used outdoor. Using sensors from smartphones to implement dead-reckoning breaks down the barriers of deploying physical infrastructures to build maps and track people/assets indoor. By using anchor points to reset the error in dead-reckoning and by adding step-counting rather than integrating the acceleration, the authors extend some related work and increase substantially the accuracy in building floorplans.


Limitation
----

The design of the solution is based on several assumptions that can be used to spot its weak- nesses. The main limitations are as follow:
1. **Large amount of data required.** The main requirement to implement the solution presented in the paper is to have enough anchor points such as locations with GPS re- ception (entry doors) or points with a special inertial data signature (stairs, escalators or elevators) to calibrate the system. We also need to collect sufficient amount of traces near these anchor points and collect distinctive WiFi signatures in different rooms.
2. **Characteristics of testbeds.** From a theoretical perspective, the paper presents a great approach in building automatic floorplans. But, by taking into consideration the requirements for this system to work ubiquitously, we realize that the authors only used two testbeds. A shopping mall to evaluate the accuracy of trace generation and anchor- based error resetting (by using widely available point of interests such as elevators), and a university campus to evaluate the floor plan construction. The testbeds chosen by the authors match with their requirements. However, as shown by Goa et al. $$^3$$, buildings may not always have the same profile such as specific infrastructures with a particular inertial signature. For instance, Goa et al. implement the solution presented in the paper in a storey containing 3 points of interests and no GPS reception. Even by adding several artificial anchor points with known GPS coordinates, they obtain a Root Mean Square Error (RMSE) of 6.26 meter with a maximum error of 8.92 meter on landmark positions and a RSME of 7.36 meter and maximum error of 9.84 meter in intersections$$^3$$. These results show a bad accuracy of the system in an uncontrolled indoor environment.
3. **Reliability of GPS.** To detect anchor points such as entry doors, the authors use an approach based on GPS signal with a low duty cycle. In the paper, the entry points are detected when there’s a GPS signal loss. As shown in a paper written by Nirjon et al.$$^4$$, indoor, GPS signal strength is about 10-100 times weaker. However, in some cases, the GPS signal is not lost but due to multipath effect and weaker signal it can give inaccurate results. To increase the accuracy of detecting anchor points based on GPS signal, the authors rely on collecting a large number of samples: up to 1,200 samples with a duty cycle of 6 minutes. This approach limits the efficiency of the system, which gives better results only when a large number of GPS entries is collected at the same anchor location.
4. **Exclusion of poorly visited areas.** As shown in the paper, after collecting the data and generating the shape of the building (using α-shapes), we can observe that the approach used by the authors excludes non visited rooms.
5. **False walking problem**. By using a pedometer approach, the authors reduced the accumulation errors of the data provided by the accelerometer. However, as mentioned in a paper by Gayle et al.$$^5$$, the pedometer gives an error of around 11%.

To extend this work and improve the performance of the system presented in the paper, one could explore the possibilities offered by proximity detection sensors such as Bluetooth to reset the error in dead-reckoning based on data collected by other users in the vicinity. Another recommendation could consist in using periodicity and similarity features to solve the false walking problems as shown by Pham et al.$$^6$$.


## References
1. R. Mautz, Indoor positioning technologies. ETH Zurich, Department of Civil, Environmen- tal and Geomatic Engineering, Institute of Geodesy and Photogrammetry, 2012.
2. G. Shen, Z. Chen, P. Zhang, T. Moscibroda, and Y. Zhang, “Walkie-markie: Indoor path- way mapping made easy,” in Presented as part of the 10th USENIX Symposium on Networked Systems Design and Implementation (NSDI 13), pp. 85–98, 2013.
3. R. Gao, M. Zhao, T. Ye, F. Ye, Y. Wang, K. Bian, T. Wang, and X. Li, “Jigsaw: In- door floor plan reconstruction via mobile crowdsensing,” in Proceedings of the 20th annual international conference on Mobile computing and networking, pp. 249–260, 2014.
4. S. Nirjon, J. Liu, G. DeJean, B. Priyantha, Y. Jin, and T. Hart, “COIN-GPS: indoor localization from direct GPS receiving,” in Proceedings of the 12th annual international conference on Mobile systems, applications, and services, pp. 301–314, 2014.
5. R. Gayle, H. J. Montoye, and J. Philpot, “Accuracy of pedometers for measuring distance walked,” Research Quarterly. American Alliance for Health, Physical Education and Recreation, vol. 48, no. 3, pp. 632–636, 1977.
6. V. T. Pham, D. A. Nguyen, N. D. Dang, H. H. Pham, V. A. Tran, K. Sandrasegaran, D.-T. Tran, et al., “Highly accurate step counting at various walking states using low-cost inertial measurement unit support indoor positioning system,” Sensors, vol. 18, no. 10, p. 3186, 2018.
