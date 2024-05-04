# Robotics : Develop a 2D Occupancy Grid Map of a Room using Overhead Cameras [INT 2]
<p class="ex1" align="justify" style="padding: 15px 15px 15px 15px">
The primary objective of this project is to develop a 2D occupancy grid map of a room using overhead
cameras, similar to the map created by a ROS2-based SLAM algorithm typically used by autonomous
mobile robots (AMRs).

Description:
+ In the initial phase, you will need to equip a room or area with four overhead RGB cameras
arranged in a 2x2 pattern, ensuring some overlap in their fields of view. The room should contain
static objects such as chairs, tables, stools, and boxes. Using the images from these cameras, and
by stitching the views together, you will create a 2D occupancy grid map of the room. The goal is
to demonstrate that this map can be effectively used by AMRs for path-planning and navigation.

+ In the second phase, the room environment will become dynamic, meaning objects like tables and
chairs can be moved around. The 2D occupancy grid map should be able to dynamically update
itself to reflect these changes, providing a new, accurate map for AMRs to navigate. Additionally,
you will add semantic labels to the map (e.g., "table", "chair", "other AMR") to provide further
context for the AMRs. This will likely require the implementation of simple object detection.
</p>

Outcomes:

Initially, you can model and simulate the room environment, objects, and overhead cameras in the
Gazebo simulator. Gazebo will also support the addition of an AMR equipped with an on-board camera
or LiDAR for SLAM map generation and navigation using the ROS2 navigation stack. This will allow for a
comparison between the map generated from the overhead cameras and the map created by the AMR
using SLAM.

<hr />

Resources: [[ VIDEO : ]] [Occupancy Grid Maps (Cyrill Stachniss)](https://youtu.be/v-Rm9TUG9LA?si=9emLvh3xzSyXKKNx), [UMich - Occupancy Grid Mapping](https://youtu.be/1f_m5aJFIj4?si=mnTasA3epEQcLFzv), [ROS2 Occupancy Grid Node for Nav2](https://youtu.be/suqhnzIyq7w?si=UthlawxajggnYC4v), [Nav 2 in ROS 2 for autonomous Navigation using SLAM for Indoor Mobile Robots](https://youtu.be/GSuqO0p2mIk?si=lFT4TSL3CrzO9pc_), [2D grid mapping and navigation with ORB-SLAM 2- indoor](https://youtu.be/HoE22wMhuKA?si=yenfgj4CrC-nV7EX), [Occupancy Grid Mapping with Known Poses](https://youtu.be/x_Ah685BFEQ?si=FXeitP2XzmE08WBz), [Occupancy Grid Mapping Simulation Using Sensor Laser In ROS](https://youtu.be/v3Y2PJcUTKE?si=-CCmY7JLZ-FUvBhk), [Robotic Mapping - Occupancy Grid Mapping](https://youtu.be/QFJehL9-pNo?si=grohiew6gVKJgA1j), [CH10 SLAM for Robotics - Occupancy Grid Mapping](https://youtu.be/VcFsY4lY_cI?si=Zp3UC4Incuv5OpVV), [ROG-Map](https://youtu.be/eDkwGXCea7w?si=JCnEQJXPFHXN8Soj), [Matlab - Lab2: Occupancy grid map](https://youtu.be/tWeLWoHHC5Q?si=jIlZlU_NATX602Kq), [2D grid mapping and navigation with ORB-SLAM 2- KITTI Dataset](https://youtu.be/FCd6p25131I?si=T4gI3yVXVH9SWCwZ), [Mobile robotics - C8: Mapping and occupancy grid](https://youtu.be/rcEFRrgmScw?si=Z4-AMVW89l5_xg4L), [Mapping for Mobile Robots and UGV](https://youtu.be/OCoVCYnCkeI?si=0L5EpcHGIcHeFudd), [NanoSLAM](https://youtu.be/XUSVLHJ87J0?si=uSyJ9vnIrPvKLJ48), [Presentation: Occupancy Grid Map Estimation Based on Visual SLAM and Image Segmentation](https://youtu.be/1i5CfddAzHk?si=hqYoXGzTQTrGRVrz), [MCMC Occupancy Grid Mapping with a Data-Driven Patch Prior](https://youtu.be/x0_8nix1HKA?si=FC0GkbGeyn7cAAPf).
 
 [[ READ : ]] [wiki-Occupancy Grid Mapping](https://en.wikipedia.org/wiki/Occupancy_grid_mapping), [CMU - Occupancy Maps](https://www.cs.cmu.edu/~16831-f14/notes/F14/16831_lecture06_agiri_dmcconac_kumarsha_nbhakta.pdf), Matlab - [occupancyMap](https://in.mathworks.com/help/nav/ref/occupancymap.html) & [Occupancy Grids](https://www.mathworks.com/help/robotics/ug/occupancy-grids.html), [CMU - Learning Occupancy Grid Maps With Forward Sensor Models
](https://faculty.iiit.ac.in/~mkrishna/ThrunOccGrid.pdf), [Occupancy grid mapping: An empirical evaluation](https://ieeexplore.ieee.org/document/4433772), [Occupancy grid for static environment perception in series automotive applications](https://www.sciencedirect.com/science/article/pii/S2405896319303945), [Grid Maps and Mapping With Known Poses](http://ais.informatik.uni-freiburg.de/teaching/ss16/robotics/slides/12-occupancy-mapping.pdf), [VoxFormer](https://github.com/NVlabs/VoxFormer), [ETH-Z ASL : wavemap](https://github.com/ethz-asl/wavemap), [dynamic-occupancy-grid-map](https://github.com/TheCodez/dynamic-occupancy-grid-map), [SSCBench](https://github.com/ai4ce/SSCBench), [Occ4cast](https://github.com/ai4ce/Occ4cast).

<hr />

## NOTES :

Occupancy Grid Maps: Occupancy Grid Map → Point Cloud → Voxel Map (3D). Each cell is either occupied or a free space.

Estimating a map from data: Given a sensor data z_{1:t} and the poses x_{1:t} of the sensor, estimate the map :

$$ p( m | z_{1:t}, x_{1:t}) = \prod_i (m_i| | z_{1:t}, x_{1:t} ) $$

Update rules to estimate the joint probability distribution:

[Baye's rule / Binary Bayes filter (for a static state)]:

$$ p(m_i | z_{1:t}, x_{1:t}) =  \dfrac{p(z_t | m_i, z_{1:t-1}, x_{1:t}) p(m_i | z_{1:t-1}, x_{1:t}) }{p(z_t | z_{1:t-1}, x_{1:t})}$$  

[Markov's rule]:

$$ p(m_i | z_{1:t}, x_{1:t}) =  \dfrac{p(z_t | m_i, x_t) p(m_i | z_{1:t-1}, x_{1:t-1}) }{p(z_t | z_{1:t-1}, x_{1:t})}$$  


If we apply Baye's rule in,

$$ p(z_t | m_i, x_t) = \dfrac{p(m_i | z_t, x_t) p(z_t, x_t) }{p(m_i, x_t)}$$

Then the equation for probability of cell being occcupied becomes,

$$ p(m_i | z_{1:t}, x_{1:t}) =  \dfrac{p(m_i | z_t, x_t) p(z_t, x_t) p(m_i | z_{1:t-1}, x_{1:t-1}) }{ p(m_i, x_t) p(z_t | z_{1:t-1}, x_{1:t})}$$  

Do exactly the same for the opposite event:

$$ p(¬m_i | z_{1:t}, x_{1:t}) =  \dfrac{p(¬m_i | z_t, x_t) p(z_t, x_t) p(¬m_i | z_{1:t-1}, x_{1:t-1}) }{ p(¬m_i, x_t) p(z_t | z_{1:t-1}, x_{1:t})}$$  

By computing the ratio of both probabilities, we obtain:

$$ \dfrac{p(m_i | z_{1:t}, x_{1:t})}{p(¬m_i | z_{1:t}, x_{1:t})} =  \dfrac{ \dfrac{p(m_i | z_t, x_t) p(z_t, x_t) p(m_i | z_{1:t-1}, x_{1:t-1}) }{ p(m_i, x_t) p(z_t | z_{1:t-1}, x_{1:t})} }{ \dfrac{p(¬m_i | z_t, x_t) p(z_t, x_t) p(¬m_i | z_{1:t-1}, x_{1:t-1}) }{ p(¬m_i, x_t) p(z_t | z_{1:t-1}, x_{1:t})} } $$


$$   \dfrac{p(m_i | z_{1:t}, x_{1:t})}{p(¬m_i | z_{1:t}, x_{1:t})} =  \dfrac{ \dfrac{p(m_i | z_t, x_t) p(m_i | z_{1:t-1}, x_{1:t-1}) }{ p(m_i, x_t)} }{ \dfrac{p(¬m_i | z_t, x_t) p(¬m_i | z_{1:t-1}, x_{1:t-1}) }{ p(¬m_i, x_t) } } $$


$$  \dfrac{p(m_i | z_{1:t}, x_{1:t})}{p(¬m_i | z_{1:t}, x_{1:t})} =  \dfrac{p(m_i | z_t, x_t) p(m_i | z_{1:t-1}, x_{1:t-1})  p(-m_i)}{p(¬m_i | z_t, x_t) p(¬m_i | z_{1:t-1}, x_{1:t-1}) p(m_i)} $$

$$  \dfrac{p(m_i | z_{1:t}, x_{1:t})}{p(¬m_i | z_{1:t}, x_{1:t})} =  \dfrac{p(m_i | z_t, x_t) * p(m_i | z_{1:t-1}, x_{1:t-1}) * (1-p(m_i))}{(1 - p(m_i | z_t, x_t)) * (1-p(m_i | z_{1:t-1}, x_{1:t-1})) * p(m_i)} $$

The first term,

$$  \dfrac{p(m_i | z_t, x_t)}{(1 - p(m_i | z_t, x_t))} $$

uses z_t, the current observation. The second term uses recursive data and third term is a prior information.

From ratio to probability, we use odds ratio (event divided by opposite event).

$$ Odds(x) =  \dfrac{p(x)}{1-p(x)} $$

$$ p(x) =  \dfrac{1}{1+\dfrac{1}{Odds(x)}} $$

#### From Ratio to Probability:

Using,

$$ p(x) = [1+Odds(x)^{-1}]^{-1} $$ 

gives us,

$$  p(m_i | z_{1:t}, x_{1:t}) = [1+\dfrac{(1 - p(m_i | z_t, x_t)) * (1-p(m_i | z_{1:t-1}, x_{1:t-1})) * p(m_i)}{p(m_i | z_t, x_t) * p(m_i | z_{1:t-1}, x_{1:t-1}) * (1-p(m_i))}]^{-1} $$

For reasons of efficiency, one performs the calculations in the log odds notation.

$$  l(m_i | z_{1:t}, x_{1:t}) = log(\dfrac{p(m_i | z_{1:t}, x_{1:t})}{1 - p(m_i | z_{1:t}, x_{1:t})}) $$

The product turns into a sum,

$$  l(m_i | z_{1:t}, x_{1:t}) = l(m_i | z_{t}, x_{t}) + l(m_i | z_{1:t-1}, x_{1:t-1}) - l(m_i)$$

$$  l(m_i | z_{1:t}, x_{1:t}) = inverse\ sensor\ model+recursive\ term + prior $$

or, in short,

$$ l_{t,i} = inv\ sensor\ model(m_i+x_t+z_t) + l_{t-1,i} - l_0 $$

## Occupancy Mapping Algorithm:

```

    occupany_grid_mapping({l_{t-1,i}},x_t,z_t):

    for all cells m_i do:
        if m_i in perceptual field of z_t then:
            l_{t,i} = l_{t-1,i + inv_sensor_model(m_i, x_t, z_t)-l_0}
        else
            l_{t,i} = l_{t-1,i}
        endif
    endfor
    return l_{t,i}

```

Highly, efficient as we only have to compute sums in third step. Moravec and Elfes proposed occupany grid mapping in the mid 80's. Developed for noisy sonar sensors and known as "mapping with known poses". Maximum Likelihood map is obtained by rounding the probability for each cell to 0 or 1.

Which cells to update for a single laser beam ?  [Bresenham's line algorithm](https://en.wikipedia.org/wiki/Bresenham%27s_line_algorithm)

## ROS2 Occupancy Grid Node for Nav2:

Path Planning: In Autonomy, Localization, Mapping and then Path Planning.

Laser Publisher [ [laser_publisher.cpp](https://github.com/noshluk2/ros2_learners/blob/main/nodes/src/laser_publisher.cpp) ]

### Creating a Package:

C++:
```bash

    $ ros2 pkg create --build-type ament_cmake <package_name>

```
Python:
```bash

    $ ros2 pkg create --build-type ament_python <package_name>

```

### Getting Information

Message:
```bash

    $ ros2 interface show <msg/msg/type>
    $ ros2 interface show nav_msgs/msg/OccupancyGrid

```
Topic:
```bash

    $ ros2 topic info <topic name>

```

### Locations

For generic header files:
```bash

    $ cd /opt/ros/humble/include/
    $ cd /opt/ros/humble/include/nav_msgs/nav_msgs/msg/

```
We can use `#include "nav_msgs/nav_msgs/msg/occupancy_grid.hpp"` and more information on ros2 nav stack is [here](https://github.com/florist-notes/aicore_s/blob/main/notes/ros.MD).
