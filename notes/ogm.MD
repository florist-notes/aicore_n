
## Occupancy Grid Map (OGM) NOTES :

An occupancy grid map (OGM) can be created using a monocular camera and semantic segmentation using deep learning. Occupancy Grid Maps: Occupancy Grid Map → Point Cloud → Voxel Map (3D). Each cell is either occupied or a free space.

Dynamic Occupancy Grid Mapping (DOGMs) is a technique that generates a local map of the environment that contains both static and dynamic information. These maps are usually generated using lidar measurements.

Ideal Steps:

+ Use semantic image segmentation to estimate free space.
+ Create an occupancy grid using the free space estimate.
+ Visualize the occupancy grid on a bird's-eye plot.
+ Create a vehicle costmap using the occupancy grid.
+ Check whether locations in the world are occupied or free.

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
    occupancy_grid_mapping({l_{t-1,i}},x_t,z_t):

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

Path Planning in [ROS 2](https://wiki.ros.org/ROS/Tutorials): In Autonomy, Localization, Mapping and then Path Planning.

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