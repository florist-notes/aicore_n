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