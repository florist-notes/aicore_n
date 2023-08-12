# SLAM

Ok, let's code a visual SLAM from scratch in c++. I am learning from my notes @github/[autonomy_perception](https://github.com/florist-notes/aicore_s/blob/main/notes/perc.MD),  @github/[robotics-slam](https://github.com/florist-notes/aicore_s/blob/main/notes/robotics_ee_hardware/roboticsmech.MD) and companion guide '[Introduction to Visual SLAM](https://link.springer.com/book/10.1007/978-981-16-4939-4)' : [ [code](https://github.com/gaoxiang12/slambook2/tree/master) : [booken](https://github.com/gaoxiang12/slambook-en) ]. Example: [ORB SLAM 3](https://github.com/UZ-SLAMLab/ORB_SLAM3), [FastSLAM](https://link.springer.com/book/10.1007/978-3-540-46402-0).

## Introduction to SLAM 

Monocular vs Stereo : The trajectory and map obtained from monocular SLAM estimation will differ from the actual rajectory and map with an unknown factor, which is called the scale. Since monocular SLAM can not determine this real scale purely based on images, this is also called the scale ambiguity. In monocular SLAM, depth can only be calculated with translational movement, and the real scale cannot be determined. These two things could cause significant trouble when applying monocular SLAM into real-world applications. The fundamental cause is that depth can not be determined from a single image. So, in order to obtain real-scaled depth, we start to use stereo and RGB-D cameras. A stereo camera consists of two synchronized monocular cameras, displaced with a known distance, namely the baseline. Because the physical distance of the baseline is
known, we can calculate each pixel’s 3D position in a very similar way to our human eyes.

The disadvantage of stereo cameras or multi-camera systems is that the configuration and calibration process is complicated. Their depth range and accuracy are limited by baseline length and camera resolution. Moreover, stereo matching and disparity
calculation also consume many computational resources and usually require GPU or FPGA to accelerate to generate real-time depth maps. Therefore, in most state-ofthe-art algorithms, the computational cost is still one of the major problems of stereo
cameras.

Depth camera (also known as RGB-D camera), similar to laser scanners, RGB-D cameras adopt infrared structure of light or Time-of-Flight (ToF) principles and measure the distance between objects and the camera by actively emitting light to the object and
receive the returned light. Most of the RGB-D cameras still suffer from issues including narrow measurement range, noisy data, small field of view, susceptibility to sunlight interference, and unable to measure transparent material. For SLAM purposes, RGB-D cameras are mainly used in indoor environments and are not suitable
for outdoor applications.
### 3D Rigid Body Motion :

Overview of a SLAM system. It describes each module of a
typical SLAM system and explains what to do and how to do it. 
### Lie Group and Lie Algebra :

Lie group and Lie algebra. Lie group manipulation with Sophus.

### Cameras and Images :

Pinhole camera model and image expression in computer. OpenCV to retrieve the camera’s intrinsic and extrinsic parameters and
generate a point cloud using the depth information through Point Cloud Library (PCL) .
### Nonlinear Optimization : 

 Nonlinear optimization, including state estimation, least squares, and gradient descent methods, e.g., Gauss-Newton and Levenburg-Marquardt method. Curve-fitting problem using the Ceres and g2o library.


## SLAM Technologies 

### Visual Odometry :

Feature-based visual odometry, which is currently the mainstream
in VO. Feature extraction and matching, epipolar geometry calculation, Perspective-n-Point (PnP) algorithm, Iterative Closest Point (ICP) algorithm, and Bundle Adjustment (BA), etc. OpenCV functions or constructing our own optimization problem in Ceres and g2o.

 Direct (or intensity-based) method for VO ; Optical flow principle and the direct method. Writing single layer and multi-layer optical flow and direct method to implement a two-view VO.

### Filters and Optimization Approaches :

Bundle Adjustment in detail and show the relationship between its sparse structure and the corresponding graph model. Ceres and g2o separately to solve the same BA problem.

Pose graph in the backend optimization. Pose graph is a more
compact representation for BA, which converts all map points into constraints between keyframes. g2o to optimize a pose graph.

### Loop Closure :

Loop closure detection, mainly Bag-of-Word (BoW) based method.
Use DBoW3 to train a dictionary from images and detect loops in
videos.

### Dense Reconstruction :

 Estimate the depth of pixels in monocular SLAM (and show why they are unreliable). Compared with monocular depth estimation, building a dense map with RGB-D cameras is much easier. Write programs for epipolar line search and patch matching to estimate depth from monocular images and then build a point cloud map and octagonal treemap from RGB-D data.

### SLAM Discussion:


 Open-Source implementations : MonoSLAM, PTAM, ORB-SLAM Series, LSD-SLAM, SVO, RTAB-MAP ; IMU-Integrated VSLAM, Semantic SLAM.