# Interview Questions |  SLAM

### General
What are the different ways to represent rotations in 3D space?

- Discuss the differences between the SO(3) matrix, Quaternion, Axis-angle, and Euler angle representations.
- What problems does gimbal lock pose in the expression of 3D rotations?
- What mathematical constraints are applicable to SO(3) matrices?

Describe the structure of the SE(3) matrix.
- What is the significance of the bottom row ([0,0,0,1]) in the SE(3) matrix?

What sensors are suitable for SLAM (Simultaneous Localization and Mapping)?
- Compare tightly-coupled fusion and loosely-coupled fusion in this context.

Why is non-linear optimization used in SLAM?
- Where do we encounter non-linearity in Visual-SLAM?
- Where is non-linearity found in LiDAR SLAM?

What optimization methods are applicable for non-linear optimization in SLAM?
- Compare gradient descent, Newton-Raphson, Gauss-Newton, and Levenberg-Marquardt methods.
- What is the trust-region method?

What is loop closure and how is it achieved in SLAM?

Define and differentiate the motion model and observation model in SLAM.

What is RANSAC?

Explain the concept of a robust kernel (or M-estimator).

Discuss the Kalman filter and particle filter.
- Highlight the differences between the Kalman filter (KF) and the Extended Kalman filter (EKF).

Contrast filter-based SLAM with graph-based SLAM.

Define the information matrix and covariance matrix in the context of SLAM.

What is the Schur complement?

Compare LU, Cholesky, QR, SVD, and Eigenvalue decomposition.
- Which methods are commonly used in SLAM and why?

Why is least squares optimization favored?
- Explain how Maximum-a-posteriori (MAP) and Maximum Likelihood Estimation (MLE) are applied in SLAM.

What representations are used to describe a map or structure in SLAM?
- Which map representation would you choose for path planning and why?
- Distinguish between sparse mapping and dense mapping.

Explain the concepts of Lie groups and Lie algebra.
- What are the Exp/Log maps?

How can multiple maps be merged into a single map in SLAM?

What is Inverse Depth Parameterization?

Describe pose graph optimization in SLAM.

Define drift in SLAM.
- What is scale drift?

How can computational costs be reduced in SLAM?
- What is keyframe-based optimization?
- Why is a Look-Up Table (LUT) considered an effective strategy?

What is relocalization in SLAM?
- How does relocalization differ from loop closure detection?

What does marginalization entail in the context of SLAM?

Explain the concept of IMU pre-integration in SLAM.

### LiDAR SLAM

Explain how the Iterative Closest Point (ICP) algorithm functions.
- Which derivative work of the ICP algorithm do you prefer and why?

Discuss the Point-to-Point and Point-to-Plane metrics in the context of the ICP algorithm.

If an ICP algorithm fails to register two sets of point clouds, what could be the possible reasons?

Explain the concept of a K-D tree.
- How is the K-D tree utilized in processing LiDAR point clouds?

Describe the Octree data structure.

In which scenarios would you use a K-D tree over an Octree for LiDAR point cloud processing, and vice versa?

What is point cloud downsampling and why is it used?
- Describe the voxelization process.
- What are the consequences of excessive downsampling?

How is ground segmentation performed in point clouds?
- What is the mathematical formulation of a 3D plane?

What is a passthrough filter?

What preprocessing techniques are available for removing outliers from LiDAR point clouds?
- How does the Statistical Outlier Removal (SOR) filter work?

Why is initial alignment crucial in ICP?

Besides x, y, and z coordinates, what additional information can be embedded in a point cloud?

What advantages are gained by integrating LiDAR with an IMU?

How is loop detection performed using LiDAR point clouds?

If a loop is detected, how should loop closure optimization be carried out?
- How does loop closure in LiDAR SLAM differ from the bundle-adjustment technique in Visual SLAM?

Why does z-drift often occur in LiDAR SLAM optimization using the ground plane?

What is LiDAR de-skewing?

What challenges arise in LiDAR SLAM when there are moving objects in the vicinity?

What is the Multi-path problem in LiDAR?

In what types of environments does LiDAR typically underperform?

What are the different types of LiDAR?

What are various methods for combining data from a camera and LiDAR?

Contrast a point cloud, mesh, and surfel.

What is a Fast Point Feature Histogram (FPFH) descriptor?

What methods are available for detecting changes in a point cloud

### Visual SLAM

Explain the process of image projection.
- What are intrinsic and extrinsic matrices?
- Which formula is used to estimate depth from a single-view image?

What does camera calibration entail and what information is gained from it?
- Provide the formulas for the K matrix and the Distortion coefficient.

Describe the characteristics of Monocular, Stereo, and RGB-D SLAM, along with their respective advantages and disadvantages.
- How is the depth map generated in RGB-D?
- How is the depth map generated in Stereo?
- Explain the concept of stereo disparity.
- Is there any way to restore scale in monocular VSLAM?

Explain bundle adjustment.
- What are the differences between local and global bundle adjustments?

What are the Essential and Fundamental matrices?
- Write down the formulas for the Essential and Fundamental matrices.
- How many degrees of freedom do the Essential and Fundamental matrices have?
- What is the 5/7/8-point algorithm?

What is the Homography matrix?

Describe the camera models you are familiar with.

Explain the process of local feature matching.
- What are the differences between a keypoint and a descriptor?
- How does a feature in deep learning differ from a feature in SLAM?
- What strategies are effective for accurate feature matching?

Explain how local feature tracking is performed.
- What can serve as a motion model?
- What methods can be used for optical flow?
- Describe template tracking.
- How does optical flow differ from direct tracking?

Explain the features and differences between PTAM, ORB-SLAM, and SVO.

What are the differences between Visual Odometry, Visual-SLAM, and Structure-from-Motion (SfM)?

Why isn’t SIFT used in real-time VSLAM?
- What are some alternatives to SIFT?
- What are the benefits of using deep learning-based local feature detection?

What is reprojection error?
- What is photometric error?

What is the Perspective-n-Point (PnP) problem?
- How do you determine the camera’s pose when there is a 2D-3D correspondence?

What are the differences between Feature-based VSLAM and Direct VSLAM?

What methods are effective in reducing blur in an image?

What is a co-visibility graph?

How is loop closure detection performed?
- Describe the Bag-of-Visual-Words and VLADs.
- How is a Bag-of-Visual-Words created?
- Explain TF-IDF.

What distinguishes a floating-point descriptor from a binary descriptor?
- How can the distance between feature descriptors be calculated?

What defines a good local feature?
- What is meant by invariance?

How is image patch similarity determined?
- Compare SSD, SAD, and NCC.

Explain Direct Linear Transform (DLT).

Describe the Image Pyramid.

Outline the methods for line/edge extraction.

Explain Triangulation.

### System design

You have a mobile robot system with four cameras mounted on the front, back, and sides. Design a system that can use it for indoor mapping, localization, and path planning.

Design a kiosk robot that can be used in an amusement park.

Design a mapping/positioning system for parking in a garage (common in the US and Europe).

Design a mapping/localization system for a parking garage.

Design an augmented reality device for use on a moving Ferris wheel.

Design an augmented reality device for a crowded subway station.

Design a robust positioning system for an unmanned forklift to be used in a logistics facility. There are multiple forklifts moving around, and people are present.

Design a SLAM system for a mobile robot to be used in a factory. There is constant lighting, but the floor is coated and highly reflective, and the factory equipment is made of metal and has few visual features and is highly reflective.

You have 10 TB of real-world data. What development pipeline would you create to make SLAM work in as many environments as possible?

Design a crowdsourced, automated HD-Map creation system for autonomous driving.

### Coding interview questions (Live / implementation)
+ Implement an image stitcher to create a panorama image from multiple consecutive images.
+ Implement LiDAR SLAM using G-ICP based odometry. Loop closure should be implemented.
+ Implement FAST keypoint detector.
+ Implement an algorithm to find the camera pose, given 2D-3D correspondence data.
+ Implement the PROSAC framework.
+ Implement the ICP algorithm.
+ Implement a brute-force matcher given a set of 2 pairs of feature descriptors.
+ Implement a Vector / Matrix container. Basic operators should work (e.g. matrix-matrix addition, matrix-matrix multiplication, matrix-vector multiplication).
+ Implement the A* algorithm
+ Implement a fast matrix multiplication algorithm.
+ ([Live](https://youtu.be/VbVQJRKXxBA?si=dw3sWac4WBGLG4Oj)) Two Sum problem
+ ([Live](https://youtu.be/u202zgM7Fqk?si=xsqptyH1F6QajMWw)) Maximum subarray sum problem
+ ([Live](https://youtu.be/6p7mfKopfLY?si=dL8X7royA_DPTJ8I)) Product of array except self problem
+ ([Live](https://youtu.be/YkacnIOt2jM?si=27T-e5EhAycBXj5v)) Subarray sum equals k problem
+ ([Live](https://youtu.be/i1u2ChnUvZM?si=qN--8UGUDkCxu_1-)) Longest common sequence problem

### [Autonomous Systems Interview Preparations](https://gist.github.com/sunsided/b76260626cefb7512a9b3579f1c9660e) | [🌸]