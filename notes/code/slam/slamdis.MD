# SLAM Algorithms:

Visual SLAM (VSLAM) has been developing rapidly due to its advantages of low-cost sensors, the easy fusion of other sensors, and richer environmental information. Traditional visionbased SLAM research has made many achievements, but it may fail to achieve wished results in challenging environments. Deep learning has promoted the development of computer vision, and the combination of deep learning and SLAM has attracted more and more attention. Semantic information, as high-level environmental information, can enable robots to better understand the surrounding environment.


 Open source SLAM systems : MonoSLAM, PTAM, ORB-SLAM Series, LSD-SLAM, SVO, RTAB-MAP ; IMU-Integrated VSLAM, Semantic SLAM.

 <table width=100%>
  <tr>
  <th>SLAM algorithm name (Sensors∗) :</th>
  </tr>
  <tr>
  <td>

  [MonoSLAM](https://github.com/hanmekim/SceneLib2) (M)
  </td>
  <td>

  [PTAM](http://www.robots.ox.ac.uk/~gk/PTAM/) (M)
  </td>
  <td>

  [DTAM](https://github.com/anuranbaka/OpenDTAM) (R)
  </td>
  <td>

  [ORB-SLAM](http://webdiis.unizar.es/~raulmur/orbslam/) (M/S/R)
  </td>
  <td>

  [LSD-SLAM](http://vision.in.tum.de/research/vslam/lsdslam) (M)
  </td>
  <td>

  [RTAB-MAP](https://github.com/introlab/rtabmap) (S/R)
  </td>
  <td>

  [SVO](https://github.com/uzh-rpg/rpg_svo) (M)
  </td>
  </tr>
 
  <tr>
  
  <td>

  [RGBD-SLAM-V2](https://github.com/felixendres/rgbdslam_v2) (R)
  </td>
  <td>

  [ROVIO](https://github.com/ethz-asl/rovio) (M+IMU)
  </td>
  <td>

  [DVO](https://github.com/tum-vision/dvo_slam) (R)
  </td>
  <td>

  [DSO](https://github.com/JakobEngel/dso) (M)
  </td>
  <td>

  [OKVIS](https://github.com/ethz-asl/okvis) (M/S+IMU)
  </td>
  <td>

  [VINS](https://github.com/HKUST-Aerial-Robotics/VINS-Mono) (M+IMU)
  </td>
  <td>

  [GMapping](http://wiki.ros.org/gmapping) (L)
  </td>
  </tr>


  <tr>
  <td>

  [Elastic fusion](https://github.com/mp3guy/ElasticFusion) (R)
  </td>
  <td>

  [Hector SLAM](http://wiki.ros.org/hector_slam) (L)
  </td>
  </tr>



 </table>

 ∗M=Monocular, S=Stereo, R=RGB-D, L=Lidar


<table width=100%>
<tr>
<th> MonoSLAM </th>
<th> PTAM </th>
</tr>
<tr>
<td>

J. Davison’s monocular SLAM [ [paper](https://ieeexplore.ieee.org/document/4160954) and [paper 2](https://ieeexplore.ieee.org/document/1238654) ] was proposed in 2007 as the first real-time monocular visual SLAM system. MonoSLAM uses the extended Kalman filter as the backend to track very sparse feature points shi-Tomasi corner point for feature point matching on the frontend. Since EKF occupies a prominent dominant position in early SLAM, MonoSLAM is also based on EKF, using the current state of the camera and all
landmark points as the state quantity, and updating the mean and covariance of the states.

The monocular camera tracks very sparse feature points in an image using active tracking technology. In EKF, each feature’s position obeys the Gaussian distribution, so we can express its mean value and uncertainty as an ellipsoid. In the right half of the picture, we can find some small ellipsoids distributed in 3D space. If the ellipsoid looks long in a specific direction, the landmark corresponding to it is more uncertain in that direction. We can imagine
that if a feature point converges, we should see it change from a very long ellipsoid
(initially very uncertain in the Z axis of the camera system) to a small point.

This approach seems to have many drawbacks today. Still, it was already a milestone work at that time because most of the previous visual SLAM systems could
not run online.


</td>
<td>

In 2007, Klein’s team proposed PTAM (Parallel Tracking and Mapping) [ [paper](https://ieeexplore.ieee.org/document/4538852) ], which
is also an important event in the development of visual SLAM. The important significance of PTAM lies in the following two points:

+ PTAM proposed and realized the parallelization of the tracking and mapping
process. Most people think it is clear that the tracking part needs to respond to
image data in real-time, and the optimization does not need to be calculated in
real-time. Backend optimization can be done slowly in the background, and then
thread synchronization can be performed when necessary. This is the first time the concept of front and backends have been distinguished in visual SLAM, leading to the structure of many later visual SLAM systems (most of the SLAMs we see now are divided into front and backends).

+ PTAM is the first real-time solution that uses nonlinear optimization instead of
traditional filters in the backend. It introduces the keyframe mechanism: we don’t
need to process each image carefully but string together several key images and
optimize its trajectory and map. Early SLAM mostly used EKF filters or their
variants. After PTAM, visual SLAM research gradually turned to the backend
dominated by nonlinear optimization.

However, from a modern perspective, PTAM can be regarded as one of the early
SLAM work combined with AR.

</td>
</tr>

<tr>
<th>ORB-SLAM Series</th>
<th>LSD-SLAM</th>
</tr>
<tr>
<td>

ORB-SLAM [ [paper](https://arxiv.org/abs/1502.00956) ] is a very famous and it was proposed in 2015 and is one of the most complete and easy-to-use systems in modern SLAM systems (if not the most complete and easy-to-use). . ORBSLAM is a peak of the mainstream feature-based open-source SLAM. Compared
with previous work, ORB-SLAM has the following obvious advantages:

+  It supports various sensor settings: monocular, binocular, and RGB-D. For most of
the sensors, it is almost possible to test it on ORB-SLAM. It has good versatility..

+ The entire system is calculated with ORB features, including the ORB dictionary for visual odometry and loop detection. It reflects that the ORB feature is
an excellent compromise between the computing platform’s efficiency and accuracy at this stage. ORB is not as time-consuming as SIFT or SURF and can be
calculated in real-time on the CPU; compared to simple corner features such as
Harris corners, it has good rotation and scaling invariance. Also, ORB provides
descriptors that enable us to perform loop detection and relocation in a large range
of motion.

+  ORB’s loop detection is its highlight. The excellent loop closure detection algorithm ensures that ORB-SLAM effectively prevents accumulated errors and can
be quickly retrieved after being lost. Many existing SLAM systems are not perfect in the relocalization point. For this reason, ORB-SLAM needs to load a large
ORB dictionary file.

+ ORB-SLAM innovatively uses the three threads structure in SLAM: Tracking
thread for real-time tracking of feature points, optimization thread for local bundle
adjustment (co-visibility Graph, commonly known as the small graph), and global
pose graph optimization thread (essential graph, known as the big graph). The
tracking thread is responsible for extracting ORB feature points for each new
image, comparing it with the most recent keyframe, calculating the feature points’
location, and roughly estimating the camera pose. The small graph solves a local
bundle adjustment problem, including the feature points in the local space and
the camera pose. This thread is responsible for refining camera poses and spatial
locations of feature points. The tracking and local mapping threads construct good
visual odometry. The third thread, the big graph thread, performs loop detection
on the global map and keyframes to eliminate accumulated errors.
Following the two-thread structure of PTAM, the three-thread design of ORBSLAM has achieved excellent tracking and mapping effects, ensuring the global
consistency of the trajectory and the map. This three-thread structure will also be
accepted and adopted by subsequent researchers.

+  ORB-SLAM has carried out many optimizations around feature points. For example, based on OpenCV feature extraction, ORB-SLAM uniforms the distribution
of image features. When optimizing the pose, a four-iteration optimization routine is used to obtain better matches. A more relaxed keyframe selection strategy
than PTAM is also used. These small improvements make ORB-SLAM far more
robust than other solutions: even for poor scenarios and poor calibration parameters, ORB-SLAM can work smoothly.


The advantages mentioned above make ORB-SLAM reach the state-of-the-art
open-source visual SLAM system. Of course, ORB-SLAM also has some shortcomings. Since the entire SLAM system uses feature points for calculation, we must detect the ORB feature for each
image, which is very time-consuming.

</td>
<td>

LSD-SLAM (Large Scale Direct monocular SLAM) is a SLAM work [ [paper](https://link.springer.com/chapter/10.1007/978-3-319-10605-2_54), [paper 2](https://openaccess.thecvf.com/content_iccv_2013/papers/Engel_Semi-dense_Visual_Odometry_2013_ICCV_paper.pdf) ] proposed by J. Engel et al. in 2014. Analogous to ORB-SLAM to feature points, LSDSLAM marks the direct method’s successful application in monocular SLAM. The
core contribution of LSD-SLAM is to apply the direct method to semi-dense monocular SLAM. It does not need to calculate the features but can also construct a semidense map. The semi-dense means estimating the pixel position with an obvious
gradient. Its main advantages are as follows:

+ The direct method of LSD-SLAM is for pixels. The authors creatively proposed
the relationship between pixel gradient and the direct method. And also, they
note the angular relationship between pixel gradient and epipolar direction in
dense reconstruction.

+ LSD-SLAM realizes the semi-dense reconstruction on CPU. The method based on feature points can only be sparse, and most of the dense reconstruction schemes use RGB-D sensors or use GPU to build dense maps.

+ The semi-dense tracking of LSD-SLAM uses some subtle tricks to ensure the
tracking’s real-time efficiency and stability. For example, LSD-SLAM does not
use a single-pixel or image block for epipolar searching but takes five points on
the epipolar line at equal distances to measure its SSD. In the depth estimation,
LSD-SLAM first initializes the depth with random numbers and then normalizes
the depth estimation after converges. When measuring the depth uncertainty, it
considers the triangle’s geometric relationship and the angle between the epipolar
line and the image gradient. The loop closure thread uses the similarity transformation group SE(3) to express the scale explicitly, which can be used to reduce
the scale drift in monocular SLAM.

Since LSD-SLAM uses the direct tracking method, it has both the direct method’s
advantages and disadvantages. For example, LSD-SLAM is very sensitive to camera
intrinsics and exposure time and is easily lost when the camera moves quickly. Also,
in the loop detection part, LSD-SLAM still relies on the feature point method for
loop detection and has not entirely eliminated the calculation of feature points.


</td>
</tr>

<tr>
<th>SVO</th>
<th>RTAB-MAP</th>
</tr>
<tr>
<td>

SVO is the abbreviation of Semi-direct Visual Odometry [[paper](https://rpg.ifi.uzh.ch/docs/ICRA14_Forster.pdf)]. It is visual odometry
based on the sparse direct method proposed by Forster et al. in 2014. The meaning of semi-direct in the original text refers to the mixed-use of feature points and direct methods: SVO tracks some key points (corner points, no descriptors), and then, like the direct method, uses the information of these key points to estimate camera movement and the map point’s position. 
SVO uses small blocks of 4 × 4 around the key points for block matching to estimate the camera’s motion. Compared with other programs, SVO’s most significant advantage is high-speed.
Due to the use of the sparse direct method, it does not have to laboriously calculate the
descriptor, nor does it need to process as much information as dense and semi-dense approaches. So, it can achieve real-time performance even on low-level computing
platforms.
Another SVO innovation is that it puts forward the concept of depth filter and
derives a depth filter based on uniform-Gaussian mixture distribution.


</td>
<td>

RTAB-MAP (Real-Time Appearance-Based Mapping) [[paper](https://www.semanticscholar.org/paper/Online-global-loop-closure-detection-for-SLAM-Labb%C3%A9-Michaud/533d8c11c8e533c34774eff76fcbab6559ee67e4)] is a classic scheme
in RGB-D SLAM. It implements everything that should be included in RGB-D
SLAM: feature-based visual odometry, bag-of-words-based loop detection, backend
pose map optimization, and point cloud and triangular mesh maps. Therefore, RTABMAP provides a complete (but somewhat huge) RGB-D SLAM solution. At present,
we can obtain the binary program directly from ROS
</td>
</tr>

</table>

<img src="slamh2.png" width=100%> 

VSLAM has the advantage of richer environmental information and is considered to be able to give mobile robots stronger perceptual ability and be applied in some specific scenarios.

 In Section 1, this note introduces the characteristics of traditional VSLAM in detail, including the direct method and the indirect method based on the front-end vision odometer, and makes a comparison between the depth camera-based VSLAM and the classical VSLAM integrated with IMU. In Section 2, this note is divided into two parts. We firstly go through the combination of deep learning and VSLAM from two neural networks, CNN and RNN. So this note summarizes the development direction of semantic VSLAM from three aspects of localization, mapping, and elimination of dynamic objects. 

 VSLAM can be divided into Traditional VSLAM ( Monocular / Stero vSLAM, RGB-D vSLAM { vSLAM based on feature based method and vSLAM based on direct method } , Visual Inertial SLAM { Loosely Coupled Visual Inertial and Tightly Coupled Visual Inertial } ) and Semantic VSLAM.


 The sensors used in the VSLAM typically include the monocular camera, stereo camera, and RGB-D camera. The monocular camera and the stereo camera have similar principles and can be used in a wide range of indoor and outdoor environments. As a special form of camera, the RGB-D camera can directly obtain image depth mainly by actively emitting infrared structured light or calculating time-of-flight (TOF). It is convenient to use, but sensitive to light, and can only be used indoors in most cases. Events camera as appeared in recent years, a new camera sensor, a picture of a different from the traditional camera. Events camera is “events”, can be as simple as “pixel brightness change”. The change of events camera output is pixel brightness, SLAM algorithm based on the event camera is still only in the preliminary study stage. TE (Absolute Trajectory Error) and RPE (Relative Pose Error) are the two most important indicators used to evaluate the accuracy of SLAM. The relative pose error is used to calculate the difference of pose changes in the same two-time stamps, which is suitable for estimating system drift. The absolute trajectory error directly calculates the difference between the real value of the camera pose and the estimated value of the SLAM system. 

 <b> ATE:</b> The absolute trajectory error is the direct difference between the estimated pose and the real pose, which can directly reflect the accuracy of the algorithm and the global trajectory consistency. It should be noted that the estimated pose and ground truth are usually not in the same coordinate system, so we need to pair them first: For stereo SLAM and RGB-D SLAM, the scale is uniform, so we need to calculate a transformation matrix from the estimated pose to the real pose by the least square method S ∈ SE (3). For monocular cameras with scale uncertainties, we need to calculate a similar transformation matrix S∈ Sim (3) from the estimated pose to the real pose. 

 
###  Traditional VSLAM:

#### Monocular/Stereo VSLAM: 

VSLAM can be divided into the direct method and indirect method according to the different image information collected by the front-end visual odometer. The indirect method needs to select a certain number of representative points from the collected images, called key points, and detect and match them in the following images to gain the camera pose. It not only saves the key information of the image but also reduces the amount of calculation, so it is widely used. The direct method uses all the information of the image without preprocessing and directly operates on pixel intensity, which has higher robustness in an environment with sparse texture. 

##### VSLAM Based on the Feature-Based Method : 

The core of indirect VSLAM is to detect, extract and match geometric features( points, lines, or planes), estimate camera pose, and build an environment map while retaining important information, it can effectively reduce calculation.

Feature extraction mostly adopted corner extraction methods in the early, such as Harris [[paper](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=88cdfbeb78058e0eb2613e79d1818c567f0920e2)], FAST [[paper](https://link.springer.com/chapter/10.1007/11744023_34%C2%A0)], GFTT [[paper](https://cir.nii.ac.jp/crid/1570009751070388480)], etc. However, in many scenarios, simple corners cannot provide reliable features, which prompts researchers to seek more stable local image features. Nowadays, typical VSLAM methods based on point features firstly use feature detection algorithms, such as SIFT [[paper](https://link.springer.com/article/10.1023/B:VISI.0000029664.99615.94)], SURF [[paper](https://www.sciencedirect.com/science/article/abs/pii/S1077314207001555)], and ORB [[paper](https://ieeexplore.ieee.org/abstract/document/6126544/)], to extract key points in the image for matching. Then gain pose after minimizing reprojection error. Feature points and corresponding descriptors in the image are employed for data association.

Comparison table of commonly used feature extraction algorithms:

<table width=100%>
<tr>
<th>Method</th>
<th>Year</th>
<th>Type</th>
<th>Speed</th>
<th>Rotation Invariance</th>
<th>Scale Invariance</th>
<th>Illumination Invariance</th>
<th>Anti Invariance</th>
</tr>
<tr>
<td>

ORB [[paper](https://ieeexplore.ieee.org/abstract/document/6126544)]
</td>
<td>2011</td>
<td>Point</td>
<td>High</td>
<td>Yes</td>
<td>Yes</td>
<td>Yes</td>
<td>Stronger</td>
</tr>
<tr>
<td>

SURF [[paper](https://www.sciencedirect.com/science/article/abs/pii/S1077314207001555)]
</td>
<td>2008</td>
<td>Point</td>
<td>Middle</td>
<td>Yes</td>
<td>Yes</td>
<td>No</td>
<td>Weak</td>
</tr>
<tr>
<td>

FAST [[paper](https://link.springer.com/chapter/10.1007/11744023_34%C2%A0)]
</td>
<td>2006</td>
<td>Point</td>
<td>High</td>
<td>No</td>
<td>Yes</td>
<td>No</td>
<td>Weak</td>
</tr>
<tr>
<td>

SIFT [[paper](https://link.springer.com/article/10.1023/B:VISI.0000029664.99615.94)]
</td>
<td>2004</td>
<td>Point</td>
<td>Low</td>
<td>Yes</td>
<td>Yes</td>
<td>Yes</td>
<td>Strong</td>
</tr>
<tr>
<td>

Shi Tomasi / GFTT [[paper](https://cir.nii.ac.jp/crid/1570009751070388480)]
</td>
<td>1994</td>
<td>Corner</td>
<td>Middle</td>
<td>Yes</td>
<td>No</td>
<td>Yes</td>
<td>Week</td>
</tr>
<tr>
<td>

Harris [[paper](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=88cdfbeb78058e0eb2613e79d1818c567f0920e2)]
</td>
<td>1988</td>
<td>Corner</td>
<td>Low</td>
<td>Yes</td>
<td>No</td>
<td>Yes</td>
<td>Weak</td>
</tr>
<tr>
<td>

LSD [[paper](https://jakobengel.github.io/pdf/engel14eccv.pdf)]
</td>
<td>2010</td>
<td>Line</td>
<td>Middle</td>
<td>Yes</td>
<td>Yes</td>
<td>Yes</td>
<td>Stronger</td>
</tr>
</table>

The commonly used line feature extraction algorithm is LSD.

SLAM Algorithms : MonoSLAM [[paper](https://ieeexplore.ieee.org/abstract/document/4160954)] in 2007, PTAM [[paper](https://ieeexplore.ieee.org/abstract/document/4538852)] in 2007, LineSLAM [[paper](https://link.springer.com/chapter/10.1007/978-3-319-03413-3_49)] in 2014, ORB-SLAM [[paper](https://ieeexplore.ieee.org/abstract/document/7219438)] in 2015, ORB SLAM 2 [[paper](https://ieeexplore.ieee.org/abstract/document/7946260)] in 2017, monocular PL SLAM [[paper](https://ieeexplore.ieee.org/abstract/document/7989522)] in 2017, stereo PL SLAM [[paper](https://ieeexplore.ieee.org/abstract/document/8680013)] in 2017, ORB SLAM 3 [[paper](https://ieeexplore.ieee.org/abstract/document/9440682)] in 2021.

 The ORB-SLAM family is one of the most widely used visual SLAM solutions due to its real-time CPU performance and robustness. However, the ORB-SLAM series relies heavily on environmental features, so it may be difficult to obtain enough feature points in an environment without texture features

##### VSLAM Based on Direct Method : 

Different from feature-based methods, the direct method operates directly on pixel intensity and can retain all information about the image. Furthermore, the direct method cancels the process of feature extraction and matching, so the computational efficiency is better than the indirect method. Furthermore, it has good adaptability to the environment with complex textures.

SLAM Algorithms : DTAM [[paper](https://ieeexplore.ieee.org/abstract/document/6126513)] in 2011, SVO [[paper](https://ieeexplore.ieee.org/abstract/document/6906584)] in 2014 (semi-direct), DSO [[paper](https://ieeexplore.ieee.org/abstract/document/7898369)] in 2018, 

SVO combines the advantages of the feature point method and direct method. The algorithm is divided into two main threads: motion estimation and mapping. Motion estimation is carried out by feature point matching, but mapping is carried out by the direct method.

DSO can calculate accurate camera attitude in poor feature point detector performance, improving the robustness of low-texture areas or blurred images. In addition, the DSO uses both geometric and photometric camera calibration results for high accuracy estimation. However, DSO only considers local geometric consistency, so it inevitably produces cumulative errors. Furthermore, it is not a complete SLAM because it does not include loop closure, map reuse, etc.

#### RGB-D SLAM

An RGB-D camera can simultaneously collect environmental color images and depth images, and directly gain depth maps mainly by actively emitting infrared structured light or calculating time-of-flight (TOF). 

Some SLAM algorithms for sensors with an RGB-D camera :

<table width=100%>
<tr>
<th>Method</th>
<th>Year</th>
<th>Camera Tracking</th>
<th>Loop Closure</th>
<th>Code Resource</th>
</tr>

<tr>
<td>

KinectFusion [[paper](https://ieeexplore.ieee.org/abstract/document/6162880)]
</td>
<td>2011</td>
<td>Diret</td>
<td>No</td>
<td>

[[code](https://github.com/ParikaGoel/KinectFusion)]
</td>
</tr>

<tr>
<td>

Kinitinuous [[paper](https://scholar.google.com/scholar_lookup?title=Kintinuous:+Spatially+extended+kinectfusion&conference=Proceedings+of+the+RSS+Workshop+on+RGB-D:+Advanced+Reasoning+with+Depth+Cameras&author=Kaess,+M.&author=Fallon,+M.&author=Johannsson,+H.&author=Leonard,+J.&publication_year=2012&pages=9)]</td>
<td>2012</td>
<td>Direct</td>
<td>Yes</td>
<td>

[[code](https://github.com/mp3guy/Kintinuous)]
</td>
</tr>


<tr>
<td>

RGB-D SLAMv2 [[paper](https://ieeexplore.ieee.org/abstract/document/6594910)]</td>
<td>2013</td>
<td>Indirect</td>
<td>Yes</td>
<td>

[[code](https://github.com/felixendres/rgbdslam_v2)]
</td>
</tr>


<tr>
<td>

ElasticFusion [[paper](https://journals.sagepub.com/doi/abs/10.1177/0278364916669237)]</td>
<td>2016</td>
<td>Direct</td>
<td>Yes</td>
<td>

[[code](https://github.com/mp3guy/ElasticFusion)]
</td>
</tr>


<tr>
<td>

DVO-SLAM [[paper](https://ieeexplore.ieee.org/abstract/document/8007295)]</td>
<td>2017</td>
<td>Direct</td>
<td>Yes</td>
<td>

[[code](https://github.com/tum-vision/dvo_slam)]
</td>
</tr>


<tr>
<td>

BundleFusion [[paper](https://dl.acm.org/doi/abs/10.1145/3072959.3054739)]</td>
<td>2017</td>
<td>Hybrid</td>
<td>Yes</td>
<td>

[[code](https://github.com/niessner/BundleFusion)]
</td>
</tr>


<tr>
<td>

RGBDTAM [[paper](https://ieeexplore.ieee.org/abstract/document/8206593)]</td>
<td>2017</td>
<td>Direct</td>
<td>Yes</td>
<td>

[[code](https://github.com/alejocb/rgbdtam)]
</td>
</tr>
</table>


`KinectFusion` is the first real-time 3D reconstruction system based on an RGB-D camera. It uses a point cloud created by the depth to estimate the camera pose through ICP (Iterative Closest Point). Then splices multi-frame point cloud collection based on the camera pose, and expresses reconstruction result by the TSDF (Truncated signed distance Function) model. The 3D model can be constructed in real-time with GPU acceleration. However, the system has not been optimized by loop closure. Furthermore, there will be obvious errors in long-term operation, and the RGB information of the RGB-D camera has not been fully utilized. 

In contrast, `ElasticFusion` makes full use of the color and depth information of the RGB-D camera. It estimates the camera pose by the color consistency of RGB and estimates the camera pose by ICP. Then improves the estimation accuracy of the camera pose by constantly optimizing and reconstructing the map. Finally, the surfel model was used for map representation, but it could only be reconstructed in a small indoor scene. `Kinitinuous` adds loop closure based on KinectFusion and makes non-rigid body transformation for 3d rigid body reconstruction by using a deformation graph for the first time. So it makes the results of two-loop closure reconstruction overlap, achieving good results in an indoor environment.



Although the RGB-D camera is more convenient to use, the RGB-D camera is extremely sensitive to light. Furthermore, there are many problems with narrow, noisy, and small horizons, so most of the situation is only used in the room. 

#### Visual-Inertial SLAM

The pure visual SLAM algorithm has achieved many achievements. However, it is still difficult to solve the effects of image blur caused by fast camera movement and poor illumination by using only the camera as a single sensor. IMU is considered to be one of the most complementary sensors to the camera. It can obtain accurate estimation at high frequency in a short time, and reduce the impact of dynamic objects on the camera. In addition, the camera data can effectively correct the cumulative drift of IMU. At the same time, due to the miniaturization and cost reduction of cameras and IMU, visual-inertial fusion has also achieved rapid development.

Nowadays, visual-inertial fusion can be divided into loosely coupled and tightly coupled according to whether image feature information is added to the state vector. Loosely coupled means the IMU and the camera estimate their motion, respectively, and then fuse their pose estimation. Tightly coupled refers to the combination of the state of IMU and the state of the camera to jointly construct the equation of motion and observation, and then perform state estimation.

##### Loosely Coupled Visual-Inertial : Generally, the fusion is performed through EKF. 
##### Tightly Coupled Visual-Inertial :

The core of the tightly coupled is to combine the states of the vision sensor and IMU through an optimized filter. It needs the image features to be added to the feature vector to jointly construct the motion equation and observation equation. 

Then perform state estimation to obtain the pose information. Tightly coupled needs full use of visual and inertial measurement information, which is complicated in method implementation but can achieve higher pose estimation accuracy. Therefore, it is also the mainstream method, and many breakthroughs have been made in this area.

This table holds information on best visual-based SLAM algorithms:

<table width=100%>
<tr>
<th></th>
<th>Method</th>
<th>Sensor</th>
<th>Frontend</th>
<th>Backend</th>
<th>Loop Closure</th>
<th>Mapping</th>
<th>Code Resources</th>
</tr>

<tr>
<td></td>
<td>

MonoSLAM [[paper](https://ieeexplore.ieee.org/abstract/document/4160954)]
</td>
<td>M</td>
<td>P</td>
<td>F</td>
<td>No</td>
<td>Sparse</td>
<td>

[[code](https://github.com/rrg-polito/mono-slam)]</td>
</tr>



<tr>
<td></td>
<td>

PTAM [[paper](https://ieeexplore.ieee.org/abstract/document/4538852)]
</td>
<td>M</td>
<td>P</td>
<td>O</td>
<td>No</td>
<td>Sparse</td>
<td>

[[code](https://github.com/Oxford-PTAM/PTAM-GPL)]</td>
</tr>



<tr>
<td></td>
<td>

ORB-SLAM2 [[paper](https://ieeexplore.ieee.org/abstract/document/7946260)]
</td>
<td>M/S/R</td>
<td>P</td>
<td>O</td>
<td>Yes</td>
<td>Sparse</td>
<td>

[[code](https://github.com/raulmur/ORB_SLAM2)]</td>
</tr>



<tr>
<td></td>
<td>

PL-SVO [[paper](https://ieeexplore.ieee.org/abstract/document/7759620)]
</td>
<td>M</td>
<td>PL</td>
<td>O</td>
<td>No</td>
<td>Sparse</td>
<td>

[[code](https://github.com/rubengooj/pl-svo)]</td>
</tr>



<tr>
<td>Visual</td>
<td>

PL-SLAM [[paper](https://ieeexplore.ieee.org/abstract/document/7989522)]
</td>
<td>M/S</td>
<td>PL</td>
<td>O</td>
<td>Yes</td>
<td>Sparse</td>
<td>

[[code](https://github.com/rubengooj/pl-slam)]</td>
</tr>



<tr>
<td></td>
<td>

DTAM [[paper](https://ieeexplore.ieee.org/abstract/document/6126513)]
</td>
<td>M</td>
<td>D</td>
<td>O</td>
<td>No</td>
<td>Dense</td>
<td>

[[code](https://github.com/anuranbaka/OpenDTAM)]</td>
</tr>



<tr>
<td></td>
<td>

SVO [[paper](https://ieeexplore.ieee.org/abstract/document/6906584)]
</td>
<td>M</td>
<td>H</td>
<td>O</td>
<td>No</td>
<td>Sparse</td>
<td>

[[code](https://github.com/uzh-rpg/rpg_svo)]</td>
</tr>



<tr>
<td></td>
<td>

LSD-SLAM [[paper](https://link.springer.com/chapter/10.1007/978-3-319-10605-2_54)]
</td>
<td>M/S</td>
<td>D</td>
<td>O</td>
<td>Yes</td>
<td>Semi Dense</td>
<td>

[[code](https://github.com/tum-vision/lsds_lam)]</td>
</tr>



<tr>
<td></td>
<td>

DSO [[paper](https://ieeexplore.ieee.org/abstract/document/7898369)]
</td>
<td>M</td>
<td>D</td>
<td>O</td>
<td>No</td>
<td>Sparse</td>
<td>

[[code](https://github.com/JakobEngel/dso)]</td>
</tr>



<tr>
<th></th>
<th>Method</th>
<th>Sensor</th>
<th>Frontend</th>
<th>Backend</th>
<th>Loop Closure</th>
<th>Mapping</th>
<th>Code Resources</th>
</tr>


<tr>
<td></td>
<td>

MSCKF [[paper](https://ieeexplore.ieee.org/abstract/document/4209642)]
</td>
<td>M + I</td>
<td>T</td>
<td>F</td>
<td>No</td>
<td>Sparse</td>
<td>

[[code](https://github.com/daniilidis-group/msckf_mono)]</td>
</tr>


<tr>
<td>Visual-inertial</td>
<td>

OKVIS [[paper](https://journals.sagepub.com/doi/abs/10.1177/0278364914554813)]</td>
<td>S + I</td>
<td>T</td>
<td>O</td>
<td>No</td>
<td>Sparse</td>
<td>

[[code](https://github.com/ethz-asl/okvis)]</td>

</tr>

<tr>
<td></td>
<td>

ROVIO [[paper](https://ieeexplore.ieee.org/abstract/document/7353389)] </td>
<td>M + I</td>
<td>T</td>
<td>F</td>
<td>No</td>
<td>Sparse</td>
<td>

[[code](https://github.com/ethz-asl/rovio)]</td>
</tr>


<tr>
<td></td>
<td>

VINS-Mono [[paper](https://ieeexplore.ieee.org/abstract/document/8421746)]</td>
<td>M + I</td>
<td>T</td>
<td>O</td>
<td>Yes</td>
<td>Sparse</td>
<td>

[[code](https://github.com/HKUST-Aerial-Robotics/VINS-Mono)]</td>
</tr>
</table>

Sensor: <b>M</b> represents Monocular camera; <b>S</b> represents Stereo camera; <b>R</b> represents RGB-D camera and <b>I</b> represents IMU. Front-end: <b>P</b> represents Point; <b>PL</b> represents Point-line; <b>D</b> represents Direct; <b>H</b> represents Hybrid. Back-end: <b>F</b> represents Filtering; <b>O</b> represents Optimization. Coupling: <b>T</b> represents Tightly.


As a supplement to cameras, inertial sensors can effectively solve the problem that a single camera cannot cope with. Visual inertial fusion is bound to become a long-term hot direction of SLAM research.


<table width=100%>
<tr>
<th>Name</th>
<th>Year</th>
<th>Visual odometry</th>
<th>Feature method</th>
<th>Optimization</th>
<th>Loop closure detection</th>
<th>Real-time efficiency</th>
<th>Robustness</th>
<th>Characteristic</th>
</tr>

<tr>
<td>Mono-SLAM

[[paper](https://ieeexplore.ieee.org/abstract/document/4160954)]</td>
<td>2007</td>
<td>Feature method</td>
<td>Shi-Tomasi</td>
<td>EKF</td>
<td>—</td>
<td>✩✩</td>
<td>✩✩</td>
<td>The Mono-SLAM is the first real-time monocular V-SLAM system.</td>
</tr>

<tr>
<td>PTAM

[[paper](https://ieeexplore.ieee.org/abstract/document/4538852)]</td>
<td>2007</td>
<td>Feature method</td>
<td>Shi-Tomasi</td>
<td>BA</td>
<td>—</td>
<td>✩✩✩</td>
<td>✩✩✩</td>
<td>The PTAM is the first monocular V-SLAM system based on nonlinear optimization of key frames and proposes the concept of front end and back end.</td>
</tr>

<tr>
<td>LSD-SLAM

[[paper](https://link.springer.com/chapter/10.1007/978-3-319-10605-2_54)]</td>
<td>2014</td>
<td>Direct method</td>
<td>—</td>
<td>Depth filter</td>
<td>√</td>
<td>✩✩✩</td>
<td>✩✩✩</td>
<td>The LSD-SLAM applies the direct method to semidense monocular V-SLAM</td>
</tr>

<tr>
<td>SVO

[[paper](https://ieeexplore.ieee.org/abstract/document/6906584)]</td>
<td>2014</td>
<td>Semidirect method</td>
<td>FAST + LKe</td>
<td>Depth filter	</td>
<td>—</td>
<td>✩✩✩✩✩✩</td>
<td>✩✩✩</td>
<td>The SVO is a visual odometer based on the sparse direct method. The biggest advantage of SVO over other schemes is its speed.</td>
</tr>

<tr>
<td>SPTAM

[[paper](https://www.sciencedirect.com/science/article/abs/pii/S0921889015302955)]</td>
<td>2017</td>
<td>Feature method</td>
<td>Shi-Tomasi</td>
<td>BA</td>
<td>√</td>
<td>✩✩✩✩</td>
<td>✩✩✩✩</td>
<td>Binocular V-SLAM system based on PTAM</td>
</tr>

<tr>
<td>LDSO

[[paper](https://ieeexplore.ieee.org/abstract/document/8593376)]</td>
<td>2018</td>
<td>Direct method</td>
<td>Frames</td>
<td>BA</td>
<td>√</td>
<td>✩✩✩✩</td>
<td>✩✩✩✩✩</td>
<td>The LDSO expands on DSO into a monocular V-SLAM system with loop detection and pose optimization.</td>
</tr>

<tr>
<td>LCSD-SLAM

[[paper](https://ieeexplore.ieee.org/abstract/document/8584894)]</td>
<td>2018</td>
<td>Semidirect method</td>
<td>Descriptor</td>
<td>BA</td>
<td>√</td>
<td>✩✩✩✩</td>
<td>✩✩✩✩✩</td>
<td>The LCSD-SLAM is a loosely coupled semidirect monocular V-SLAM.</td>
</tr>

<tr>
<td>Maplab

[[paper](https://ieeexplore.ieee.org/abstract/document/8276568)]</td>
<td>2018</td>
<td>Semidirect method</td>
<td>Descriptor</td>
<td>—</td>
<td>√</td>
<td>✩✩✩✩</td>
<td>✩✩✩</td>
<td>The maplab is a flexible and versatile multi-robot and multimode V-SLAM framework.</td>
</tr>

<tr>
<td>RE-SLAM

[[paper](https://ieeexplore.ieee.org/abstract/document/8794462)]</td>
<td>2019</td>
<td>Semidirect method</td>
<td>Edges</td>
<td>BA</td>
<td>√</td>
<td>✩✩✩</td>
<td>✩✩✩✩✩✩</td>
<td>The RE-SLAM is a real-time robust edge-based V-SLAM system</td>
</tr>

<tr>
<td>CCM-SLAM

[[paper](https://onlinelibrary.wiley.com/doi/abs/10.1002/rob.21854)]</td>
<td>2019</td>
<td>Feature method</td>
<td>ORB</td>
<td>BA</td>
<td>√</td>
<td>✩✩✩</td>
<td>✩✩✩✩✩</td>
<td>The CCM-SLAM is a kind of multi-robot cooperative V-SLAM, which can be used for cooperative mapping in unknown environment.</td>
</tr>

<tr>
<td>ORB-SLAM3

[[paper](https://ieeexplore.ieee.org/abstract/document/9440682)]</td>
<td>2020</td>
<td>Feature method</td>
<td>ORB</td>
<td>BA</td>
<td>√</td>
<td>✩✩✩✩✩</td>
<td>✩✩✩✩✩</td>
<td>The ORB-SLAM3 is the first feature-based tightly coupled VIO system that relies only on maximum a posteriori estimation.</td>
</tr>

<tr>
<td>VINS-fusion

[[paper](https://ieeexplore.ieee.org/abstract/document/9562103)]</td>
<td>2021</td>
<td>Semidirect method</td>
<td>Shi-Tomasi + LK</td>
<td>BA</td>
<td>√</td>
<td>✩✩✩✩✩</td>
<td>✩✩✩</td>
<td>The Vins-fusion is a multi-sensor state estimator based on optimization, which can achieve accurate self-localization for autonomous applications.</td>
</tr>

<tr>
<td>VOLDOR-SLAM

[[paper](https://arxiv.org/abs/2104.06800)]</td>
<td>2021</td>
<td>Dense-indirect method</td>
<td>Optical flow</td>
<td>BA</td>
<td>√</td>
<td>✩✩✩✩</td>
<td>✩✩✩✩</td>
<td>The VOLDOR is an intensive indirect visual ranging method that takes an externally estimated optical flow field as input</td>
</tr>

<tr>
<td>ESVO

[[paper](https://ieeexplore.ieee.org/abstract/document/9386209)]</td>
<td>2021</td>
<td>Semidirect method</td>
<td>Events</td>
<td>Depth filter</td>
<td>—</td>
<td>✩✩✩✩✩✩</td>
<td>✩✩✩✩</td>
<td>The ESVO is a novel pipeline for real-time visual odometry using a stereo event-based camera.</td>
</tr>


</table>

There are several new SLAM methods like - HKU-MaRS-HBA, Pharos SLAM and few of them can be found at [Hilti SLAM Challenge leaderboard](https://hilti-challenge.com/).