# Motion Models in Machine Vision :

[ [Lecture 08 - Motion Model - 2014](https://youtu.be/0XRFxxpj4sM?si=XLCfuvJUf3drruxe) | [CV](https://www.youtube.com/playlist?list=PLd3hlSJsX_ImKP68wfKZJVIPTd8Ie5u-9), [Overview | Optical Flow](https://www.youtube.com/watch?v=lnXFcmLB7sM&list=PL2zRqk16wsdoYzrWStffqBAoUY8XdvatV) ]


Motion models in computer vision are mathematical formulations that describe how objects move within a visual scene. They play a critical role in a variety of applications, such as video surveillance, autonomous driving, robotics, and augmented reality. Here’s an in-depth overview of motion models:

### Types of Motion Models

### + `Translational Motion Model`:

Description: This model assumes that the object moves in a straight line, characterized by translation along the x and y axes. Mathematical Formulation:

$$ 𝑥′=𝑥+𝑡 $$

Where 𝑥 is the original position, 𝑡 is the translation vector, and 𝑥′ is the new position.


### + `Rotational Motion Model`:

Description: This model accounts for objects rotating around a fixed point, usually the center of rotation.
Mathematical Formulation:

$$ 𝑥′ = 𝑅(\theta) 𝑥 $$

Where 𝑅(𝜃) is the rotation matrix for an angle 𝜃.


### + `Affine Motion Model`:

Description: This model includes translation, rotation, scaling, and shearing. It is more flexible and can describe more complex motions.
Mathematical Formulation:

$$ 𝑥′= 𝐴𝑥 + 𝑡 $$

Where 𝐴 is a 2×2 matrix that includes rotation, scaling, and shearing components, and 𝑡 is the translation vector.

### + `Homography Motion Model`:

Description: This model describes the motion between two planes in 3D space and is used when the camera undergoes more complex transformations.
Mathematical Formulation:

$$ 𝑥′=𝐻𝑥 $$

Where 𝐻 is a 3×3 homography matrix.

### + `Optical Flow Model`:

Description: This model estimates the motion of each pixel in a sequence of images. It is highly detailed and useful for analyzing fine movements within the scene.
Mathematical Formulation:
The optical flow constraint equation (OFCE):

$$ 𝐼_𝑥 𝑢 + 𝐼_𝑦 𝑣 + 𝐼_𝑡 = 0 $$

Where  `I_x`,  `I_y`, and `I_t` are the partial derivatives of the image intensity I with respect to x, y, and time, respectively, and (u,v) is the optical flow vector.


Motion libraries : github@/[motion-detection](https://github.com/topics/motion-detection), [deepgaze](https://github.com/mpatacchiola/deepgaze), [motion-capture](https://github.com/topics/motion-capture), [motion-tracking](https://github.com/topics/motion-tracking), [motion-transfer](https://github.com/topics/motion-transfer), [optical-flow](https://github.com/topics/optical-flow), [awesome-optical-flow](https://github.com/hzwer/Awesome-Optical-Flow), [mmflow](https://github.com/open-mmlab/mmflow), [crowd-flow](https://github.com/tsenst/CrowdFlow), [bflow](https://github.com/uzh-rpg/bflow).

## Machine Vision :

#### +  Image Formation, Perspective Projection, Time Derivative, Motion Field : [[notes](./code/cv/mv/5546a6b8d36a2d997929ba1aeb8c5ed3_MIT6_801F20_lec2.pdf)]


#### +  Time to Contact, Focus of Expansion, Direct Motion Vision Methods, Noise Gain : [[notes](./code/cv/mv/8479ce6555a34d442ae26270b4a056fc_MIT6_801F20_lec3.pdf)]


#### +  Fixed Optical Flow, Optical Mouse, Constant Brightness Assumption, Closed Form Solution  : [[notes](./code/cv/mv/d77aaf27b83126aca7f3afe1f0dcd808_MIT6_801F20_lec4.pdf)]


#### +  TCC and FOR MontiVision Demos, Vanishing Point, Use of VPs in Camera Calibration : [[notes](./code/cv/mv/0219f6ffeee5b3c30b6b1f654db65df1_MIT6_801F20_lec5.pdf)]


#### +  Photometric Stereo, Noise Gain, Error Amplification, Eigenvalues and Eigenvectors Review : [[notes](./code/cv/mv/75d0a565a80b022f2816f53616c45721_MIT6_801F20_lec6.pdf)]


#### +  Gradient Space, Reflectance Map, Image Irradiance Equation, Gnomonic Projection : [[notes](./code/cv/mv/0e2c2f75b7dc6602db91b55694ab963e_MIT6_801F20_lec7.pdf)]


#### +  Shading, Special Cases, Lunar Surface, Scanning Electron Microscope, Green's Theorem : [[notes](./code/cv/mv/54c4d5416aa7fc5d02820cb469d6441b_MIT6_801F20_lec8.pdf)]


#### +  Shape from Shading, General Case - From First Order Nonlinear PDE to Five ODEs : [[notes](./code/cv/mv/383f904e731a095132c13a9779220de8_MIT6_801F20_lec9.pdf)]


#### +  Characteristic Strip Expansion, Shape from Shading, Iterative Solutions : [[notes](./code/cv/mv/617445f0e31836831b40d42cb2f11a10_MIT6_801F20_lec10.pdf)]


#### +  Edge Detection, Subpixel Position, CORDIC, Line Detection  : [[notes](./code/cv/mv/5b03361b8fe3ccb51a0ce3c685dd88ae_MIT6_801F20_lec11.pdf)]


#### +  Blob Analysis, Binary Image Processing, Green's Theorem, Derivative and Integral : [[notes](./code/cv/mv/5e90d5693d5d378d3f19bf67913295aa_MIT6_801F20_lec12.pdf)]


#### +  Object Detection, Recognition and Pose Determination, PatQuick : [[notes](./code/cv/mv/63bf90743360a41d70eda346816d8304_MIT6_801F20_lec13.pdf)]


#### +  Inspection in PatQuick, Hough Transform, Homography, Position Determination, Multi-Scale : [[notes](./code/cv/mv/783e51bd8138fb098f22554024a57459_MIT6_801F20_lec14.pdf)]


#### +  Alignment, PatMax, Distance Field, Filtering and Sub-Sampling : [[notes](./code/cv/mv/c22810674f0599193f8b553b348de3b6_MIT6_801F20_lec15.pdf)]


#### +  Fast Convolution, Low Pass Filter Approximations, Integral Images : [[notes](./code/cv/mv/1233c0204538d1731732739c8491687b_MIT6_801F20_lec16.pdf)]


#### +  Photogrammetry, Orientation, Axes of Inertia, Symmetry, Orientation : [[notes](./code/cv/mv/e11dd76b64a8bd8ee48efefcdeea2bf6_MIT6_801F20_lec17.pdf)]


#### +  Rotation and How to Represent It, Unit Quaternions, the Space of Rotations : [[notes](./code/cv/mv/fac101ccf4d7b8cdf775666d2d1e2146_MIT6_801F20_lec18.pdf)]


#### +  Absolute Orientation in Closed Form, Outliers and Robustness, RANSAC : [[notes](./code/cv/mv/3684c9529d76a9a87fe3db7ae5e91f71_MIT6_801F20_lec19.pdf)]


#### +  Space of Rotations, Regular Tessellations, Critical Surfaces, Binocular Stereo : [[notes](./code/cv/mv/5e98d8a6c0edce797859526dece67aea_MIT6_801F20_lec20.pdf)]


#### +  Relative Orientation, Binocular Stereo, Structure, Quadrics, Calibration, Reprojection : [[notes](./code/cv/mv/62d1f2dbe217acc92df2112bdcff6568_MIT6_801F20_lec21.pdf)]


#### +  Exterior Orientation, Recovering Position & Orientation, Bundle Adjustment, Object Shape : [[notes](./code/cv/mv/a75ecc8eb2304a00e13985e1592ca368_MIT6_801F20_lec22.pdf)]


#### +  Gaussian Image, Solids of Revolution, Direction Histograms, Regular Polyhedra : [[notes](./code/cv/mv/c5308aca0506b5504baf7ebcd9878551_MIT6_801F20_lec23.pdf)]

[[homework 1](./code/cv/mv/hw/44cf7a39fd1de47fc359dc801be9c9fc_MIT6_801F20_hw1.pdf)] | [[homework 2](./code/cv/mv/hw/480e8d0833ac9552ee071d15319ef2cf_MIT6_801F20_hw2.pdf)] | [[homework 3](./code/cv/mv/hw/48f8beaf1ec24b134f10d91fc10b40e4_MIT6_801F20_hw3.pdf)] | [[homework 4](./code/cv/mv/hw/910c933c8b46a3a3c8c0ff804b1514c7_MIT6_801F20_hw4.pdf)] | [[homework 5](./code/cv/mv/hw/9ff064d1c944de62cb896e7c9d9571db_MIT6_801F20_hw5.pdf)]