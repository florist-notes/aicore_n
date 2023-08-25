# SLAM Algorithms II:

## Semantic VSLAM:

Semantic SLAM refers to a SLAM system that can not only obtain geometric information of the unknown environment and robot movement information but also detect and identify targets in the scene. It can obtain semantic information such as their functional attributes and relationship with surrounding objects, and even understand the contents of the whole environment.

In the early stage, some researchers tried to improve the performance of VSLAM by extracting semantic information in the environment using neural networks such as CNN. In the modern stage, target detection, semantic segmentation, and other deep learning methods are powerful tools to promote the development of semantic VSLAM.

Interesting work:
    + [Unsupervised CNN for Single View Depth Estimation: Geometry to the Rescue](https://link.springer.com/chapter/10.1007/978-3-319-46484-8_45)
    + [Learning Deep Representation for Place Recognition in SLAM](https://link.springer.com/chapter/10.1007/978-3-319-69900-4_71)
    + [Unsupervised learning to detect loops using deep neural networks for visual SLAM system](https://link.springer.com/article/10.1007/s10514-015-9516-2)
    + [Variational Bayesian Approach to Condition-Invariant Feature Extraction for Visual Place Recognition](https://www.mdpi.com/2076-3417/11/19/8976)
    + [Online Mutual Adaptation of Deep Depth Prediction and Visual SLAM](https://arxiv.org/abs/2111.04096)
    + [SymbioLCD: Ensemble-Based Loop Closure Detection using CNN-Extracted Objects and Visual Bag-of-Words](https://ieeexplore.ieee.org/abstract/document/9636622)

### Neural Networks with VSLAM:

CNN can capture spatial features from the image, which help us accurately identify the object and its relationship with other objects in the image. The characteristic of RNN is that it can process an image or numerical data. Because of the memory capacity of the network itself, it can learn data types with contextual correlation. CNN has the advantages of extracting features of things with a certain model, and then classifying, identifying, predicting, or deciding based on the features. It can be helpful to different modules of VSLAM. In addition, RNN has great advantages in helping to establish consistency between nearby frames.

#### +  CNN with VSLAM:

<table width=100%>
<tr>
<th>Part</th>
<th>Method</th>
<th>Contribution</th>
</tr>

<tr>
<td>Image Depth Estimation</td>
<td>

CNN-SLAM [[paper](https://arxiv.org/abs/1704.03489)]
</td>
<td>The depth estimation is performed only on the keyframe, which improves the computing efficiency.</td>
</tr>


<tr>
<td></td>
<td>

UnDeepVo [[paper](https://ieeexplore.ieee.org/abstract/document/8461251)]
</td>
<td>Real-scale monocular vision odometer is realized in an unsupervised way.</td>
</tr>


<tr>
<td></td>
<td>

Code-SLAM [[paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Bloesch_CodeSLAM_--_Learning_CVPR_2018_paper.html)]
 </td>
<td>A real-time monocular SLAM system is implemented that allows simultaneous optimization of camera motion and maps.</td>
</tr>


<tr>
<td></td>
<td>

DVSO [[paper](https://openaccess.thecvf.com/content_ECCV_2018/html/Nan_Yang_Deep_Virtual_Stereo_ECCV_2018_paper.html)]</td>
<td>Design a novel deep network that refines predicted depth from a single image in a two-stage process.</td>
</tr>

<hr>


<tr>
<td>Pose estimation</td>
<td>

DeTone et al.  [[paper](https://arxiv.org/abs/1707.07410)]</td>
<td>It uses only the location of points, not the descriptor of local points.</td>
</tr>

<tr>
<td></td>
<td>

VINet [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/11215)]</td>
<td>The ability to combine the information in a specific area naturally and cleverly can significantly reduce drift.</td>
</tr>

<tr>
<td></td>
<td>

D3VO [[paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_D3VO_Deep_Depth_Deep_Pose_and_Deep_Uncertainty_for_Monocular_CVPR_2020_paper.html)]</td>
<td>The proposed monocular visual odometer framework utilizes deep learning networks at three levels.</td>
</tr>

<tr>
<td></td>
<td>

Zhu et al. [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231221013874)]</td>
<td>Present a novel four-branch network to learn the rotation and translation by leveraging Convolutional Neural Networks (CNNs) to focus on different quadrants of optical flow input.</td>
</tr>

<hr>

<tr>
<td>Loop closure</td>
<td>

Memon et al. [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0921889019308425)]</td>
<td>Two deep neural networks are used together to speed up the loop closure detection and to ignore the effect of mobile objects on loop closure detection.</td>
</tr>


<tr>
<td></td>
<td>

Li et al. [[paper](https://ieeexplore.ieee.org/abstract/document/9340907)]</td>
<td>Train a visual vocabulary of local features with a Bag of Words (BoW) method. Based on the local features, global features, and vocabulary, a highly reliable loop closure detection method is built.</td>
</tr>


<tr>
<td></td>
<td>

Qin et al. [[paper](https://www.sciencedirect.com/science/article/abs/pii/S1047320321000389)]</td>
<td>Models the visual scene as a semantic sub-graph by only preserving the semantic and geometric information from object detection.</td>
</tr>

<hr>

<tr>
<td>Semantic information</td>
<td>

CNN-SLAM [[paper](https://arxiv.org/abs/1704.03489)]</td>
<td>By integrating Geometry and semantic information, a map with semantic information is generated.</td>
</tr>

<tr>
<td></td>
<td>

Naseer et al. [[paper](https://ieeexplore.ieee.org/abstract/document/7989305)]</td>
<td>To achieve real-time semantic segmentation and maintain a good efficiency of differentiation.</td>
</tr>


<tr>
<td></td>
<td>

SemanticFusion [[paper](https://ieeexplore.ieee.org/abstract/document/7989538)]</td>
<td>The semantic prediction of CNN’s multiple views can be probabilistically integrated into the map.</td>
</tr>

<tr>
<td></td>
<td>

Qin et al. [[paper](https://ieeexplore.ieee.org/abstract/document/9340939)]</td>
<td>A novel semantic feature used in the visual SLAM framework is proposed.</td>
</tr>

<tr>
<td></td>
<td>

Bowman et al. [[paper](https://ieeexplore.ieee.org/abstract/document/7989203)]</td>
<td>An optimization problem for sensor state and semantic landmark location is proposed.
</td>
</tr>
</table>

CNN has achieved good results in replacing some modules of the traditional VSLAM algorithm, such as depth estimation and loop closure detection. Its stability is still not as good as the traditional VSLAM algorithm . In contrast, the semantic information extraction of the CNN system has brought better effects. The process of traditional VSLAM is optimized by using CNN to extract the semantic information of the environment with higher-level features, making the traditional VSLAM achieve better results. Using a neural network to extract semantic information and combining it with VSLAM will be an area of great interest. With the help of semantic information, the data association is upgraded from the traditional pixel level to the object level. The perceptual geometric environment information is assigned with semantic labels to obtain a high-level semantic map. It can help the robot to understand the autonomous environment and human–computer interaction.


#### +  RNN with VSLAM:



<table width=100%>
<tr>
<th>Part</th>
<th>Method</th>
<th>Contribution</th>
</tr>

<tr>
<td>VO</td>
<td>

Xue et al. [[paper](https://link.springer.com/chapter/10.1007/978-3-030-20876-9_19)]
</td>
<td>Proposing a dual-branch recurrent network to learn the rotation and translation separately by leveraging current CNN for feature representation and RNN for image sequence reasoning.</td>
</tr>


<tr>
<td></td>
<td>

Teed et al. [[paper](https://proceedings.neurips.cc/paper/2021/hash/89fcd07f20b6785b92134bd6c1d0fa42-Abstract.html)]
</td>
<td>It consists of recurrent iterative updates of camera pose and pixel-wise depth through a Dense Bundle Adjustment layer.</td>
</tr>


<tr>
<td></td>
<td>

DA-RNN [[paper](https://arxiv.org/abs/1703.03098)]
 </td>
<td>A novel framework for joint 3D scene mapping and semantic labeling.</td>
</tr>


<tr>
<td></td>
<td>

DeepSeqSLAM [[paper](https://arxiv.org/abs/2011.08518)]</td>
<td>A trainable CNN+RNN architecture for jointly learning visual and positional representations from a single monocular image sequence of a route.</td>
</tr>

<hr>


<tr>
<td>VIO</td>
<td>

Clark et al.  [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/11215)]</td>
<td>It is the first end-to-end trainable method for visual-inertial odometry which performs a fusion of the data at an intermediate feature-representation level.</td>
</tr>

<tr>
<td></td>
<td>

DeepVIO [[paper](https://ieeexplore.ieee.org/abstract/document/8968467)]</td>
<td>It reduces the impacts of inaccurate Camera-IMU calibrations and unsynchronized and missing data.</td>
</tr>

<tr>
<td></td>
<td>

Chen et al. [[paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Selective_Sensor_Fusion_for_Neural_Visual-Inertial_Odometry_CVPR_2019_paper.html)]</td>
<td>It proposes a novel end-to-end selective sensor fusion framework for monocular VIO.</td>
</tr>

<tr>
<td></td>
<td>

Yasin et al. [[paper](https://www.sciencedirect.com/science/article/pii/S0893608022000752)]</td>
<td>Using adversarial training and self-adaptive visual-inertial sensor fusion.</td>
</tr>

<tr>
<td></td>
<td>

Wong et al. [[paper](https://ieeexplore.ieee.org/abstract/document/8972600)]</td>
<td>The fusion method of visual inertia + depth data set is proposed for the first time to further enhance the complementary advantages of visual and inertial sensors.</td>
</tr>
</table>

An excellent algorithm combining neural networks with VSLAM :

<table width=100%>
<tr>
<th></th>
<th>Method</th>
<th>Year</th>
<th>Sensor</th>
<th>Neural Network</th>
<th>Supervision</th>
</tr>

<tr>
<td>VO</td>
<td>

CNN-SLAM [[paper](https://arxiv.org/abs/1704.03489)] </td>
<td>2017</td>
<td>Monocular</td>
<td>CNN</td>
<td>Supervised</td>
</tr>


<tr>
<td></td>
<td>

DeepVo [[paper](https://ieeexplore.ieee.org/abstract/document/7989236)] </td>
<td>2017</td>
<td>Monocular</td>
<td>R-CNN</td>
<td>Supervised</td>
</tr>


<tr>
<td></td>
<td>

Code-SLAM [[paper](https://openaccess.thecvf.com/content_cvpr_2018/html/Bloesch_CodeSLAM_--_Learning_CVPR_2018_paper.html)] </td>
<td>2018</td>
<td>Monocular</td>
<td>U-Net</td>
<td>Supervised</td>
</tr>


<tr>
<td></td>
<td>

DVSO [[paper](https://openaccess.thecvf.com/content_ECCV_2018/html/Nan_Yang_Deep_Virtual_Stereo_ECCV_2018_paper.html)] </td>
<td>2018</td>
<td>Stereo</td>
<td>DispNet</td>
<td>Semi - Supervised</td>
</tr>


<tr>
<td></td>
<td>

UnDeepVo [[paper](https://ieeexplore.ieee.org/abstract/document/8461251)] </td>
<td>2018</td>
<td>Monocular</td>
<td>VGG encoder-decoder	</td>
<td>Unsupervised</td>
</tr>


<tr>
<td></td>
<td>

CNN-SVO [[paper](https://ieeexplore.ieee.org/abstract/document/8794425)] </td>
<td>2019</td>
<td>Monocular</td>
<td>CNN</td>
<td>Hybrid</td>
</tr>


<tr>
<td></td>
<td>

GANVO [[paper](https://ieeexplore.ieee.org/abstract/document/8793512)] </td>
<td>2019</td>
<td>Monocular</td>
<td>GAN</td>
<td>Unsupervised</td>
</tr>


<tr>
<td></td>
<td>

Li et al. [[paper](https://ieeexplore.ieee.org/abstract/document/8793706)] </td>
<td>2019</td>
<td>Monocular</td>
<td>CNN</td>
<td>Supervised</td>
</tr>


<tr>
<td></td>
<td>

D3VO [[paper](https://openaccess.thecvf.com/content_CVPR_2020/html/Yang_D3VO_Deep_Depth_Deep_Pose_and_Deep_Uncertainty_for_Monocular_CVPR_2020_paper.html)] </td>
<td>2020</td>
<td>Monocular</td>
<td>CNN</td>
<td>Hybrid</td>
</tr>


<tr>
<td></td>
<td>

DeepSeqSLAM [[paper](https://arxiv.org/abs/2011.08518)] </td>
<td>2020</td>
<td>Monocular</td>
<td>CNN+RNN</td>
<td>Supervised</td>
</tr>


<tr>
<td></td>
<td>

DeepSLAM [[paper](https://ieeexplore.ieee.org/abstract/document/9047170)] </td>
<td>2021</td>
<td>Monocular</td>
<td>R-CNN</td>
<td>Unsupervised</td>
</tr>


<tr>
<td></td>
<td>

LIFT-SLAM [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0925231221007803)] </td>
<td>2021</td>
<td>Monocular</td>
<td>DNN</td>
<td>Supervised</td>
</tr>


<tr>
<td></td>
<td>

Zhang et al. [[paper](https://www.mdpi.com/1424-8220/21/14/4735)] </td>
<td>2021</td>
<td>Stereo</td>
<td>U-Net encoder-decoder</td>
<td>Unsupervised</td>
</tr>

<tr>
<td>VIO</td>
<td>

VINet [[paper](https://ojs.aaai.org/index.php/AAAI/article/view/11215)] </td>
<td>2017</td>
<td>Monocular + IMU	</td>
<td>CNN + LSTM</td>
<td>Supervised</td>
</tr>


<tr>
<td></td>
<td>

VIOLearner [[paper](https://ieeexplore.ieee.org/abstract/document/8691513)] </td>
<td>2020</td>
<td>Monocular + IMU	</td>
<td>CNN</td>
<td>Unsupervised</td>
</tr>


<tr>
<td></td>
<td>

DeepVIO [[paper](https://ieeexplore.ieee.org/abstract/document/8968467)] </td>
<td>2019</td>
<td>Stereo + IMU</td>
<td>CNN + LSTM</td>
<td>Supervised</td>
</tr>


<tr>
<td></td>
<td>

Chen et al [[paper](https://openaccess.thecvf.com/content_CVPR_2019/html/Chen_Selective_Sensor_Fusion_for_Neural_Visual-Inertial_Odometry_CVPR_2019_paper.html)] </td>
<td>2019</td>
<td>Monocular + IMU	</td>
<td>FlowNet + LSTM</td>
<td>Supervised</td>
</tr>


<tr>
<td></td>
<td>

Kim et al. [[paper](https://ieeexplore.ieee.org/abstract/document/9324985)] </td>
<td>2021</td>
<td>Monocular + IMU	</td>
<td>CNN + LSTM</td>
<td>Unsupervised</td>
</tr>


<tr>
<td></td>
<td>

Gurturk et al. [[paper](https://www.sciencedirect.com/science/article/abs/pii/S0263224121008198)] </td>
<td>2021</td>
<td>Monocular + IMU	</td>
<td>CNN + LSTM</td>
<td>Supervised</td>
</tr>

</table>

### Modern Semantic VSLAM:

<table width=100%>
<tr>
<th>Field</th>
<th>Model</th>
<th>Year</th>
<th>Contribution</th>
</tr>

<tr>
<td>Object detection</td>
<td>

R-CNN [[paper](https://openaccess.thecvf.com/content_cvpr_2014/html/Girshick_Rich_Feature_Hierarchies_2014_CVPR_paper.html)]</td>
<td>2014</td>
<td>The first algorithm that successfully applied deep learning to target detection.</td>
</tr>

<tr>
<td></td>
<td>

Fast R-CNN [[paper](https://arxiv.org/abs/1504.08083)]</td>
<td>2015</td>
<td>Image feature extraction is performed only once.</td>
</tr>

<tr>
<td></td>
<td>

Faster R-CNN [[paper](https://arxiv.org/abs/1506.01497)]</td>
<td>2017</td>
<td>Integrated into a network, the comprehensive performance has been greatly improved.</td>
</tr>


<tr>
<td></td>
<td>

SSD [[paper](https://link.springer.com/chapter/10.1007/978-3-319-46448-0_2)]</td>
<td>2016</td>
<td>SSD was an early incarnation of the single-phase model.</td>
</tr>


<tr>
<td></td>
<td>

YOLO [[paper](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Redmon_You_Only_Look_CVPR_2016_paper.html)]</td>
<td>2016</td>
<td>Think of detection as a regression problem, using a network to output positions and categories.</td>
</tr>


<tr>
<td></td>
<td>

YOLOv5 [[paper](https://github.com/ultralytics/yolov5)]</td>
<td>2020</td>
<td>The environment is easy to configure and model training is very fast.</td>
</tr>


<tr>
<td>Semantic segmentation</td>
<td>

FCN [[paper](https://openaccess.thecvf.com/content_cvpr_2015/html/Long_Fully_Convolutional_Networks_2015_CVPR_paper.html)]</td>
<td>2015</td>
<td>It opens the first application of a convolutional neural network in semantic segmentation.</td>
</tr>


<tr>
<td></td>
<td>

SegNet [[paper](https://ieeexplore.ieee.org/abstract/document/7803544)]</td>
<td>2017</td>
<td>A completely symmetrical structure is adopted.</td>
</tr>

<tr>
<td></td>
<td>

DeepLabv1 [[paper](https://arxiv.org/abs/1412.7062)]</td>
<td>2014</td>
<td>Atrous convolution.
</td>
</tr>

<tr>
<td></td>
<td>

DeepLabv3+ [[paper](https://github.com/Tramac/awesome-semantic-segmentation-pytorch)]</td>
<td>2018</td>
<td>Greatly reduce the number of parameters.</td>
</tr>


<tr>
<td></td>
<td>

PSPnet [[paper](https://openaccess.thecvf.com/content_cvpr_2017/html/Zhao_Pyramid_Scene_Parsing_CVPR_2017_paper.html)]</td>
<td>2017</td>
<td>A Pyramid Pooling Module can aggregate contextual information from different regions.</td>
</tr>

<tr>
<td>Instance segmentation</td>
<td>

Mask R-CNN [[paper](https://openaccess.thecvf.com/content_iccv_2017/html/He_Mask_R-CNN_ICCV_2017_paper.html)]</td>
<td>2017</td>
<td>It can not only detect the target in the image but also give a high-quality segmentation result for each target.</td>
</tr>

<tr>
<td></td>
<td>

YOLACT [[paper](https://arxiv.org/abs/1904.02689)]</td>
<td>2019</td>
<td>Based on the one-stage target detection algorithm, the overall architecture design is very lightweight and achieves good results in speed and effect.
</td>
</tr>

</table>

Please check my notes on perception for more information on image detection algorithms. @github/[autonomy-perception](https://github.com/florist-notes/aicore_s/blob/main/notes/perc.MD).

#### Semantic with Location :

Location accuracy is one of the most basic assessment standards in the SLAM system and is a precondition for mobile robots to perform many tasks. Introducing environmental semantic information can effectively improve the scale uncertainty and cumulative drift in visual SLAM localization, thus improving the localization accuracy to varying degrees.

Bowman et al. [[paper](https://ieeexplore.ieee.org/abstract/document/7989203)] proposed a sensor state estimation and semantic landmark location optimization problem, which integrates metric information, semantic information, and data association. After obtaining semantic information from target detection, they introduced the Expectation-Maximization (EM) and calculated the probability of data association according to the result of semantic classification. They successfully converted semantic SLAM into a probability problem and improved the localization accuracy of the SLAM system.

In 2018, ETH Zurich proposed VSO [[paper](https://openaccess.thecvf.com/content_ECCV_2018/html/Konstantinos-Nektarios_Lianos_VSO_Visual_Semantic_ECCV_2018_paper.html)] based on semantic information for autonomous driving scenarios. This scheme solves the problem of visual SLAM localization in the environment of outdoor lighting changes. It establishes constraints between semantic information with images and takes advantage of the advantage that semantic information is not affected by Angle of view, scale, and illumination.

In 2020, Zhao et al. [[paper](https://ieeexplore.ieee.org/abstract/document/8901910)] of Xi ’an Jiaotong University proposed a landmark visual semantic SLAM system for a large-scale outdoor environment. Its core is to combine a 3D point cloud in ORB-SLAM with semantic segmentation information in the convolutional neural network model PSPNET-101. It can build a 3D semantic map of a large-scale environment. They proposed a method to associate real landmarks with a point cloud map. It associates architectural landmarks with the semantic point cloud and associates landmarks obtained from Google Maps with a semantic 3D map for urban area navigation.

#### Semantic with Mapping :

Another key juncture of VSLAM and deep learning is the semantic map construction of SLAM, and most semantic VSLAM systems are based on this idea [[paper](https://ieeexplore.ieee.org/abstract/document/9145704)] .

<table width=100%>
<tr>
<th>Reference</th>
<th>Year</th>
<th>Sensor</th>
<th>Semantic Labelling</th>
<th>Map</th>
<th>Contribution</th>
</tr>

<tr>
<td>

Vineet et al.  [[paper](https://ieeexplore.ieee.org/abstract/document/7138983)]
</td>
<td>2015</td>
<td>S</td>
<td>Random Forest</td>
<td>Voxel</td>
<td>The first system can perform dense, large-scale, outdoor semantic reconstruction of a scene in real-time.</td>
</tr>

<tr>
<td>

Zhao et al.  [[paper](https://link.springer.com/article/10.1007/s11370-016-0201-x)]
</td>
<td>2016</td>
<td>D</td>
<td>SVM</td>
<td>Voxel</td>
<td>Use temporal information and higher-order cliques to enforce the labeling consistency for each image labeling result.</td>
</tr>


<tr>
<td>

Li et al.  [[paper](https://arxiv.org/abs/1611.04144)]
</td>
<td>2016</td>
<td>D</td>
<td>Deeplabv2</td>
<td>Voxel</td>
<td>There is no need to obtain a semantic segmentation for each frame in a sequence.</td>
</tr>

<tr>
<td>

SemanticFusion  [[paper](https://ieeexplore.ieee.org/abstract/document/7989538)]
</td>
<td>2016</td>
<td>D</td>
<td>CNN with CRF</td>
<td>Surfel</td>
<td>Allows the CNN’s semantic predictions from multiple viewpoints to be probabilistically fused into a dense semantically annotated map.</td>
</tr>

<tr>
<td>

Yang et al.  [[paper](https://ieeexplore.ieee.org/abstract/document/8202212)]
</td>
<td>2017</td>
<td>S</td>
<td>CNN with CRF</td>
<td>Grid</td>
<td>Further, optimize 3D grid labels through a novel CRF model.</td>
</tr>

<tr>
<td>

Panopticfusion  [[paper](https://ieeexplore.ieee.org/abstract/document/8967890)]
</td>
<td>2020</td>
<td>D</td>
<td>PSPNET with CRF Mask R-CNN with CRF</td>
<td>Voxel</td>
<td>A novel online volumetric semantic mapping system at the level of stuff and things.</td>
</tr>

<tr>
<td>

Kimera [[paper](https://ieeexplore.ieee.org/abstract/document/9196885)]
</td>
<td>2020</td>
<td>S + I</td>
<td>Pixel-wise</td>
<td>Mesh</td>
<td>It is modular and allows replacing each module or executing them in isolation.</td>
</tr>

<tr>
<td>

AVP-SLAM  [[paper](https://ieeexplore.ieee.org/abstract/document/9340939)]
</td>
<td>2020</td>
<td>M + I + E</td>
<td>U-Net</td>
<td>Voxel</td>
<td>Autonomous parking.</td>
</tr>

<tr>
<td>

RoadMap  [[paper](https://ieeexplore.ieee.org/abstract/document/9561663)]
</td>
<td>2021</td>
<td>R + M + I + E	</td>
<td>CNN</td>
<td>Voxel</td>
<td>A framework of on-vehicle mapping, on-cloud maintenance, and user-end localization.</td>
</tr>
</table>

Sensor: <b>S</b> represents Stereo camera; <b>M</b> represents Monocular camera; <b>I</b> represents IMU; <b>E</b> represents encoder; <b>R</b> represents RTK-GPS and <b>D</b> represents RGB-D camera.

#### Elimination of Dynamic Objects :

Traditional VSLAM algorithms assume that objects in the environment are static or low-motion, which affects the applicability of the VSLAM system in actual scenes. When dynamic objects exist in the environment (such as people, vehicles and pets), they will bring wrong observation data to the system and reduce the accuracy and robustness of the system. Traditional methods solve the influence of some outliers on the system through the RANSAC algorithm. However, if dynamic objects occupy most of the image area or moving objects are fast, reliable observation data still cannot be obtained.

Now, the solutions to the problem of disturbance brought by dynamic objects to the SLAM system are consistent. That is, before the visual odometer, using target detection and image segmentation algorithm to filter out the dynamic areas in the image. Then use static environment points to calculate the nearby positions of the camera and construct a map containing semantic information. Figure 24 shows a typical structure. Although the influence of dynamic objects cannot be completely solved, the robustness of the system is greatly improved.



<table width=100%>
<tr>
<th>Model</th>
<th>Year</th>
<th>Sensor</th>
<th>Scene</th>
<th>Dynamic Detection</th>
<th>Dataset</th>
<th>Code Resource</th>
</tr>

<tr>
<td>

Reddy et al.  [[paper](https://ieeexplore.ieee.org/abstract/document/7759663)]
</td>
<td>2016</td>
<td>Stereo</td>
<td>Outdoor</td>
<td>

[model](https://ieeexplore.ieee.org/abstract/document/5940558) </td>
<td>KITTI</td>
<td>0</td>
</tr>

<tr>
<td>

DynaSLAM [[paper](https://ieeexplore.ieee.org/abstract/document/8421015)]
</td>
<td>2018</td>
<td>Monocular/Stereo/ RGB-D</td>
<td>Outdoor/Indoor</td>
<td>Mask R-CNN</td>
<td>KITTI/TUM RGB-D</td>
<td>

[[code](https://github.com/BertaBescos/DynaSLAM)]
</td>
</tr>

<tr>
<td>

DS-SLAM [[paper](https://ieeexplore.ieee.org/abstract/document/8593691)]
</td>
<td>2018</td>
<td>RGB-D</td>
<td>Indoor</td>
<td>SegNet</td>
<td>TUM RGB-D</td>
<td>

[[code](https://github.com/ivipsourcecode/DS-SLAM)]
</td>
</tr>

<tr>
<td>

Detect-SLAM [[paper](https://ieeexplore.ieee.org/abstract/document/8354219)]
</td>
<td>2018</td>
<td>RGB-D</td>
<td>Indoor</td>
<td>SSD</td>
<td>TUM RGB-D</td>
<td>

[[code](https://github.com/liadbiz/detect-slam)]
</td>
</tr>

<tr>
<td>

Wang et al. [[paper](https://www.mdpi.com/2072-4292/11/11/1363)]
</td>
<td>2019</td>
<td>RGB-D</td>
<td>Indoor</td>
<td>YOLOv3</td>
<td>NYU Depth Dataset V2</td>
<td>0</td>
</tr>

<tr>
<td>

SLAMANTIC [[paper](https://openaccess.thecvf.com/content_ICCVW_2019/html/DL4VSLAM/Schorghuber_SLAMANTIC_-_Leveraging_Semantics_to_Improve_VSLAM_in_Dynamic_Environments_ICCVW_2019_paper.html)]
</td>
<td>2019</td>
<td>Monocular/Stereo</td>
<td>Outdoor</td>
<td>Mask R-CNN</td>
<td>TUM RGB-D/ VKITTI</td>
<td>

[[code](https://github.com/mthz/slamantic)]
</td>
</tr>

<tr>
<td>

DynSLAM [[paper](https://ieeexplore.ieee.org/abstract/document/8462974)]
</td>
<td>2018</td>
<td>Stereo</td>
<td>Outdoor</td>
<td>Cascades 

[[ref](https://openaccess.thecvf.com/content_cvpr_2016/html/Dai_Instance-Aware_Semantic_Segmentation_CVPR_2016_paper.html)]
</td>
<td>KITTI</td>
<td>

[[code](https://github.com/AndreiBarsan/DynSLAM)]
</td>
</tr>

<tr>
<td>

STDyn-SLAM [[paper](https://ieeexplore.ieee.org/abstract/document/9706470)]
</td>
<td>2022</td>
<td>Stereo</td>
<td>Outdoor</td>
<td>SegNet</td>
<td>KITTI</td>
<td>

[[code](https://github.com/DanielaEsparza/STDyn-SLAM)]
</td>
</tr>

<tr>
<td>

PoseFusion [[paper](https://link.springer.com/chapter/10.1007/978-3-030-33950-0_66)]
</td>
<td>2018</td>
<td>RGB-D</td>
<td>Indoor</td>
<td>OpenPose</td>
<td>Freiburg RGB-D SLAM</td>
<td>

[[code](https://github.com/conix-center/posefusion)]
</td>
</tr>

<tr>
<td>

RDS-SLAM [[paper](https://ieeexplore.ieee.org/abstract/document/9318990)]
</td>
<td>2021</td>
<td>RGB-D</td>
<td>Indoor</td>
<td>SegNet/Mask R-CNN</td>
<td>TUM RGB-D</td>
<td>

[[code](https://github.com/yubaoliu/RDS-SLAM)]
</td>
</tr>

<tr>
<td>

YO-SLAM [[paper](https://ieeexplore.ieee.org/abstract/document/9445920)]
</td>
<td>2021</td>
<td>RGB-D</td>
<td>Indoor</td>
<td>Yolact</td>
<td>TUM RGB-D</td>
<td>0</td>
</tr>

<tr>
<td>

Zhang et al. [[paper](https://www.mdpi.com/1424-8220/21/17/5889)]
</td>
<td>2021</td>
<td>Panoramic</td>
<td>Data</td>
<td>Yolact</td>
<td>

[[ref](https://ieeexplore.ieee.org/abstract/document/6942637)]
</td>
<td>0</td>
</tr>

<tr>
<td>

DOE-SLAM [[paper](https://www.mdpi.com/1424-8220/21/9/3091)]
</td>
<td>2021</td>
<td>Monocular</td>
<td>Indoor</td>
<td>self-initiated *</td>
<td>TUM RGB-D</td>
<td>0</td>
</tr>

<tr>
<td>

DRSO-SLAM [[paper](https://ieeexplore.ieee.org/abstract/document/9602705)]
</td>
<td>2021</td>
<td>RGB-D</td>
<td>Indoor</td>
<td>Mask R-CNN</td>
<td>TUM RGB-D</td>
<td>0</td>
</tr>

<tr>
<td>

DDL-SLAM [[paper](https://ieeexplore.ieee.org/abstract/document/9082634)]
</td>
<td>2020</td>
<td>RGB-D</td>
<td>Indoor</td>
<td>DUNet</td>
<td>TUM RGB-D</td>
<td>0</td>
</tr>

<tr>
<td>

RDMO-SLAM [[paper](https://ieeexplore.ieee.org/abstract/document/9497091)]
</td>
<td>2021</td>
<td>RGB-D</td>
<td>Indoor</td>
<td>Mask R-CNN</td>
<td>TUM RGB-D</td>
<td>0</td>
</tr>

</table>

This visual slam roadmap is amazing and standsd out. @github/[[visual-slam-roadmap](https://github.com/changh95/visual-slam-roadmap)]

