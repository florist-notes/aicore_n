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

#### Semantic with Mapping :

Another key juncture of VSLAM and deep learning is the semantic map construction of SLAM, and most semantic VSLAM systems are based on this idea [[paper](https://ieeexplore.ieee.org/abstract/document/9145704)] .

#### Elimination of Dynamic Objects :

Traditional VSLAM algorithms assume that objects in the environment are static or low-motion, which affects the applicability of the VSLAM system in actual scenes. When dynamic objects exist in the environment (such as people, vehicles and pets), they will bring wrong observation data to the system and reduce the accuracy and robustness of the system. Traditional methods solve the influence of some outliers on the system through the RANSAC algorithm. However, if dynamic objects occupy most of the image area or moving objects are fast, reliable observation data still cannot be obtained.

Now, the solutions to the problem of disturbance brought by dynamic objects to the SLAM system are consistent. That is, before the visual odometer, using target detection and image segmentation algorithm to filter out the dynamic areas in the image. Then use static environment points to calculate the nearby positions of the camera and construct a map containing semantic information. Figure 24 shows a typical structure. Although the influence of dynamic objects cannot be completely solved, the robustness of the system is greatly improved.

