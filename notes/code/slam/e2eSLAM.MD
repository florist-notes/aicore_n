# End-to-End SLAM System :

Knowing the principles of bricks and cement does not mean that we can build a
grand palace. We will implement a simplified version of stereo VO and then see its running effect in the [Kitti dataset](https://www.cvlibs.net/datasets/kitti/eval_odometry.php). This VO consists of a frontend of optical flow tracking and a backend of sliding window BA.

### Visual Odometry (VO) Frontend:

+ <b>Optical Flow Tracking:</b> The frontend of this SLAM system primarily relies on optical flow tracking. Optical flow is a computer vision technique that estimates the motion of objects in an image by tracking the displacement of pixels between consecutive frames.

+ <b>Feature Tracking:</b> The VO frontend identifies and tracks distinct visual features or keypoints in the scene. These features could include corners, edges, or other salient points that are easily distinguishable in the images.

+ <b>Relative Pose Estimation:</b> By tracking these features across frames, the VO frontend calculates the relative motion (translation and rotation) of the camera between frames. This information is used to estimate the camera's trajectory and, consequently, the robot's pose as it moves through the environment.

### Sliding Window Bundle Adjustment (BA) Backend:

+ <b>Bundle Adjustment:</b> Bundle Adjustment is a technique used in the backend of SLAM systems to optimize the entire trajectory and the 3D positions of landmarks simultaneously. It refines the camera poses and landmark positions by minimizing the re-projection error between observed keypoints and their estimated positions in 3D space.

+ <b>Sliding Window Approach:</b> In this hybrid SLAM system, a sliding window BA is employed. Instead of optimizing the entire trajectory, a fixed-size window of recent camera poses and landmarks is considered. This sliding window limits computational complexity while still providing global consistency in the map.

Goal :
+ Implement a stereo visual SLAM from scratch.
+ Understand the problems that are prone to occur in VO and how to fix them.

Framework of the system:

1. `bin` stores the compiled binary file;
2. `include/myslam` stores the header files of the SLAM module.
3. `src` stores the source code files, mainly .cpp files.
4. `test` stores the files used for testing, which are also .cpp files.
5. `config` stores the configuration files.
6. `cmake_modules` saves the cmake files of third-party libraries, which are used by
libraries such as g2o.


We just determined the location of the code file.

+ The most basic unit we deal with is the image. In stereo VO, that is a pair of
images. We might as well call it a frame.
+ We will detect visual features on the frame. These features are many 2D pixels.
+ If we see a feature multiple times, we use the triangulation method to calculate its 3D position, which forms the landmarks or map points.

### System  Pipeline :

 So overall, our program has two essential modules:

<img src="./slam_framework.png" width=100%> 

  + <b> Frontend : </b> We get an image frame from the sensor, and the frontend is responsible
for extracting the features in the image. It then performs optical flow tracking or
feature matching with the previous frame and calculates the frame’s position based
on the optical flow result. If necessary, new feature points should be added and
triangulated. The result of the frontend processing will be used as the initial value
of the backend optimization.
  + <b> Backend </b> The backend is a slower thread. It gets the processed keyframes and landmark points, optimizes them, and then returns the optimized results. The backend
should control the optimization problem’s scale within a certain range and cannot
keep growing over time.

We put a map module between the front and backends to handle the data flow between them. Since the front and backends process data in separate threads, the interaction process should be: (1)
The frontend adds new data to the map after finding a new keyframe. (2) When the
backend detects that the map has new data, it runs an optimization routine and then
resets the map scale. The old keyframes and map points are removed if necessary.


We have determined the general system flow, which will help the subsequent coding realization. Of course, in addition to the core algorithm, we also need some small peripheral modules to make the system more convenient, such as:

  + We should have a camera class to manage the intrinsic and extrinsics as well as the projection functions.
  + We need a configuration file management class to facilitate reading content from configuration files. Some critical parameters can be store in the configuration file for quick debugging;
  + Because the algorithm runs on the Kitti dataset, we need to read the image data according to Kitti’s storage format, which should also be handled by a separate class.
  + We need a visualization module to observe the running status of the system. Otherwise, we have to scratch our heads against a series of numeric values.


## Implement the Basic Data Structure :

The frame struct is : 

```cpp

#pragma once

#ifndef MYSLAM_FRAME_H
#define MYSLAM_FRAME_H

#include "myslam/camera.h"
#include "myslam/common_include.h"

namespace myslam {

// forward declare
struct MapPoint;
struct Feature;

/**
  * frame
  * Each frame is assigned an independent id, and the key frame is assigned a key frame ID
  */
struct Frame {
    public:
     EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
     typedef std::shared_ptr<Frame> Ptr;

     unsigned long id_ = 0; // id of this frame
     unsigned long keyframe_id_ = 0; // id of keyframe
     bool is_keyframe_ = false; // Whether it is a keyframe
     double time_stamp_; // Timestamp, not used yet
     SE3 pose_; // Tcw form Pose
     std::mutex pose_mutex_; // Pose data lock
     cv::Mat left_img_, right_img_; // stereo images

     // extracted features in left image
     std::vector<std::shared_ptr<Feature>> features_left_;
     // corresponding features in right image, set to nullptr if no corresponding
     std::vector<std::shared_ptr<Feature>> features_right_;

    public: // data members
     Frame() {}

     Frame(long id, double time_stamp, const SE3 &pose, const Mat &left,
           const Mat &right);

     // set and get pose, thread safe
     SE3 Pose() {
         std::unique_lock<std::mutex> lck(pose_mutex_);
         return pose_;
     }

     void SetPose(const SE3 &pose) {
         std::unique_lock<std::mutex> lck(pose_mutex_);
         pose_ = pose;
     }

     /// Set the keyframe and assign and keyframe id
     void SetKeyFrame();

     /// Factory construction mode, assign id
     static std::shared_ptr<Frame> CreateFrame();
};

} // namespace myslam

#endif // MYSLAM_FRAME_H

```

The feature struct:

```cpp

#pragma once

#ifndef MYSLAM_FEATURE_H
#define MYSLAM_FEATURE_H

#include <memory>
#include <opencv2/features2d.hpp>
#include "myslam/common_include.h"

namespace myslam {

struct Frame;
struct MapPoint;

/**
  * 2D feature points
  * will be associated with a map point after triangulation
  */
struct Feature {
    public:
     EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
     typedef std::shared_ptr<Feature> Ptr;

     std::weak_ptr<Frame> frame_; // frame holding this feature
     cv::KeyPoint position_; // 2D extraction position
     std::weak_ptr<MapPoint> map_point_; // associated map point

     bool is_outlier_ = false; // Whether it is an outlier point
     bool is_on_left_image_ = true; // Whether the logo is on the left image, false is the right image

    public:
     Feature() {}

     Feature(std::shared_ptr<Frame> frame, const cv::KeyPoint &kp)
         : frame_(frame), position_(kp) {}
};
} // namespace myslam

#endif // MYSLAM_FEATURE_H

```

Finally the map point, or the landmark:

```cpp

#pragma once
#ifndef MYSLAM_MAPPOINT_H
#define MYSLAM_MAPPOINT_H

#include "myslam/common_include.h"

namespace myslam {
struct Frame;
struct Feature;

/**
  * Waypoint class
  * Feature points form landmark points after triangulation
  */
struct MapPoint {
    public:
     EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
     typedef std::shared_ptr<MapPoint> Ptr;
     unsigned long id_ = 0; //ID
     bool is_outlier_ = false;
     Vec3 pos_ = Vec3::Zero(); // Position in world
     std::mutex data_mutex_;
     int observed_times_ = 0; // being observed by feature matching algo.
     std::list<std::weak_ptr<Feature>> observations_;

     MapPoint() {}

     MapPoint(long id, Vec3 position);

     Vec3 Pos() {
         std::unique_lock<std::mutex> lck(data_mutex_);
         return pos_;
     }

     void SetPos(const Vec3 &pos) {
         std::unique_lock<std::mutex> lck(data_mutex_);
         pos_ = pos;
     };

     void AddObservation(std::shared_ptr<Feature> feature) {
         std::unique_lock<std::mutex> lck(data_mutex_);
         observations_.push_back(feature);
         observed_times_++;
     }

     void RemoveObservation(std::shared_ptr<Feature> feat);

     std::list<std::weak_ptr<Feature>> GetObs() {
         std::unique_lock<std::mutex> lck(data_mutex_);
         return observations_;
     }

     // factory function
     static MapPoint::Ptr CreateNewMappoint();
};
} // namespace myslam

#endif // MYSLAM_MAPPOINT_H

```

The most important thing about MapPoint is its 3D position, which is the `pos_`
variable, also needs to be locked. Its `observation_` variable records the features that
observed this map point. Because the feature may be judged as an outlier, it needs to
be locked when the observation part is changed.
So far, we have realized the basic data structure. In the framework, we let the map
class actually hold these frame and map point objects, so we also need to define a
map class:

```cpp
#pragma once
#ifndef MAP_H
#define MAP_H

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/mappoint.h"

namespace myslam {

/**
  * @brief map
  * Interaction with the map: the front end calls InsertKeyframe and InsertMapPoint to insert new frames and map points, the back end maintains the structure of the map, determines outlier/elimination, etc.
  */
class Map {
    public:
     EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
     typedef std::shared_ptr<Map> Ptr;
     typedef std::unordered_map<unsigned long, MapPoint::Ptr> LandmarksType;
     typedef std::unordered_map<unsigned long, Frame::Ptr> KeyframesType;

     Map() {}

     /// Add a keyframe
     void InsertKeyFrame(Frame::Ptr frame);
     /// Add a map vertex
     void InsertMapPoint(MapPoint::Ptr map_point);

     /// Get all map points
     LandmarksType GetAllMapPoints() {
         std::unique_lock<std::mutex> lck(data_mutex_);
         return landmarks_;
     }
     /// Get all keyframes
     KeyframesType GetAllKeyFrames() {
         std::unique_lock<std::mutex> lck(data_mutex_);
         return keyframes_;
     }

     /// Get the active map point
     LandmarksType GetActiveMapPoints() {
         std::unique_lock<std::mutex> lck(data_mutex_);
         return active_landmarks_;
     }

     /// Get the active keyframe
     KeyframesType GetActiveKeyFrames() {
         std::unique_lock<std::mutex> lck(data_mutex_);
         return active_keyframes_;
     }

     /// Clear the points in the map where the number of observations is zero
     void CleanMap();

    private:
     // set the old keyframe as inactive
     void RemoveOldKeyframe();

     std::mutex data_mutex_;
     LandmarksType landmarks_; // all landmarks
     LandmarksType active_landmarks_; // active landmarks
     KeyframesType keyframes_; // all key-frames
     KeyframesType active_keyframes_; // all key-frames

     Frame::Ptr current_frame_ = nullptr;

     // settings
     int num_active_keyframes_ = 7; // number of active keyframes
};
} // namespace myslam

#endif // MAP_H

```

The map stores all keyframes and corresponding landmarks in a hash form and
maintains an activated keyframe and map point set. Here the concept of activation
is what we call the sliding window before. The backend will optimize the activated
keyframes and landmark points and fix the rest to control the optimization scale.

## Implement the Frontend :

For simplicity, we first determine the frontend processing logic:
1. The frontend has three states: initialization, normal tracking, and tracking lost.
2. In the initialization state, we do the triangulation according to the optical flow
matching between the left and right eyes. We will establish the initial map when
successful.
3. In the tracking phase, the front end calculates the optical flow from the previous
frame to the current frame and estimates the image pose based on the optical flow
result. This optical flow is used only for the left eye image to save the computation
resource.
4. If the tracked features are fewer than a threshold, we set the current frame as a
keyframe. For keyframes, do the following things:
• Extract new feature points;
• Find the corresponding points of these points on the right, and use triangulation
to create new landmarks;
• Add new keyframes and landmarks to the map and trigger a backend optimization.
• If the tracking is lost, reset the frontend system and reinitialize it.
According to this logic, the frontend processing flow is written as follows:

```cpp

#include <opencv2/opencv.hpp>

#include "myslam/algorithm.h"
#include "myslam/backend.h"
#include "myslam/config.h"
#include "myslam/feature.h"
#include "myslam/frontend.h"
#include "myslam/g2o_types.h"
#include "myslam/map.h"
#include "myslam/viewer.h"

namespace myslam {

Frontend::Frontend() {
     gftt_ =
         cv::GFTTDetector::create(Config::Get<int>("num_features"), 0.01, 20);
     num_features_init_ = Config::Get<int>("num_features_init");
     num_features_ = Config::Get<int>("num_features");
}

bool Frontend::AddFrame(myslam::Frame::Ptr frame) {
     current_frame_ = frame;

     switch (status_) {
         case FrontendStatus::INITING:
             StereoInit();
             break;
         case FrontendStatus::TRACKING_GOOD:
         case FrontendStatus::TRACKING_BAD:
             Track();
             break;
         case FrontendStatus::LOST:
             Reset();
             break;
     }

     last_frame_ = current_frame_;
     return true;
}

bool Frontend::Track() {
     if (last_frame_) {
         current_frame_->SetPose(relative_motion_ * last_frame_->Pose());
     }

     int num_track_last = TrackLastFrame();
     tracking_inliers_ = EstimateCurrentPose();

     if (tracking_inliers_ > num_features_tracking_) {
         // tracking good
         status_ = FrontendStatus::TRACKING_GOOD;
     } else if (tracking_inliers_ > num_features_tracking_bad_) {
         // tracking bad
         status_ = FrontendStatus::TRACKING_BAD;
     } else {
         // lost
         status_ = FrontendStatus::LOST;
     }

     InsertKeyframe();
     relative_motion_ = current_frame_->Pose() * last_frame_->Pose().inverse();

     if (viewer_) viewer_->AddCurrentFrame(current_frame_);
     return true;
}

bool Frontend::InsertKeyframe() {
     if (tracking_inliers_ >= num_features_needed_for_keyframe_) {
         // still have enough features, don't insert keyframe
         return false;
     }
     // current frame is a new keyframe
     current_frame_->SetKeyFrame();
     map_->InsertKeyFrame(current_frame_);

     LOG(INFO) << "Set frame " << current_frame_->id_ << " as keyframe "
               << current_frame_->keyframe_id_;

     SetObservationsForKeyFrame();
     DetectFeatures(); // detect new features

     // track in right image
     FindFeaturesInRight();
     // triangulate map points
     TriangulateNewPoints();
     // update backend because we have a new keyframe
     backend_->UpdateMap();

     if (viewer_) viewer_->UpdateMap();

     return true;
}

void Frontend::SetObservationsForKeyFrame() {
     for (auto &feat : current_frame_->features_left_) {
         auto mp = feat->map_point_.lock();
         if (mp) mp->AddObservation(feat);
     }
}

int Frontend::TriangulateNewPoints() {
     std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};
     SE3 current_pose_Twc = current_frame_->Pose().inverse();
     int cnt_triangulated_pts = 0;
     for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
         if (current_frame_->features_left_[i]->map_point_.expired() &&
             current_frame_->features_right_[i] != nullptr) {
             // The feature points in the left image are not associated with map points and there are matching points in the right image, try triangulation
             std::vector<Vec3> points {
                 camera_left_->pixel2camera(
                     Vec2(current_frame_->features_left_[i]->position_.pt.x,
                          current_frame_->features_left_[i]->position_.pt.y)),
                 camera_right_->pixel2camera(
                     Vec2(current_frame_->features_right_[i]->position_.pt.x,
                          current_frame_->features_right_[i]->position_.pt.y))};
             Vec3 pworld = Vec3::Zero();

             if (triangulation(poses, points, pworld) && pworld[2] > 0) {
                 auto new_map_point = MapPoint::CreateNewMappoint();
                 pworld = current_pose_Twc * pworld;
                 new_map_point->SetPos(pworld);
                 new_map_point->AddObservation(
                     current_frame_->features_left_[i]);
                 new_map_point->AddObservation(
                     current_frame_->features_right_[i]);

                 current_frame_->features_left_[i]->map_point_ = new_map_point;
                 current_frame_->features_right_[i]->map_point_ = new_map_point;
                 map_->InsertMapPoint(new_map_point);
                 cnt_triangulated_pts++;
             }
         }
     }
     LOG(INFO) << "new landmarks: " << cnt_triangulated_pts;
     return cnt_triangulated_pts;
}

int Frontend::EstimateCurrentPose() {
     // setup g2o
     typedef g2o::BlockSolver_6_3 BlockSolverType;
     typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType>
         LinearSolverType;
     auto solver = new g2o::OptimizationAlgorithmLevenberg(
         g2o::make_unique<BlockSolverType>(
             g2o::make_unique<LinearSolverType>()));
     g2o::SparseOptimizer optimizer;
     optimizer.setAlgorithm(solver);

     //vertex
     VertexPose *vertex_pose = new VertexPose(); // camera vertex_pose
     vertex_pose->setId(0);
     vertex_pose->setEstimate(current_frame_->Pose());
     optimizer. addVertex(vertex_pose);

     //K
     Mat33 K = camera_left_->K();

     // edges
     int index = 1;
     std::vector<EdgeProjectionPoseOnly *> edges;
     std::vector<Feature::Ptr> features;
     for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
         auto mp = current_frame_->features_left_[i]->map_point_.lock();
         if (mp) {
             features.push_back(current_frame_->features_left_[i]);
             EdgeProjectionPoseOnly *edge =
                 new EdgeProjectionPoseOnly(mp->pos_, K);
             edge->setId(index);
             edge->setVertex(0, vertex_pose);
             edge->setMeasurement(
                 toVec2(current_frame_->features_left_[i]->position_.pt));
             edge->setInformation(Eigen::Matrix2d::Identity());
             edge->setRobustKernel(new g2o::RobustKernelHuber);
             edges.push_back(edge);
             optimizer. addEdge(edge);
             index++;
         }
     }

     // estimate the Pose the determine the outliers
     const double chi2_th = 5.991;
     int cnt_outlier = 0;
     for (int iteration = 0; iteration < 4; ++iteration) {
         vertex_pose->setEstimate(current_frame_->Pose());
         optimizer.initializeOptimization();
         optimizer. optimize(10);
         cnt_outlier = 0;

         // count the outliers
         for (size_t i = 0; i < edges. size(); ++i) {
             auto e = edges[i];
             if (features[i]->is_outlier_) {
                 e->computeError();
             }
             if (e->chi2() > chi2_th) {
                 features[i]->is_outlier_ = true;
                 e->setLevel(1);
                 cnt_outlier++;
             } else {
                 features[i]->is_outlier_ = false;
                 e->setLevel(0);
             };

             if (iteration == 2) {
                 e->setRobustKernel(nullptr);
             }
         }
     }

     LOG(INFO) << "Outlier/Inlier in pose estimating: " << cnt_outlier << "/"
               << features. size() - cnt_outlier;
     // Set pose and outlier
     current_frame_->SetPose(vertex_pose->estimate());

     LOG(INFO) << "Current Pose = \n" << current_frame_->Pose().matrix();

     for (auto &feat : features) {
         if (feat->is_outlier_) {
             feat->map_point_.reset();
             feat->is_outlier_ = false; // maybe we can still use it in future
         }
     }
     return features. size() - cnt_outlier;
}

int Frontend::TrackLastFrame() {
     // use LK flow to estimate points in the right image
     std::vector<cv::Point2f> kps_last, kps_current;
     for (auto &kp : last_frame_->features_left_) {
         if (kp->map_point_.lock()) {
             // use project point
             auto mp = kp->map_point_.lock();
             auto px =
                 camera_left_->world2pixel(mp->pos_, current_frame_->Pose());
             kps_last.push_back(kp->position_.pt);
             kps_current.push_back(cv::Point2f(px[0], px[1]));
         } else {
             kps_last.push_back(kp->position_.pt);
             kps_current.push_back(kp->position_.pt);
         }
     }

     std::vector<uchar> status;
     Mat error;
     cv::calcOpticalFlowPyrLK(
         last_frame_->left_img_, current_frame_->left_img_, kps_last,
         kps_current, status, error, cv::Size(11, 11), 3,
         cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                          0.01),
         cv::OPTFLOW_USE_INITIAL_FLOW);

     int num_good_pts = 0;

     for (size_t i = 0; i < status. size(); ++i) {
         if (status[i]) {
             cv::KeyPoint kp(kps_current[i], 7);
             Feature::Ptr feature(new Feature(current_frame_, kp));
             feature->map_point_ = last_frame_->features_left_[i]->map_point_;
             current_frame_->features_left_.push_back(feature);
             num_good_pts++;
         }
     }

     LOG(INFO) << "Find " << num_good_pts << " in the last image.";
     return num_good_pts;
}

bool Frontend::StereoInit() {
     int num_features_left = DetectFeatures();
     int num_coor_features = FindFeaturesInRight();
     if (num_coor_features < num_features_init_) {
         return false;
     }

     bool build_map_success = BuildInitMap();
     if (build_map_success) {
         status_ = FrontendStatus::TRACKING_GOOD;
         if (viewer_) {
             viewer_->AddCurrentFrame(current_frame_);
             viewer_->UpdateMap();
         }
         return true;
     }
     return false;
}

int Frontend::DetectFeatures() {
     cv::Mat mask(current_frame_->left_img_.size(), CV_8UC1, 255);
     for (auto &feat : current_frame_->features_left_) {
         cv::rectangle(mask, feat->position_.pt - cv::Point2f(10, 10),
                       feat->position_.pt + cv::Point2f(10, 10), 0, CV_FILLED);
     }

     std::vector<cv::KeyPoint> keypoints;
     gftt_->detect(current_frame_->left_img_, keypoints, mask);
     int cnt_detected = 0;
     for (auto &kp : keypoints) {
         current_frame_->features_left_.push_back(
             Feature::Ptr(new Feature(current_frame_, kp)));
         cnt_detected++;
     }

     LOG(INFO) << "Detect " << cnt_detected << "new features";
     return cnt_detected;
}

int Frontend::FindFeaturesInRight() {
     // use LK flow to estimate points in the right image
     std::vector<cv::Point2f> kps_left, kps_right;
     for (auto &kp : current_frame_->features_left_) {
         kps_left.push_back(kp->position_.pt);
         auto mp = kp->map_point_.lock();
         if (mp) {
             // use projected points as initial guess
             auto px =
                 camera_right_->world2pixel(mp->pos_, current_frame_->Pose());
             kps_right.push_back(cv::Point2f(px[0], px[1]));
         } else {
             // use same pixel in left iamge
             kps_right.push_back(kp->position_.pt);
         }
     }

     std::vector<uchar> status;
     Mat error;
     cv::calcOpticalFlowPyrLK(
         current_frame_->left_img_, current_frame_->right_img_, kps_left,
         kps_right, status, error, cv::Size(11, 11), 3,
         cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                          0.01),
         cv::OPTFLOW_USE_INITIAL_FLOW);

     int num_good_pts = 0;
     for (size_t i = 0; i < status. size(); ++i) {
         if (status[i]) {
             cv::KeyPoint kp(kps_right[i], 7);
             Feature::Ptr feat(new Feature(current_frame_, kp));
             feat->is_on_left_image_ = false;
             current_frame_->features_right_.push_back(feat);
             num_good_pts++;
         } else {
             current_frame_->features_right_.push_back(nullptr);
         }
     }
     LOG(INFO) << "Find " << num_good_pts << " in the right image.";
     return num_good_pts;
}

bool Frontend::BuildInitMap() {
     std::vector<SE3> poses{camera_left_->pose(), camera_right_->pose()};
     size_t cnt_init_landmarks = 0;
     for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
         if (current_frame_->features_right_[i] == nullptr) continue;
         // create map point from triangulation
         std::vector<Vec3> points {
             camera_left_->pixel2camera(
                 Vec2(current_frame_->features_left_[i]->position_.pt.x,
                      current_frame_->features_left_[i]->position_.pt.y)),
             camera_right_->pixel2camera(
                 Vec2(current_frame_->features_right_[i]->position_.pt.x,
                      current_frame_->features_right_[i]->position_.pt.y))};
         Vec3 pworld = Vec3::Zero();

         if (triangulation(poses, points, pworld) && pworld[2] > 0) {
             auto new_map_point = MapPoint::CreateNewMappoint();
             new_map_point->SetPos(pworld);
             new_map_point->AddObservation(current_frame_->features_left_[i]);
             new_map_point->AddObservation(current_frame_->features_right_[i]);
             current_frame_->features_left_[i]->map_point_ = new_map_point;
             current_frame_->features_right_[i]->map_point_ = new_map_point;
             cnt_init_landmarks++;
             map_->InsertMapPoint(new_map_point);
         }
     }
     current_frame_->SetKeyFrame();
     map_->InsertKeyFrame(current_frame_);
     backend_->UpdateMap();

     LOG(INFO) << "Initial map created with " << cnt_init_landmarks
               << "map points";

     return true;
}

bool Frontend::Reset() {
     LOG(INFO) << "Reset is not implemented.";
     return true;
}

} // namespace myslam

```

In the `TrackLastFrame()`, we call the optical flow of OpenCV to track the feature
points:

## Implement the Backend

Compared with the frontend, the logic of the backend implementation will be more
complicated. The overall backend implementation is as follows:


```cpp

#ifndef MYSLAM_BACKEND_H
#define MYSLAM_BACKEND_H

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/map.h"

namespace myslam {
class Map;

/**
  * rear end
  * There is a separate optimization thread, and the optimization is started when the Map is updated
  * Map update is triggered by the front end
  */
class Backend {
    public:
     EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
     typedef std::shared_ptr<Backend> Ptr;

     /// Start the optimization thread in the constructor and suspend
     Backend();

     // Set the left and right destination cameras to obtain internal and external parameters
     void SetCameras(Camera::Ptr left, Camera::Ptr right) {
         cam_left_ = left;
         cam_right_ = right;
     }

     /// Set the map
     void SetMap(std::shared_ptr<Map> map) { map_ = map; }

     /// Trigger map update and start optimization
     void UpdateMap();

     /// Close the backend thread
     void Stop();

    private:
     /// Backend thread
     void BackendLoop();

     /// Optimize the given keyframe and waypoint
     void Optimize(Map::KeyframesType& keyframes, Map::LandmarksType& landmarks);

     std::shared_ptr<Map> map_;
     std::thread backend_thread_;
     std::mutex data_mutex_;

     std::condition_variable map_update_;
     std::atomic<bool> backend_running_;

     Camera::Ptr cam_left_ = nullptr, cam_right_ = nullptr;
};

} // namespace myslam

#endif // MYSLAM_BACKEND_H

```

After the backend is started, it will wait for the condition variable of map_update_.
When the map update is triggered, take the activated keyframes and map points from
the map and perform optimization:

```cpp

#include "myslam/backend.h"
#include "myslam/algorithm.h"
#include "myslam/feature.h"
#include "myslam/g2o_types.h"
#include "myslam/map.h"
#include "myslam/mappoint.h"

namespace myslam {

Backend::Backend() {
     backend_running_.store(true);
     backend_thread_ = std::thread(std::bind(&Backend::BackendLoop, this));
}

void Backend::UpdateMap() {
     std::unique_lock<std::mutex> lock(data_mutex_);
     map_update_.notify_one();
}

void Backend::Stop() {
     backend_running_.store(false);
     map_update_.notify_one();
     backend_thread_. join();
}

void Backend::BackendLoop() {
     while (backend_running_.load()) {
         std::unique_lock<std::mutex> lock(data_mutex_);
         map_update_.wait(lock);

         /// The backend only optimizes the activated Frames and Landmarks
         Map::KeyframesType active_kfs = map_->GetActiveKeyFrames();
         Map::LandmarksType active_landmarks = map_->GetActiveMapPoints();
         Optimize(active_kfs, active_landmarks);
     }
}

void Backend::Optimize(Map::KeyframesType &keyframes,
                        Map::LandmarksType &landmarks) {
     // setup g2o
     typedef g2o::BlockSolver_6_3 BlockSolverType;
     typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType>
         LinearSolverType;
     auto solver = new g2o::OptimizationAlgorithmLevenberg(
         g2o::make_unique<BlockSolverType>(
             g2o::make_unique<LinearSolverType>()));
     g2o::SparseOptimizer optimizer;
     optimizer.setAlgorithm(solver);

     // pose vertex, use Keyframe id
     std::map<unsigned long, VertexPose *> vertices;
     unsigned long max_kf_id = 0;
     for (auto &keyframe : keyframes) {
         auto kf = keyframe.second;
         VertexPose *vertex_pose = new VertexPose(); // camera vertex_pose
         vertex_pose->setId(kf->keyframe_id_);
         vertex_pose->setEstimate(kf->Pose());
         optimizer. addVertex(vertex_pose);
         if (kf->keyframe_id_ > max_kf_id) {
             max_kf_id = kf->keyframe_id_;
         }

         vertices.insert({kf->keyframe_id_, vertex_pose});
     }

     // Signpost vertex, indexed by signpost id
     std::map<unsigned long, VertexXYZ *> vertices_landmarks;

     // K and left and right external parameters
     Mat33 K = cam_left_->K();
     SE3 left_ext = cam_left_->pose();
     SE3 right_ext = cam_right_->pose();

     // edges
     int index = 1;
     double chi2_th = 5.991; // robust kernel threshold
     std::map<EdgeProjection *, Feature::Ptr> edges_and_features;

     for (auto &landmark : landmarks) {
         if (landmark. second->is_outlier_) continue;
         unsigned long landmark_id = landmark. second->id_;
         auto observations = landmark. second->GetObs();
         for (auto &obs : observations) {
             if (obs. lock() == nullptr) continue;
             auto feat = obs. lock();
             if (feat->is_outlier_ || feat->frame_.lock() == nullptr) continue;

             auto frame = feat->frame_.lock();
             EdgeProjection *edge = nullptr;
             if (feat->is_on_left_image_) {
                 edge = new EdgeProjection(K, left_ext);
             } else {
                 edge = new EdgeProjection(K, right_ext);
             }

             // If the landmark has not been added to the optimization, add a new vertex
             if (vertices_landmarks. find(landmark_id) ==
                 vertices_landmarks. end()) {
                 VertexXYZ *v = new VertexXYZ;
                 v->setEstimate(landmark. second->Pos());
                 v->setId(landmark_id + max_kf_id + 1);
                 v->setMarginalized(true);
                 vertices_landmarks.insert({landmark_id, v});
                 optimizer. addVertex(v);
             }

             edge->setId(index);
             edge->setVertex(0, vertices.at(frame->keyframe_id_)); // pose
             edge->setVertex(1, vertices_landmarks.at(landmark_id)); // landmark
             edge->setMeasurement(toVec2(feat->position_.pt));
             edge->setInformation(Mat22::Identity());
             auto rk = new g2o::RobustKernelHuber();
             rk->setDelta(chi2_th);
             edge->setRobustKernel(rk);
             edges_and_features. insert({edge, feat});

             optimizer. addEdge(edge);

             index++;
         }
     }

     // do optimization and eliminate the outliers
    optimizer.initializeOptimization();
    optimizer.optimize(10);

    int cnt_outlier = 0, cnt_inlier = 0;
    int iteration = 0;
    while (iteration < 5) {
        cnt_outlier = 0;
        cnt_inlier = 0;
        // determine if we want to adjust the outlier threshold
        for (auto &ef : edges_and_features) {
            if (ef.first->chi2() > chi2_th) {
                cnt_outlier++;
            } else {
                cnt_inlier++;
            }
        }
        double inlier_ratio = cnt_inlier / double(cnt_inlier + cnt_outlier);
        if (inlier_ratio > 0.5) {
            break;
        } else {
            chi2_th *= 2;
            iteration++;
        }
    }

    for (auto &ef : edges_and_features) {
        if (ef.first->chi2() > chi2_th) {
            ef.second->is_outlier_ = true;
            // remove the observation
            ef.second->map_point_.lock()->RemoveObservation(ef.second);
        } else {
            ef.second->is_outlier_ = false;
        }
    }

    LOG(INFO) << "Outlier/Inlier in optimization: " << cnt_outlier << "/"
              << cnt_inlier;

    // Set pose and lanrmark position
    for (auto &v : vertices) {
        keyframes.at(v.first)->SetPose(v.second->estimate());
    }
    for (auto &v : vertices_landmarks) {
        landmarks.at(v.first)->SetPos(v.second->estimate());
    }
}

}  // namespace myslam

```



