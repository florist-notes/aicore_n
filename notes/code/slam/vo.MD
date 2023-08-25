# Visual Odometry (Frontend)

Visual Odometry (VO) / Egomotion is the process of estimating the pose and motion of the camera from an image sequence, and is fundamental capability required for computer vision and autonomous navigation robots. In the field of computer vision, egomotion refers to estimating a camera's motion relative to a rigid scene.

Goal of Study
+ Study how to extract feature points from images and match feature points in
multiple images.
+ Learn the principle of epipolar geometry and use epipolar constraints to
recover the camera’s 3D motion between two images.
+ Study how to solve the PNP problem and use the known correspondence
between the 3D structure and the 2D image to solve the camera’s 3D motion.
+ Study the ICP algorithm and use the point clouds matching to estimate 3D
motion.
+ Study how to obtain the 3D structure of corresponding points on the 2D
image through triangulation.

## Feature Method :

ORB Feature : [  The FAST Key Point, BRIEF Descriptor ] ; Feature Matching . ORB is basically a fusion of FAST keypoint detector and BRIEF descriptor with many modifications to enhance the performance. First it use FAST to find keypoints, then apply Harris corner measure to find top N points among them. It also use pyramid to produce multiscale-features.

Article : [introduction to orb](https://medium.com/data-breach/introduction-to-orb-oriented-fast-and-rotated-brief-4220e8ec40cf), ORB Features from Scratch ([example.cpp](./orb_self_en.cpp)).


Feature Extraction and Matching (how to use ORB) :

```cpp

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
   if (argc != 3) {
     cout << "usage: feature_extraction img1 img2" << endl;
     return 1;
   }
   //-- read the image
   Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
   Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
   assert(img_1.data != nullptr && img_2.data != nullptr);

   //-- initialization
   std::vector<KeyPoint> keypoints_1, keypoints_2;
   Mat descriptors_1, descriptors_2;
   Ptr<FeatureDetector> detector = ORB::create();
   Ptr<DescriptorExtractor> descriptor = ORB::create();
   Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

   //-- Step 1: Detect Oriented FAST corner position
   chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
   detector->detect(img_1, keypoints_1);
   detector->detect(img_2, keypoints_2);

   //-- Step 2: Calculate the BRIEF descriptor according to the corner position
   descriptor->compute(img_1, keypoints_1, descriptors_1);
   descriptor->compute(img_2, keypoints_2, descriptors_2);
   chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
   chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
   cout << "extract ORB cost = " << time_used. count() << " seconds." << endl;

   Mat outimg1;
   drawKeypoints(img_1, keypoints_1, outimg1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
   imshow("ORB features", outimg1);

   //-- Step 3: Match the BRIEF descriptors in the two images, using the Hamming distance
   vector<DMatch> matches;
   t1 = chrono::steady_clock::now();
   matcher->match(descriptors_1, descriptors_2, matches);
   t2 = chrono::steady_clock::now();
   time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
   cout << "match ORB cost = " << time_used. count() << " seconds." << endl;

   //-- Step 4: Match point pair screening
   // Calculate the minimum and maximum distance
   auto min_max = minmax_element(matches.begin(), matches.end(),
                                 [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
   double min_dist = min_max.first->distance;
   double max_dist = min_max. second->distance;

   printf("-- Max dist : %f \n", max_dist);
   printf("-- Min dist : %f \n", min_dist);

   // When the distance between descriptors is greater than twice the minimum distance,
   // it is considered that the match is wrong. But sometimes the minimum distance will be 
   // very small, set an experience value of 30 as the lower limit.
   std::vector<DMatch> good_matches;
   for (int i = 0; i < descriptors_1.rows; i++) {
     if (matches[i].distance <= max(2 * min_dist, 30.0)) {
       good_matches.push_back(matches[i]);
     }
   }

   //-- Step 5: Draw the matching result
   Mat img_match;
   Mat img_goodmatch;
   drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match);
   drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch);
   imshow("all matches", img_match);
   imshow("good matches", img_goodmatch);
   waitKey(0);

   return 0;
}

```

Calculate the Camera Motion : [ 2D–2D: Epipolar Geometry,  Solving Camera Motion with Epipolar
Constraints ]

Epipolar geometry is the geometry of stereo vision. When two cameras view a 3D scene from two distinct positions, there are a number of geometric relations between the 3D points and their projections onto the 2D images that lead to constraints between the image points.

## Triangulation

Computer vision, as a multidisciplinary field, plays a pivotal role in numerous domains such as robotics, augmented reality, and autonomous navigation. One fundamental problem within computer vision is the reconstruction of three-dimensional (3D) scenes from two-dimensional (2D) images, a process often referred to as "triangulation." This abstract highlights the advancements and applications of triangulation techniques in computer vision.

Triangulation involves the estimation of 3D points in space by analyzing the relative positions of corresponding points in multiple 2D images. Over the years, significant progress has been made in this field, driven by advancements in camera technology, computational power, and algorithm development. This abstract delves into three key aspects of triangulation in computer vision:

+ Mathematical Foundations: Triangulation relies on principles of geometry, optics, and linear algebra. Traditional methods like the Direct Linear Transform (DLT) have been refined, and novel algorithms based on robust statistical estimation, such as RANSAC, have emerged to handle noisy data and outliers effectively.

+ Stereo Vision: Stereo vision, a popular application of triangulation, involves using a pair of calibrated cameras to reconstruct 3D scenes. Modern approaches integrate machine learning techniques, like deep neural networks, to enhance feature matching and disparity estimation, thus improving the accuracy and speed of triangulation.

```cpp

#include <iostream>
#include <opencv2/opencv.hpp>
// #include "extra.h" // used in opencv2
using namespace std;
using namespace cv;

void find_feature_matches(
   const Mat &img_1, const Mat &img_2,
   std::vector<KeyPoint> &keypoints_1,
   std::vector<KeyPoint> &keypoints_2,
   std::vector<DMatch> &matches);

void pose_estimation_2d2d(
   const std::vector<KeyPoint> &keypoints_1,
   const std::vector<KeyPoint> &keypoints_2,
   const std::vector<DMatch> &matches,
   Mat &R, Mat &t);

void triangulation(
   const vector<KeyPoint> &keypoint_1,
   const vector<KeyPoint> &keypoint_2,
   const std::vector<DMatch> &matches,
   const Mat &R, const Mat &t,
   vector<Point3d> &points
);

/// For drawing
inline cv::Scalar get_color(float depth) {
   float up_th = 50, low_th = 10, th_range = up_th - low_th;
   if (depth > up_th) depth = up_th;
   if (depth < low_th) depth = low_th;
   return cv::Scalar(255 * depth / th_range, 0, 255 * (1 - depth / th_range));
}

// Convert pixel coordinates to camera normalized coordinates
Point2f pixel2cam(const Point2d &p, const Mat &K);

int main(int argc, char **argv) {
   if (argc != 3) {
     cout << "usage: triangulation img1 img2" << endl;
     return 1;
   }
   //-- read the image
   Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
   Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);

   vector<KeyPoint> keypoints_1, keypoints_2;
   vector<DMatch> matches;
   find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
   cout << "A total of found" << matches.size() << "Group matching points" << endl;

   //-- estimate motion between two images
   Mat R, t;
   pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

   //-- triangulation
   vector<Point3d> points;
   triangulation(keypoints_1, keypoints_2, matches, R, t, points);

   //-- Verify the reprojection relationship between the triangulated point and the feature point
   Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
   Mat img1_plot = img_1. clone();
   Mat img2_plot = img_2. clone();
   for (int i = 0; i < matches. size(); i++) {
     // first image
     float depth1 = points[i].z;
     cout << "depth: " << depth1 << endl;
     Point2d pt1_cam = pixel2cam(keypoints_1[matches[i].queryIdx].pt, K);
     cv::circle(img1_plot, keypoints_1[matches[i].queryIdx].pt, 2, get_color(depth1), 2);

     // second image
     Mat pt2_trans = R * (Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t;
     float depth2 = pt2_trans.at<double>(2, 0);
     cv::circle(img2_plot, keypoints_2[matches[i].trainIdx].pt, 2, get_color(depth2), 2);
   }
   cv::imshow("img 1", img1_plot);
   cv::imshow("img 2", img2_plot);
   cv::waitKey();

   return 0;
}

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                           std::vector<KeyPoint> &keypoints_1,
                           std::vector<KeyPoint> &keypoints_2,
                           std::vector<DMatch> &matches) {
   //-- initialization
   Mat descriptors_1, descriptors_2;
   // used in OpenCV3
   Ptr<FeatureDetector> detector = ORB::create();
   Ptr<DescriptorExtractor> descriptor = ORB::create();
   // use this if you are in OpenCV2
   // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
   // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
   Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
   //-- Step 1: Detect Oriented FAST corner position
   detector->detect(img_1, keypoints_1);
   detector->detect(img_2, keypoints_2);

   //-- Step 2: Calculate the BRIEF descriptor according to the corner position
   descriptor->compute(img_1, keypoints_1, descriptors_1);
   descriptor->compute(img_2, keypoints_2, descriptors_2);

   //-- Step 3: Match the BRIEF descriptors in the two images, using the Hamming distance
   vector<DMatch> match;
   // BFMatcher matcher ( NORM_HAMMING );
   matcher->match(descriptors_1, descriptors_2, match);

   //-- Step 4: Match point pair screening
   double min_dist = 10000, max_dist = 0;

   // Find the minimum and maximum distances between all matches, that is, the distance between
   // the most similar and least similar two sets of points
   for (int i = 0; i < descriptors_1.rows; i++) {
     double dist = match[i].distance;
     if (dist < min_dist) min_dist = dist;
     if (dist > max_dist) max_dist = dist;
   }

   printf("-- Max dist : %f \n", max_dist);
   printf("-- Min dist : %f \n", min_dist);

   // When the distance between descriptors is greater than twice the minimum distance,
   // it is considered that the match is wrong. But sometimes the minimum distance will be 
   // very small, set an experience value of 30 as the lower limit.
   for (int i = 0; i < descriptors_1.rows; i++) {
     if (match[i].distance <= max(2 * min_dist, 30.0)) {
       matches.push_back(match[i]);
     }
   }
}

void pose_estimation_2d2d(
   const std::vector<KeyPoint> &keypoints_1,
   const std::vector<KeyPoint> &keypoints_2,
   const std::vector<DMatch> &matches,
   Mat & R, Mat & t) {
   // Camera internal reference, TUM Freiburg2
   Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

   //-- Convert the matching point to the form of vector<Point2f>
   vector<Point2f> points1;
   vector<Point2f> points2;

   for (int i = 0; i < (int) matches. size(); i++) {
     points1.push_back(keypoints_1[matches[i].queryIdx].pt);
     points2.push_back(keypoints_2[matches[i].trainIdx].pt);
   }

   //-- calculate the essential matrix
   Point2d principal_point(325.1, 249.7); //Camera principal point, TUM dataset calibration value
   int focal_length = 521; //Camera focal length, TUM dataset calibration value
   Mat essential_matrix;
   essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);

   //-- Recover rotation and translation information from the essential matrix.
   recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
}

void triangulation(
   const vector<KeyPoint> &keypoint_1,
   const vector<KeyPoint> &keypoint_2,
   const std::vector<DMatch> &matches,
   const Mat &R, const Mat &t,
   vector<Point3d> &points) {
   Mat T1 = (Mat_<float>(3, 4) <<
     1, 0, 0, 0,
     0, 1, 0, 0,
     0, 0, 1, 0);
   Mat T2 = (Mat_<float>(3, 4) <<
     R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
     R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
     R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0)
   );

   Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
   vector<Point2f> pts_1, pts_2;
   for (DMatch m: matches) {
     // Convert pixel coordinates to camera coordinates
     pts_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt, K));
     pts_2.push_back(pixel2cam(keypoint_2[m.trainIdx].pt, K));
   }

   Mat pts_4d;
   cv::triangulatePoints(T1, T2, pts_1, pts_2, pts_4d);

   // Convert to non-homogeneous coordinates
   for (int i = 0; i < pts_4d.cols; i++) {
     Mat x = pts_4d.col(i);
     x /= x.at<float>(3, 0); // normalize
     Point3d p(
       x.at<float>(0, 0),
       x.at<float>(1, 0),
       x.at<float>(2, 0)
     );
     points. push_back(p);
   }
}

Point2f pixel2cam(const Point2d &p, const Mat &K) {
   return Point2f
     (
       (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
       (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
     );
}

```

## 3D–2D PnP : Perspective-n-Point

[ Direct Linear Transformation, P3P ]

Solve PnP by Minimizing the Reprojection Error : Use OpenCV EPnP to Solve the Pose :

```cpp

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <sophus/se3.hpp>
#include <chrono>

using namespace std;
using namespace cv;

void find_feature_matches(
   const Mat &img_1, const Mat &img_2,
   std::vector<KeyPoint> &keypoints_1,
   std::vector<KeyPoint> &keypoints_2,
   std::vector<DMatch> &matches);

// Convert pixel coordinates to camera normalized coordinates
Point2d pixel2cam(const Point2d &p, const Mat &K);

// BA by g2o
typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;

void bundleAdjustmentG2O(
   const VecVector3d &points_3d,
   const VecVector2d &points_2d,
   const Mat &K,
   Sophus::SE3d &pose
);

// BA by gauss-newton
void bundleAdjustmentGaussNewton(
   const VecVector3d &points_3d,
   const VecVector2d &points_2d,
   const Mat &K,
   Sophus::SE3d &pose
);

int main(int argc, char **argv) {
   if (argc != 5) {
     cout << "usage: pose_estimation_3d2d img1 img2 depth1 depth2" << endl;
     return 1;
   }
   //-- read the image
   Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
   Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);
   assert(img_1.data && img_2.data && "Can not load images!");

   vector<KeyPoint> keypoints_1, keypoints_2;
   vector<DMatch> matches;
   find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
   cout << "A total of found" << matches.size() << "Group matching points" << endl;

   // Create 3D points
   Mat d1 = imread(argv[3], CV_LOAD_IMAGE_UNCHANGED); // The depth map is a 16-bit unsigned number, a single-channel image
   Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
   vector<Point3f> pts_3d;
   vector<Point2f> pts_2d;
   for (DMatch m: matches) {
     ushort d = d1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
     if (d == 0) // bad depth
       continue;
     float dd = d / 5000.0;
     Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
     pts_3d. push_back(Point3f(p1.x * dd, p1.y * dd, dd));
     pts_2d.push_back(keypoints_2[m.trainIdx].pt);
   }

   cout << "3d-2d pairs: " << pts_3d. size() << endl;

   chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
   Mat r, t;
   solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false); 
   // call OpenCV's PnP solution, you can choose EPNP, DLS and other methods
   Mat R;
   cv::Rodrigues(r, R); 
   // r is in the form of a rotation vector, converted to a matrix using the Rodrigues formula
   chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
   chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
   cout << "solve pnp in opencv cost time: " << time_used.count() << " seconds." << endl;

   cout << "R=" << endl << R << endl;
   cout << "t=" << endl << t << endl;

   VecVector3d pts_3d_eigen;
   VecVector2d pts_2d_eigen;
   for (size_t i = 0; i < pts_3d. size(); ++i) {
     pts_3d_eigen.push_back(Eigen::Vector3d(pts_3d[i].x, pts_3d[i].y, pts_3d[i].z));
     pts_2d_eigen.push_back(Eigen::Vector2d(pts_2d[i].x, pts_2d[i].y));
   }

   cout << "calling bundle adjustment by gauss newton" << endl;
   Sophus::SE3d pose_gn;
   t1 = chrono::steady_clock::now();
   bundleAdjustmentGaussNewton(pts_3d_eigen, pts_2d_eigen, K, pose_gn);
   t2 = chrono::steady_clock::now();
   time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
   cout << "solve pnp by gauss newton cost time: " << time_used.count() << " seconds." << endl;

   cout << "calling bundle adjustment by g2o" << endl;
   Sophus::SE3d pose_g2o;
   t1 = chrono::steady_clock::now();
   bundleAdjustmentG2O(pts_3d_eigen, pts_2d_eigen, K, pose_g2o);
   t2 = chrono::steady_clock::now();
   time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
   cout << "solve pnp by g2o cost time: " << time_used. count() << " seconds." << endl;
   return 0;
}

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                           std::vector<KeyPoint> &keypoints_1,
                           std::vector<KeyPoint> &keypoints_2,
                           std::vector<DMatch> &matches) {
   //-- initialization
   Mat descriptors_1, descriptors_2;
   // used in OpenCV3
   Ptr<FeatureDetector> detector = ORB::create();
   Ptr<DescriptorExtractor> descriptor = ORB::create();
   // use this if you are in OpenCV2
   // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
   // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
   Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
   //-- Step 1: Detect Oriented FAST corner position
   detector->detect(img_1, keypoints_1);
   detector->detect(img_2, keypoints_2);

   //-- Step 2: Calculate the BRIEF descriptor according to the corner position
   descriptor->compute(img_1, keypoints_1, descriptors_1);
   descriptor->compute(img_2, keypoints_2, descriptors_2);

   //-- Step 3: Match the BRIEF descriptors in the two images, using the Hamming distance
   vector<DMatch> match;
   // BFMatcher matcher ( NORM_HAMMING );
   matcher->match(descriptors_1, descriptors_2, match);

   //-- Step 4: Match point pair screening
   double min_dist = 10000, max_dist = 0;

   //Find the minimum and maximum distances between all matches, that is, the distance 
   // between the most similar and least similar two sets of points
   for (int i = 0; i < descriptors_1.rows; i++) {
     double dist = match[i].distance;
     if (dist < min_dist) min_dist = dist;
     if (dist > max_dist) max_dist = dist;
   }

   printf("-- Max dist : %f \n", max_dist);
   printf("-- Min dist : %f \n", min_dist);

   // When the distance between descriptors is greater than twice the minimum distance, 
   // it is considered that the match is wrong. But sometimes the minimum distance will be 
   // very small, set an experience value of 30 as the lower limit.
   for (int i = 0; i < descriptors_1.rows; i++) {
     if (match[i].distance <= max(2 * min_dist, 30.0)) {
       matches.push_back(match[i]);
     }
   }
}

Point2d pixel2cam(const Point2d &p, const Mat &K) {
   return Point2d
     (
       (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
       (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
     );
}

void bundleAdjustmentGaussNewton(
   const VecVector3d &points_3d,
   const VecVector2d &points_2d,
   const Mat &K,
   Sophus::SE3d &pose) {
   typedef Eigen::Matrix<double, 6, 1> Vector6d;
   const int iterations = 10;
   double cost = 0, lastCost = 0;
   double fx = K.at<double>(0, 0);
   double fy = K.at<double>(1, 1);
   double cx = K.at<double>(0, 2);
   double cy = K.at<double>(1, 2);

   for (int iter = 0; iter < iterations; iter++) {
     Eigen::Matrix<double, 6, 6> H = Eigen::Matrix<double, 6, 6>::Zero();
     Vector6d b = Vector6d::Zero();

     cost = 0;
     // compute cost
     for (int i = 0; i < points_3d. size(); i++) {
       Eigen::Vector3d pc = pose * points_3d[i];
       double inv_z = 1.0 / pc[2];
       double inv_z2 = inv_z * inv_z;
       Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy);

       Eigen::Vector2d e = points_2d[i] - proj;

       cost += e. squaredNorm();
       Eigen::Matrix<double, 2, 6> J;
       J << -fx * inv_z,
         0,
         fx * pc[0] * inv_z2,
         fx * pc[0] * pc[1] * inv_z2,
         -fx - fx * pc[0] * pc[0] * inv_z2,
         fx * pc[1] * inv_z,
         0,
         -fy *inv_z,
         fy * pc[1] * inv_z2,
         fy + fy * pc[1] * pc[1] * inv_z2,
         -fy * pc[0] * pc[1] * inv_z2,
         -fy * pc[0] * inv_z;

       H += J.transpose() * J;
       b += -J.transpose() * e;
     }

     Vector6d dx;
     dx = H.ldlt().solve(b);

     if (isnan(dx[0])) {
       cout << "result is nan!" << endl;
       break;
     }

     if (iter > 0 && cost >= lastCost) {
       // cost increase, update is not good
       cout << "cost: " << cost << ", last cost: " << lastCost << endl;
       break;
     }

     // update your estimation
     pose = Sophus::SE3d::exp(dx) * pose;
     lastCost = cost;

     cout << "iteration" << iter << " cost=" << std::setprecision(12) << cost << endl;
     if (dx. norm() < 1e-6) {
       // converge
       break;
     }
   }

   cout << "pose by g-n: \n" << pose.matrix() << endl;
}

/// vertex and edges used in g2o ba
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
   EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

   virtual void setToOriginImpl() override {
     _estimate = Sophus::SE3d();
   }

   /// left multiplication on SE3
   virtual void oplusImpl(const double *update) override {
     Eigen::Matrix<double, 6, 1> update_eigen;
     update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
     _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
   }

   virtual bool read(istream &in) override {}

   virtual bool write(ostream &out) const override {}
};

class EdgeProjection : public g2o::BaseUnaryEdge<2, Eigen::Vector2d, VertexPose> {
public:
   EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

   EdgeProjection(const Eigen::Vector3d &pos, const Eigen::Matrix3d &K) : _pos3d(pos), _K(K) {}

   virtual void computeError() override {
     const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);
     Sophus::SE3d T = v->estimate();
     Eigen::Vector3d pos_pixel = _K * (T * _pos3d);
     pos_pixel /= pos_pixel[2];
     _error = _measurement - pos_pixel.head<2>();
   }

   virtual void linearizeOplus() override {
     const VertexPose *v = static_cast<VertexPose *> (_vertices[0]);
     Sophus::SE3d T = v->estimate();
     Eigen::Vector3d pos_cam = T * _pos3d;
     double fx = _K(0, 0);
     double fy = _K(1, 1);
     double cx = _K(0, 2);
     double cy = _K(1, 2);
     double X = pos_cam[0];
     double Y = pos_cam[1];
     double Z = pos_cam[2];
     double Z2 = Z * Z;
     _jacobianOplusXi
       << -fx / Z, 0, fx * X / Z2, fx * X * Y / Z2, -fx - fx * X * X / Z2, fx * Y / Z,
       0, -fy / Z, fy * Y / (Z * Z), fy + fy * Y * Y / Z2, -fy * X * Y / Z2, -fy * X / Z;
   }

   virtual bool read(istream &in) override {}

   virtual bool write(ostream &out) const override {}

private:
   Eigen::Vector3d_pos3d;
   Eigen::Matrix3d_K;
};

void bundleAdjustmentG2O(
   const VecVector3d &points_3d,
   const VecVector2d &points_2d,
   const Mat &K,
   Sophus::SE3d &pose) {

   // Build graph optimization, first set g2o
   typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> BlockSolverType; // pose is 6, landmark is 3
   typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; 
   // linear solver type
   // Gradient descent method, you can choose from GN, LM, DogLeg
   auto solver = new g2o::OptimizationAlgorithmGaussNewton(
     g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
   g2o::SparseOptimizer optimizer; // graph model
   optimizer.setAlgorithm(solver); // set the solver
   optimizer.setVerbose(true); // turn on debug output

   //vertex
   VertexPose *vertex_pose = new VertexPose(); // camera vertex_pose
   vertex_pose->setId(0);
   vertex_pose->setEstimate(Sophus::SE3d());
   optimizer. addVertex(vertex_pose);

   //K
   Eigen::Matrix3d K_eigen;
   K_eigen <<
           K.at<double>(0, 0), K.at<double>(0, 1), K.at<double>(0, 2),
     K.at<double>(1, 0), K.at<double>(1, 1), K.at<double>(1, 2),
     K.at<double>(2, 0), K.at<double>(2, 1), K.at<double>(2, 2);

   // edges
   int index = 1;
   for (size_t i = 0; i < points_2d. size(); ++i) {
     auto p2d = points_2d[i];
     auto p3d = points_3d[i];
     EdgeProjection *edge = new EdgeProjection(p3d, K_eigen);
     edge->setId(index);
     edge->setVertex(0, vertex_pose);
     edge->setMeasurement(p2d);
     edge->setInformation(Eigen::Matrix2d::Identity());
     optimizer. addEdge(edge);
     index++;
   }

   chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
   optimizer. setVerbose(true);
   optimizer.initializeOptimization();
   optimizer. optimize(10);
   chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
   chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
   cout << "optimization costs time: " << time_used.count() << "seconds." << endl;
   cout << "pose estimated by g2o =\n" << vertex_pose->estimate().matrix() << endl;
   pose = vertex_pose->estimate();
}

```

## 3D–3D Iterative Closest Point (ICP)

[ Using Linear Algebra (SVD), Using Non-linear Optimization ]

```cpp

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <chrono>
#include <sophus/se3.hpp>

using namespace std;
using namespace cv;

void find_feature_matches(
   const Mat &img_1, const Mat &img_2,
   std::vector<KeyPoint> &keypoints_1,
   std::vector<KeyPoint> &keypoints_2,
   std::vector<DMatch> &matches);

// Convert pixel coordinates to camera normalized coordinates
Point2d pixel2cam(const Point2d &p, const Mat &K);

void pose_estimation_3d3d(
   const vector<Point3f> &pts1,
   const vector<Point3f> &pts2,
   Mat & R, Mat & t
);

void bundleAdjustment(
   const vector<Point3f> &points_3d,
   const vector<Point3f> &points_2d,
   Mat & R, Mat & t
);

/// vertex and edges used in g2o ba
class VertexPose : public g2o::BaseVertex<6, Sophus::SE3d> {
public:
   EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

   virtual void setToOriginImpl() override {
     _estimate = Sophus::SE3d();
   }

   /// left multiplication on SE3
   virtual void oplusImpl(const double *update) override {
     Eigen::Matrix<double, 6, 1> update_eigen;
     update_eigen << update[0], update[1], update[2], update[3], update[4], update[5];
     _estimate = Sophus::SE3d::exp(update_eigen) * _estimate;
   }

   virtual bool read(istream &in) override {}

   virtual bool write(ostream &out) const override {}
};

/// g2o edge
class EdgeProjectXYZRGBDPoseOnly : public g2o::BaseUnaryEdge<3, Eigen::Vector3d, VertexPose> {
public:
   EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

   EdgeProjectXYZRGBDPoseOnly(const Eigen::Vector3d &point) : _point(point) {}

   virtual void computeError() override {
     const VertexPose *pose = static_cast<const VertexPose *> ( _vertices[0] );
     _error = _measurement-pose->estimate() * _point;
   }

   virtual void linearizeOplus() override {
     VertexPose *pose = static_cast<VertexPose *>(_vertices[0]);
     Sophus::SE3d T = pose->estimate();
     Eigen::Vector3d xyz_trans = T * _point;
     _jacobianOplusXi.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
     _jacobianOplusXi.block<3, 3>(0, 3) = Sophus::SO3d::hat(xyz_trans);
   }

   bool read(istream &in) {}

   bool write(ostream &out) const {}

protected:
   Eigen::Vector3d_point;
};

int main(int argc, char **argv) {
   if (argc != 5) {
     cout << "usage: pose_estimation_3d3d img1 img2 depth1 depth2" << endl;
     return 1;
   }
   //-- read the image
   Mat img_1 = imread(argv[1], CV_LOAD_IMAGE_COLOR);
   Mat img_2 = imread(argv[2], CV_LOAD_IMAGE_COLOR);

   vector<KeyPoint> keypoints_1, keypoints_2;
   vector<DMatch> matches;
   find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
   cout << "A total of found" << matches.size() << "Group matching points" << endl;

   // Create 3D points
   Mat depth1 = imread(argv[3], CV_LOAD_IMAGE_UNCHANGED); 
   // The depth map is a 16-bit unsigned number, a single-channel image
   Mat depth2 = imread(argv[4], CV_LOAD_IMAGE_UNCHANGED); 
   // The depth map is a 16-bit unsigned number, a single-channel image
   Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
   vector<Point3f> pts1, pts2;

   for (DMatch m: matches) {
     ushort d1 = depth1.ptr<unsigned short>(int(keypoints_1[m.queryIdx].pt.y))[int(keypoints_1[m.queryIdx].pt.x)];
     ushort d2 = depth2.ptr<unsigned short>(int(keypoints_2[m.trainIdx].pt.y))[int(keypoints_2[m.trainIdx].pt.x)];
     if (d1 == 0 || d2 == 0) // bad depth
       continue;
     Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
     Point2d p2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
     float dd1 = float(d1) / 5000.0;
     float dd2 = float(d2) / 5000.0;
     pts1.push_back(Point3f(p1.x * dd1, p1.y * dd1, dd1));
     pts2.push_back(Point3f(p2.x * dd2, p2.y * dd2, dd2));
   }

   cout << "3d-3d pairs: " << pts1. size() << endl;
   Mat R, t;
   pose_estimation_3d3d(pts1, pts2, R, t);
   cout << "ICP via SVD results: " << endl;
   cout << "R = " << R << endl;
   cout << "t = " << t << endl;
   cout << "R_inv = " << R.t() << endl;
   cout << "t_inv = " << -R.t() * t << endl;

   cout << "calling bundle adjustment" << endl;

   bundleAdjustment(pts1, pts2, R, t);

   // verify p1 = R * p2 + t
   for (int i = 0; i < 5; i++) {
     cout << "p1 = " << pts1[i] << endl;
     cout << "p2 = " << pts2[i] << endl;
     cout << "(R*p2+t) = " <<
          R * (Mat_<double>(3, 1) << pts2[i].x, pts2[i].y, pts2[i].z) + t
          << endl;
     cout << endl;
   }
}

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                           std::vector<KeyPoint> &keypoints_1,
                           std::vector<KeyPoint> &keypoints_2,
                           std::vector<DMatch> &matches) {
   //-- initialization
   Mat descriptors_1, descriptors_2;
   // used in OpenCV3
   Ptr<FeatureDetector> detector = ORB::create();
   Ptr<DescriptorExtractor> descriptor = ORB::create();
   // use this if you are in OpenCV2
   // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
   // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
   Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
   //-- Step 1: Detect Oriented FAST corner position
   detector->detect(img_1, keypoints_1);
   detector->detect(img_2, keypoints_2);

   //-- Step 2: Calculate the BRIEF descriptor according to the corner position
   descriptor->compute(img_1, keypoints_1, descriptors_1);
   descriptor->compute(img_2, keypoints_2, descriptors_2);

   //-- Step 3: Match the BRIEF descriptors in the two images, using the Hamming distance
   vector<DMatch> match;
   // BFMatcher matcher ( NORM_HAMMING );
   matcher->match(descriptors_1, descriptors_2, match);

   //-- Step 4: Match point pair screening
   double min_dist = 10000, max_dist = 0;

   // Find the minimum and maximum distances between all matches, that is, the distance between
   //  the most similar and least similar two sets of points
   for (int i = 0; i < descriptors_1.rows; i++) {
     double dist = match[i].distance;
     if (dist < min_dist) min_dist = dist;
     if (dist > max_dist) max_dist = dist;
   }

   printf("-- Max dist : %f \n", max_dist);
   printf("-- Min dist : %f \n", min_dist);

   //When the distance between descriptors is greater than twice the minimum distance, 
   // it is considered that the match is wrong. But sometimes the minimum distance will be 
   // very small, set an experience value of 30 as the lower limit.
   for (int i = 0; i < descriptors_1.rows; i++) {
     if (match[i].distance <= max(2 * min_dist, 30.0)) {
       matches.push_back(match[i]);
     }
   }
}

Point2d pixel2cam(const Point2d &p, const Mat &K) {
   return Point2d(
     (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
     (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
   );
}

void pose_estimation_3d3d(const vector<Point3f> &pts1,
                           const vector<Point3f> &pts2,
                           Mat & R, Mat & t) {
   Point3f p1, p2; // center of mass
   int N = pts1. size();
   for (int i = 0; i < N; i++) {
     p1 += pts1[i];
     p2 += pts2[i];
   }
   p1 = Point3f(Vec3f(p1) / N);
   p2 = Point3f(Vec3f(p2) / N);
   vector<Point3f> q1(N), q2(N); // remove the center
   for (int i = 0; i < N; i++) {
     q1[i] = pts1[i] - p1;
     q2[i] = pts2[i] - p2;
   }

   // compute q1*q2^T
   Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
   for (int i = 0; i < N; i++) {
     W += Eigen::Vector3d(q1[i].x, q1[i].y, q1[i].z) * Eigen::Vector3d(q2[i].x, q2[i].y, q2 [i].z).transpose();
   }
   cout << "W=" << W << endl;

   // SVD on W
   Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
   Eigen::Matrix3d U = svd.matrixU();
   Eigen::Matrix3d V = svd. matrixV();

   cout << "U=" << U << endl;
   cout << "V=" << V << endl;

   Eigen::Matrix3d R_ = U * (V.transpose());
   if (R_.determinant() < 0) {
     R_ = -R_;
   }
   Eigen::Vector3d t_ = Eigen::Vector3d(p1.x, p1.y, p1.z) - R_ * Eigen::Vector3d(p2.x, p2.y, p2.z);

   // convert to cv::Mat
   R = (Mat_<double>(3, 3) <<
     R_(0, 0), R_(0, 1), R_(0, 2),
     R_(1, 0), R_(1, 1), R_(1, 2),
     R_(2, 0), R_(2, 1), R_(2, 2)
   );
   t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}

void bundleAdjustment(
   const vector<Point3f> &pts1,
   const vector<Point3f> &pts2,
   Mat & R, Mat & t) {
   // Build graph optimization, first set g2o
   typedef g2o::BlockSolverX BlockSolverType;
   typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // linear solver type
   // Gradient descent method, you can choose from GN, LM, DogLeg
   auto solver = new g2o::OptimizationAlgorithmLevenberg(
     g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
   g2o::SparseOptimizer optimizer; // graph model
   optimizer.setAlgorithm(solver); // set the solver
   optimizer.setVerbose(true); // turn on debug output

   //vertex
   VertexPose *pose = new VertexPose(); // camera pose
   pose->setId(0);
   pose->setEstimate(Sophus::SE3d());
   optimizer. addVertex(pose);

   // edges
   for (size_t i = 0; i < pts1. size(); i++) {
     EdgeProjectXYZRGBDPoseOnly *edge = new EdgeProjectXYZRGBDPoseOnly(
       Eigen::Vector3d(pts2[i].x, pts2[i].y, pts2[i].z));
     edge->setVertex(0, pose);
     edge->setMeasurement(Eigen::Vector3d(
       pts1[i].x, pts1[i].y, pts1[i].z));
     edge->setInformation(Eigen::Matrix3d::Identity());
     optimizer. addEdge(edge);
   }

   chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
   optimizer.initializeOptimization();
   optimizer. optimize(10);
   chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
   chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
   cout << "optimization costs time: " << time_used.count() << "seconds." << endl;

   cout << endl << "after optimization:" << endl;
   cout << "T=\n" << pose->estimate().matrix() << endl;

   // convert to cv::Mat
   Eigen::Matrix3d R_ = pose->estimate().rotationMatrix();
   Eigen::Vector3d t_ = pose->estimate().translation();
   R = (Mat_<double>(3, 3) <<
     R_(0, 0), R_(0, 1), R_(0, 2),
     R_(1, 0), R_(1, 1), R_(1, 2),
     R_(2, 0), R_(2, 1), R_(2, 2)
   );
   t = (Mat_<double>(3, 1) << t_(0, 0), t_(1, 0), t_(2, 0));
}



```

resources : [Review of visual odometry: types, approaches, challenges, and applications](https://springerplus.springeropen.com/articles/10.1186/s40064-016-3573-7), @ScienceDirect/[visual-odometry](https://www.sciencedirect.com/topics/computer-science/visual-odometry), [Visual Odometry with Monocular Camera For Beginners: A Project in OpenCV](https://youtu.be/N451VeA8XRA?si=Xl-s8w9Vn9QfFcSi), [Point Cloud Processing with Open3D](https://www.youtube.com/watch?v=zF3MreN1w6c&list=PLkmvobsnE0GEZugH1Di2Cr_f32qYkv7aN), [YOLO Object Detection](https://www.youtube.com/watch?v=UYLp0-iOvFc&list=PLkmvobsnE0GEfcliu9SXhtAQyyIiw9Kl0), [UZH : Visual Odometry and SLAM](https://www.youtube.com/watch?v=_yuZmzJoWUc&list=PLxXaypZSkh7K32tkvMIpFeG0MX53fNhq6), [Evolution of Visual Odometry Techniques](https://arxiv.org/ftp/arxiv/papers/1804/1804.11142.pdf), [A Review of Visual Odometry Methods and Its Applications for Autonomous Driving](https://arxiv.org/abs/2009.09193), [Visual Odometry on the Mars Exploration Rovers](https://www-robotics.jpl.nasa.gov/media/documents/vo_ras.pdf), [A Practical Survey on Visual Odometry for Autonomous Driving in Challenging Scenarios and Conditions](https://ieeexplore.ieee.org/document/9817108), [RPG - Visual and Inertial Odometry and SLAM](https://rpg.ifi.uzh.ch/research_vo.html).