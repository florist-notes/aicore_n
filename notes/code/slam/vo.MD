# Visual Odometry (Frontend)


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

  ORB Feature : [  The FAST Key Point, BRIEF Descriptor ] ; Feature Matching

ORB is basically a fusion of FAST keypoint detector and BRIEF descriptor with many modifications to enhance the performance. First it use FAST to find keypoints, then apply Harris corner measure to find top N points among them. It also use pyramid to produce multiscale-features.

article : [introduction to orb](https://medium.com/data-breach/introduction-to-orb-oriented-fast-and-rotated-brief-4220e8ec40cf), ORB Features from Scratch ([example.cpp](./orb_self_en.cpp)).


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

   //When the distance between descriptors is greater than twice the minimum distance, it is considered that the match is wrong. But sometimes the minimum distance will be very small, set an experience value of 30 as the lower limit.
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

## Triangulation

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

   //Find the minimum and maximum distances between all matches, that is, the distance between the most similar and least similar two sets of points
   for (int i = 0; i < descriptors_1.rows; i++) {
     double dist = match[i].distance;
     if (dist < min_dist) min_dist = dist;
     if (dist > max_dist) max_dist = dist;
   }

   printf("-- Max dist : %f \n", max_dist);
   printf("-- Min dist : %f \n", min_dist);

   //When the distance between descriptors is greater than twice the minimum distance, it is considered that the match is wrong. But sometimes the minimum distance will be very small, set an experience value of 30 as the lower limit.
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
