# Dense 3D Reconstruction

Dense 3D reconstruction, a fundamental task in computer vision and computer graphics, has witnessed remarkable progress in recent years. This abstract provides an overview of the key techniques and applications in the field of dense 3D reconstruction.

The process of dense 3D reconstruction involves the generation of detailed three-dimensional models of objects, scenes, or environments from two-dimensional image data. This process has evolved from traditional structure-from-motion (SfM) and stereo matching methods to more advanced techniques, such as multi-view stereo (MVS), depth sensing, and deep learning-based approaches.

Several critical aspects of dense 3D reconstruction are discussed in this abstract:

+ <b>Techniques</b>: The abstract reviews various techniques employed in dense 3D reconstruction, including traditional approaches like SfM and stereo matching, which rely on geometric principles to estimate depth. Additionally, it covers recent advancements in MVS, which exploit multiple views to create high-fidelity 3D models. Deep learning-based methods, such as convolutional neural networks (CNNs) and generative adversarial networks (GANs), have also made significant contributions, enabling accurate reconstructions even in challenging scenarios.

+ <b>Applications</b>: Dense 3D reconstruction finds applications in a wide range of fields. This abstract highlights its significance in computer vision for object recognition, tracking, and scene understanding. It also discusses its role in robotics, where accurate 3D models aid in navigation and manipulation tasks. Furthermore, dense 3D reconstruction is pivotal in cultural heritage preservation, virtual reality, and augmented reality, enhancing immersive experiences and historical documentation.

+ <b>Challenges and Future Directions</b>: While substantial progress has been made, dense 3D reconstruction continues to face challenges, such as scalability, robustness to varying lighting conditions, and real-time performance. The abstract explores potential future directions, including the integration of 3D reconstruction with semantic understanding, improved fusion of sensor data, and the development of lightweight algorithms for mobile and edge devices.

Goal :

+ Learn how to estimate the dense depth in monocular SLAM.
+ Implement the dense depth estimation in monocular SLAM.
+ Learn some of the commonly used map forms in RGB-D reconstruction.

In dense reconstruction, we need to know each pixel’s distance (or most of the pixels). There are roughly the following solutions for this:
1. Use monocular cameras and estimate the depth using triangulation after motion.
2. Use stereo cameras by its disparity (similar for more than two eyes).
3. Use the depth sensor in RGB-D cameras to directly get the depth.

[ Epipolar Line Search and Block Matching,  Gaussian Depth Filters ]

## Monocular Dense Reconstruction:

dataset : [remode](http://rpg.ifi.uzh.ch/datasets/remode_test_data.zip)

```cpp

#include <iostream>
#include <vector>
#include <fstream>

using namespace std;

#include <boost/timer.hpp>

// for sophus
#include <sophus/se3.hpp>

using Sophus::SE3d;

// for eigen
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace Eigen;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;

/************************************************
* This program demonstrates the dense depth estimation of a monocular camera under a known trajectory
* Using epipolar search + NCC matching, corresponding
***************************************************/

// ------------------------------------------------ ------------------
// parameters
const int boarder = 20; // edge width
const int width = 640; // image width
const int height = 480; // image height
const double fx = 481.2f; // internal camera parameters
const double fy = -480.0f;
const double cx = 319.5f;
const double cy = 239.5f;
const int ncc_window_size = 3; // Half width of window taken by NCC
const int ncc_area = (2 * ncc_window_size + 1) * (2 * ncc_window_size + 1); // NCC window area
const double min_cov = 0.1; // Convergence decision: minimum variance
const double max_cov = 10; // divergence determination: maximum variance

// ------------------------------------------------ ------------------
// important functions
/// Read data from REMODE dataset
bool readDatasetFiles(
     const string &path,
     vector<string> &color_image_files,
     vector<SE3d> &poses,
     cv::Mat &ref_depth
);

/**
  * Update the depth estimate based on the new image
  * @param ref reference image
  * @param curr current image
  * @param T_C_R The pose from the reference image to the current image
  * @param depth depth
  * @param depth_cov depth variance
  * @return success
  */
bool update(
     const Mat &ref,
     const Mat & curr,
     const SE3d &T_C_R,
     Mat & depth,
     Mat & depth_cov2
);

/**
  * Polar search
  * @param ref reference image
  * @param curr current image
  * @param T_C_R pose
  * @param pt_ref The position of the point in the reference image
  * @param depth_mu depth mean
  * @param depth_cov depth variance
  * @param pt_curr current point
  * @param epipolar_direction epipolar direction
  * @return success
  */
bool epipolarSearch(
     const Mat &ref,
     const Mat & curr,
     const SE3d &T_C_R,
     const Vector2d &pt_ref,
     const double &depth_mu,
     const double &depth_cov,
     Vector2d &pt_curr,
     Vector2d & epipolar_direction
);

/**
  * update depth filter
  * @param pt_ref reference image point
  * @param pt_curr current image point
  * @param T_C_R pose
  * @param epipolar_direction epipolar direction
  * @param depth mean depth
  * @param depth_cov2 depth direction
  * @return success
  */
bool updateDepthFilter(
     const Vector2d &pt_ref,
     const Vector2d &pt_curr,
     const SE3d &T_C_R,
     const Vector2d & epipolar_direction,
     Mat & depth,
     Mat & depth_cov2
);

/**
  * Calculate NCC score
  * @param ref reference image
  * @param curr current image
  * @param pt_ref reference point
  * @param pt_curr current point
  * @return NCC score
  */
double NCC(const Mat &ref, const Mat &curr, const Vector2d &pt_ref, const Vector2d &pt_curr);

// bilinear grayscale interpolation
inline double getBilinearInterpolatedValue(const Mat &img, const Vector2d &pt) {
     uchar *d = &img.data[int(pt(1, 0)) * img.step + int(pt(0, 0))];
     double xx = pt(0, 0) - floor(pt(0, 0));
     double yy = pt(1, 0) - floor(pt(1, 0));
     return ((1 - xx) * (1 - yy) * double(d[0]) +
             xx * (1 - yy) * double(d[1]) +
             (1 - xx) * yy * double(d[img.step]) +
             xx * yy * double(d[img.step + 1])) / 255.0;
}

// ------------------------------------------------ ------------------
// some widgets
// display the estimated depth map
void plotDepth(const Mat &depth_truth, const Mat &depth_estimate);

// pixel to camera coordinate system
inline Vector3d px2cam(const Vector2d px) {
     return Vector3d(
         (px(0, 0) - cx) / fx,
         (px(1, 0) - cy) / fy,
         1
     );
}

// camera coordinate system to pixel
inline Vector2d cam2px(const Vector3d p_cam) {
     return Vector2d(
         p_cam(0, 0) * fx / p_cam(2, 0) + cx,
         p_cam(1, 0) * fy / p_cam(2, 0) + cy
     );
}

// Check if a point is within the bounding box of the image
inline bool inside(const Vector2d &pt) {
     return pt(0, 0) >= boarder && pt(1, 0) >= boarder
            && pt(0, 0) + boarder < width && pt(1, 0) + boarder <= height;
}

// display epipolar matches
void showEpipolarMatch(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_curr);

// show polar lines
void showEpipolarLine(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_min_curr,
                       const Vector2d &px_max_curr);

/// Evaluation depth estimation
void evaluateDepth(const Mat &depth_truth, const Mat &depth_estimate);
// ------------------------------------------------ ------------------

int main(int argc, char **argv) {
     if (argc != 2) {
         cout << "Usage: dense_mapping path_to_test_dataset" << endl;
         return -1;
     }

     // read data from dataset
     vector<string> color_image_files;
     vector<SE3d> poses_TWC;
     Mat ref_depth;
     bool ret = readDatasetFiles(argv[1], color_image_files, poses_TWC, ref_depth);
     if (ret == false) {
         cout << "Reading image files failed!" << endl;
         return -1;
     }
     cout << "read total " << color_image_files. size() << "files." << endl;

     // first image
     Mat ref = imread(color_image_files[0], 0); // gray-scale image
     SE3d pose_ref_TWC = poses_TWC[0];
     double init_depth = 3.0; // initial value of depth
     double init_cov2 = 3.0; // initial value of variance
     Mat depth(height, width, CV_64F, init_depth); // depth map
     Mat depth_cov2(height, width, CV_64F, init_cov2); // depth map variance

     for (int index = 1; index < color_image_files. size(); index++) {
         cout << "*** loop " << index << " ***" << endl;
         Mat curr = imread(color_image_files[index], 0);
         if (curr. data == nullptr) continue;
         SE3d pose_curr_TWC = poses_TWC[index];
         SE3d pose_T_C_R = pose_curr_TWC.inverse() * pose_ref_TWC; 
         // Coordinate transformation relationship: T_C_W * T_W_R = T_C_R
         update(ref, curr, pose_T_C_R, depth, depth_cov2);
         evaluateDepth(ref_depth, depth);
         plotDepth(ref_depth, depth);
         imshow("image", curr);
         waitKey(1);
     }

     cout << "estimation returns, saving depth map ..." << endl;
     imwrite("depth.png", depth);
     cout << "done." << endl;

     return 0;
}

bool readDatasetFiles(
     const string &path,
     vector<string> &color_image_files,
     std::vector<SE3d> &poses,
     cv::Mat &ref_depth) {
     ifstream fin(path + "/first_200_frames_traj_over_table_input_sequence.txt");
     if (!fin) return false;

     while (!fin. eof()) {
         // Data format: image file name tx, ty, tz, qx, qy, qz, qw, note that it is TWC instead of TCW
         string image;
         fin >> image;
         double data[7];
         for (double &d:data) fin >> d;

         color_image_files.push_back(path + string("/images/") + image);
         poses.push_back(
             SE3d(Quaterniond(data[6], data[3], data[4], data[5]),
                  Vector3d(data[0], data[1], data[2]))
         );
         if (!fin.good()) break;
     }
     fin. close();

     // load reference depth
     fin.open(path + "/depthmaps/scene_000.depth");
     ref_depth = cv::Mat(height, width, CV_64F);
     if (!fin) return false;
     for (int y = 0; y < height; y++)
         for (int x = 0; x < width; x++) {
             double depth = 0;
             fin >> depth;
             ref_depth.ptr<double>(y)[x] = depth / 100.0;
         }

     return true;
}

// Update the entire depth map
bool update(const Mat &ref, const Mat &curr, const SE3d &T_C_R, Mat &depth, Mat &depth_cov2) {
     for (int x = boarder; x < width - boarder; x++)
         for (int y = boarder; y < height - boarder; y++) {
             // loop through each pixel
             if (depth_cov2.ptr<double>(y)[x] < min_cov || depth_cov2.ptr<double>(y)[x] > max_cov)
              // depth has converged or diverged
                 continue;
             // search for a match of (x,y) on the epipolar line
             Vector2d pt_curr;
             Vector2d epipolar_direction;
             bool ret = epipolarSearch(
                 ref,
                 curr,
                 T_C_R,
                 Vector2d(x, y),
                 depth.ptr<double>(y)[x],
                 sqrt(depth_cov2.ptr<double>(y)[x]),
                 pt_curr,
                 epipolar_direction
             );

             if (ret == false) // match failed
                 continue;

             // uncomment this to show matches
             // showEpipolarMatch(ref, curr, Vector2d(x, y), pt_curr);

             // The match is successful, update the depth map
             updateDepthFilter(Vector2d(x, y), pt_curr, T_C_R, epipolar_direction, depth, depth_cov2);
         }
}

// polar search
// See the book 12.2 and 12.3 for the method
bool epipolarSearch(
     const Mat &ref, const Mat &curr,
     const SE3d &T_C_R, const Vector2d &pt_ref,
     const double &depth_mu, const double &depth_cov,
     Vector2d &pt_curr, Vector2d &epipolar_direction) {
     Vector3d f_ref = px2cam(pt_ref);
     f_ref. normalize();
     Vector3d P_ref = f_ref * depth_mu; // P vector of the reference frame

     Vector2d px_mean_curr = cam2px(T_C_R * P_ref); // Pixels projected by depth mean
     double d_min = depth_mu - 3 * depth_cov, d_max = depth_mu + 3 * depth_cov;
     if (d_min < 0.1) d_min = 0.1;
     Vector2d px_min_curr = cam2px(T_C_R * (f_ref * d_min)); // Pixels projected by minimum depth
     Vector2d px_max_curr = cam2px(T_C_R * (f_ref * d_max)); // Pixels projected by maximum depth

     Vector2d epipolar_line = px_max_curr - px_min_curr; // epipolar line (line segment form)
     epipolar_direction = epipolar_line; // epipolar line direction
     epipolar_direction.normalize();
     double half_length = 0.5 * epipolar_line.norm(); // Half length of epipolar line segment
     if (half_length > 100) half_length = 100; // we don't want to search too many things

     // Uncomment this sentence to display epipolar lines (line segments)
     // showEpipolarLine( ref, curr, pt_ref, px_min_curr, px_max_curr );

     // Search on the epipolar line, take the depth mean point as the center, and 
     // take half lengths on the left and right sides
     double best_ncc = -1.0;
     Vector2d best_px_curr;
     for (double l = -half_length; l <= half_length; l += 0.7) { // l+=sqrt(2)
         Vector2d px_curr = px_mean_curr + l * epipolar_direction; // points to be matched
         if (!inside(px_curr))
             continue;
         // Calculate the NCC between the point to be matched and the reference frame
         double ncc = NCC(ref, curr, pt_ref, px_curr);
         if (ncc > best_ncc) {
             best_ncc = ncc;
             best_px_curr = px_curr;
         }
     }
     if (best_ncc < 0.85f) // only trust matches with high NCC
         return false;
     pt_curr = best_px_curr;
     return true;
}

double NCC(
     const Mat &ref, const Mat &curr,
     const Vector2d &pt_ref, const Vector2d &pt_curr) {
     // zero mean - normalized cross correlation
     // Calculate the mean first
     double mean_ref = 0, mean_curr = 0;
     vector<double> values_ref, values_curr; 
     // mean value of reference frame and current frame
     for (int x = -ncc_window_size; x <= ncc_window_size; x++)
         for (int y = -ncc_window_size; y <= ncc_window_size; y++) {
             double value_ref = double(ref.ptr<uchar>(int(y + pt_ref(1, 0)))[int(x + pt_ref(0, 0))]) / 255.0;
             mean_ref += value_ref;

             double value_curr = getBilinearInterpolatedValue(curr, pt_curr + Vector2d(x, y));
             mean_curr += value_curr;

             values_ref.push_back(value_ref);
             values_curr.push_back(value_curr);
         }

     mean_ref /= ncc_area;
     mean_curr /= ncc_area;

     // Calculate Zero mean NCC
     double numerator = 0, demoniator1 = 0, demoniator2 = 0;
     for (int i = 0; i < values_ref. size(); i++) {
         double n = (values_ref[i] - mean_ref) * (values_curr[i] - mean_curr);
         numerator += n;
         demoniator1 += (values_ref[i] - mean_ref) * (values_ref[i] - mean_ref);
         demoniator2 += (values_curr[i] - mean_curr) * (values_curr[i] - mean_curr);
     }
     return numerator / sqrt(demoniator1 * demoniator2 + 1e-10); 
     // prevent denominator from zero
}

bool updateDepthFilter(
     const Vector2d &pt_ref,
     const Vector2d &pt_curr,
     const SE3d &T_C_R,
     const Vector2d & epipolar_direction,
     Mat & depth,
     Mat &depth_cov2) {
     // I don't know if anyone still reads this paragraph
     // calculate depth with triangulation
     SE3d T_R_C = T_C_R.inverse();
     Vector3d f_ref = px2cam(pt_ref);
     f_ref. normalize();
     Vector3d f_curr = px2cam(pt_curr);
     f_curr. normalize();

     // equation
     // d_ref * f_ref = d_cur * ( R_RC * f_cur ) + t_RC
     // f2 = R_RC * f_cur
     // Transform into the following matrix equations
     // => [ f_ref^T f_ref, -f_ref^T f2 ] [d_ref] [f_ref^T t]
     // [ f_2^T f_ref, -f2^T f2 ] [d_cur] = [f2^T t ]
     Vector3d t = T_R_C.translation();
     Vector3d f2 = T_R_C.so3() * f_curr;
     Vector2d b = Vector2d(t.dot(f_ref), t.dot(f2));
     Matrix2d A;
     A(0, 0) = f_ref.dot(f_ref);
     A(0, 1) = -f_ref.dot(f2);
     A(1, 0) = -A(0, 1);
     A(1, 1) = -f2.dot(f2);
     Vector2d ans = A.inverse() * b;
     Vector3d xm = ans[0] * f_ref; // ref side result
     Vector3d xn = t + ans[1] * f2; // cur result
     Vector3d p_esti = (xm + xn) / 2.0; // The position of P, take the average of the two
     double depth_estimation = p_esti.norm(); // depth value

     // calculate uncertainty (in one pixel as error)
     Vector3d p = f_ref * depth_estimation;
     Vector3d a = p - t;
     double t_norm = t.norm();
     double a_norm = a.norm();
     double alpha = acos(f_ref. dot(t) / t_norm);
     double beta = acos(-a.dot(t) / (a_norm * t_norm));
     Vector3d f_curr_prime = px2cam(pt_curr + epipolar_direction);
     f_curr_prime.normalize();
     double beta_prime = acos(f_curr_prime.dot(-t) / t_norm);
     double gamma = M_PI - alpha - beta_prime;
     double p_prime = t_norm * sin(beta_prime) / sin(gamma);
     double d_cov = p_prime - depth_estimation;
     double d_cov2 = d_cov * d_cov;

     // Gaussian fusion
     double mu = depth.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))];
     double sigma2 = depth_cov2.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))];

     double mu_fuse = (d_cov2 * mu + sigma2 * depth_estimation) / (sigma2 + d_cov2);
     double sigma_fuse2 = (sigma2 * d_cov2) / (sigma2 + d_cov2);

     depth.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))] = mu_fuse;
     depth_cov2.ptr<double>(int(pt_ref(1, 0)))[int(pt_ref(0, 0))] = sigma_fuse2;

     return true;
}

// The latter are too simple and I will not comment (actually because of laziness)
void plotDepth(const Mat &depth_truth, const Mat &depth_estimate) {
     imshow("depth_truth", depth_truth * 0.4);
     imshow("depth_estimate", depth_estimate * 0.4);
     imshow("depth_error", depth_truth - depth_estimate);
     waitKey(1);
}

void evaluateDepth(const Mat &depth_truth, const Mat &depth_estimate) {
     double ave_depth_error = 0; // average error
     double ave_depth_error_sq = 0; // square error
     int cnt_depth_data = 0;
     for (int y = boarder; y < depth_truth. rows - boarder; y++)
         for (int x = boarder; x < depth_truth. cols - boarder; x++) {
             double error = depth_truth.ptr<double>(y)[x] - depth_estimate.ptr<double>(y)[x];
             ave_depth_error += error;
             ave_depth_error_sq += error * error;
             cnt_depth_data++;
         }
     ave_depth_error /= cnt_depth_data;
     ave_depth_error_sq /= cnt_depth_data;

     cout << "Average squared error = " << ave_depth_error_sq << ", average error: " << ave_depth_error << endl;
}



void showEpipolarMatch(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_curr) {
    Mat ref_show, curr_show;
    cv::cvtColor(ref, ref_show, CV_GRAY2BGR);
    cv::cvtColor(curr, curr_show, CV_GRAY2BGR);

    cv::circle(ref_show, cv::Point2f(px_ref(0, 0), px_ref(1, 0)), 5, cv::Scalar(0, 0, 250), 2);
    cv::circle(curr_show, cv::Point2f(px_curr(0, 0), px_curr(1, 0)), 5, cv::Scalar(0, 0, 250), 2);

    imshow("ref", ref_show);
    imshow("curr", curr_show);
    waitKey(1);
}

void showEpipolarLine(const Mat &ref, const Mat &curr, const Vector2d &px_ref, const Vector2d &px_min_curr,
                      const Vector2d &px_max_curr) {

    Mat ref_show, curr_show;
    cv::cvtColor(ref, ref_show, CV_GRAY2BGR);
    cv::cvtColor(curr, curr_show, CV_GRAY2BGR);

    cv::circle(ref_show, cv::Point2f(px_ref(0, 0), px_ref(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::circle(curr_show, cv::Point2f(px_min_curr(0, 0), px_min_curr(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::circle(curr_show, cv::Point2f(px_max_curr(0, 0), px_max_curr(1, 0)), 5, cv::Scalar(0, 255, 0), 2);
    cv::line(curr_show, Point2f(px_min_curr(0, 0), px_min_curr(1, 0)), Point2f(px_max_curr(0, 0), px_max_curr(1, 0)),
             Scalar(0, 255, 0), 1);

    imshow("ref", ref_show);
    imshow("curr", curr_show);
    waitKey(1);
}

```


[ Pixel Gradients, Inverse Depth Filter,  Pre-Transform the Image, ]

## RGB-D Point Cloud Mapping:

 The so-called point cloud is a map represented by a set of discrete points. The most basic point contains threedimensional coordinates of x, y,z and also color information of r, g, b. Since the
RGB-D camera provides a color map and a depth map, it is easy to calculate the RGB-D point cloud based on the camera’s internal parameters. If the camera’s pose is obtained, then we can directly merge the keyframes into a global point cloud.

```cpp

#include <iostream>
#include <fstream>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Geometry>
#include <boost/format.hpp> // for formatting strings
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>

int main(int argc, char **argv) {
     vector<cv::Mat> colorImgs, depthImgs; // color image and depth image
     vector<Eigen::Isometry3d> poses; // camera pose

     ifstream fin("./data/pose.txt");
     if (!fin) {
         cerr << "cannot find pose file" << endl;
         return 1;
     }

     for (int i = 0; i < 5; i++) {
         boost::format fmt("./data/%s/%d.%s"); //image file format
         colorImgs.push_back(cv::imread((fmt % "color" % (i + 1) % "png").str()));
         depthImgs.push_back(cv::imread((fmt % "depth" % (i + 1) % "png").str(), -1)); // use -1 to read the original image

         double data[7] = {0};
         for (int i = 0; i < 7; i++) {
             fin >> data[i];
         }
         Eigen::Quaterniond q(data[6], data[3], data[4], data[5]);
         Eigen::Isometry3d T(q);
         T.pretranslate(Eigen::Vector3d(data[0], data[1], data[2]));
         poses. push_back(T);
     }

     // Calculate the point cloud and stitch it
     // internal parameters of the camera
     double cx = 319.5;
     double cy = 239.5;
     double fx = 481.2;
     double fy = -480.0;
     double depthScale = 5000.0;

     cout << "Converting image to point cloud..." << endl;

     // Define the format used by the point cloud: XYZRGB is used here
     typedef pcl::PointXYZRGB PointT;
     typedef pcl::PointCloud<PointT> PointCloud;

     // Create a new point cloud
     PointCloud::Ptr pointCloud(new PointCloud);
     for (int i = 0; i < 5; i++) {
         PointCloud::Ptr current(new PointCloud);
         cout << "Conversion image: " << i + 1 << endl;
         cv::Mat color = colorImgs[i];
         cv::Mat depth = depthImgs[i];
         Eigen::Isometry3d T = poses[i];
         for (int v = 0; v < color.rows; v++)
             for (int u = 0; u < color.cols; u++) {
                 unsigned int d = depth.ptr<unsigned short>(v)[u]; // depth value
                 if (d == 0) continue; // 0 means not measured
                 Eigen::Vector3d point;
                 point[2] = double(d) / depthScale;
                 point[0] = (u - cx) * point[2] / fx;
                 point[1] = (v - cy) * point[2] / fy;
                 Eigen::Vector3d pointWorld = T * point;

                 PointT p;
                 p.x = pointWorld[0];
                 p.y = pointWorld[1];
                 p.z = pointWorld[2];
                 p.b = color.data[v * color.step + u * color.channels()];
                 p.g = color.data[v * color.step + u * color.channels() + 1];
                 p.r = color.data[v * color.step + u * color.channels() + 2];
                 current->points.push_back(p);
             }
         // depth filter and statistical removal
         PointCloud::Ptr tmp(new PointCloud);
         pcl::StatisticalOutlierRemoval<PointT> statistical_filter;
         statistical_filter.setMeanK(50);
         statistical_filter.setStddevMulThresh(1.0);
         statistical_filter.setInputCloud(current);
         statistical_filter. filter(*tmp);
         (*pointCloud) += *tmp;
     }

     pointCloud->is_dense = false;
     cout << "point cloud total" << pointCloud->size() << "points." << endl;

     //voxel filter
     pcl::VoxelGrid<PointT> voxel_filter;
     double resolution = 0.03;
     voxel_filter.setLeafSize(resolution, resolution, resolution); // resolution
     PointCloud::Ptr tmp(new PointCloud);
     voxel_filter.setInputCloud(pointCloud);
     voxel_filter. filter(*tmp);
     tmp->swap(*pointCloud);

     cout << "After filtering, the point cloud has a total of" << pointCloud->size() << "points." << endl;

     pcl::io::savePCDFileBinary("map.pcd", *pointCloud);
     return 0;
}

```

### Building Meshes from Point Cloud:

```cpp

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/surface/surfel_smoothing.h>
#include <pcl/surface/mls.h>
#include <pcl/surface/gp3.h>
#include <pcl/surface/impl/mls.hpp>

// typedefs
typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointCloud<PointT>::Ptr PointCloudPtr;
typedef pcl::PointXYZRGBNormal SurfelT;
typedef pcl::PointCloud<SurfelT> SurfelCloud;
typedef pcl::PointCloud<SurfelT>::Ptr SurfelCloudPtr;

SurfelCloudPtr reconstructSurface(
        const PointCloudPtr &input, float radius, int polynomial_order) {
    pcl::MovingLeastSquares<PointT, SurfelT> mls;
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>);
    mls.setSearchMethod(tree);
    mls.setSearchRadius(radius);
    mls.setComputeNormals(true);
    mls.setSqrGaussParam(radius * radius);
    mls.setPolynomialFit(polynomial_order > 1);
    mls.setPolynomialOrder(polynomial_order);
    mls.setInputCloud(input);
    SurfelCloudPtr output(new SurfelCloud);
    mls.process(*output);
    return (output);
}

pcl::PolygonMeshPtr triangulateMesh(const SurfelCloudPtr &surfels) {
    // Create search tree*
    pcl::search::KdTree<SurfelT>::Ptr tree(new pcl::search::KdTree<SurfelT>);
    tree->setInputCloud(surfels);

    // Initialize objects
    pcl::GreedyProjectionTriangulation<SurfelT> gp3;
    pcl::PolygonMeshPtr triangles(new pcl::PolygonMesh);

    // Set the maximum distance between connected points (maximum edge length)
    gp3.setSearchRadius(0.05);

    // Set typical values for the parameters
    gp3.setMu(2.5);
    gp3.setMaximumNearestNeighbors(100);
    gp3.setMaximumSurfaceAngle(M_PI / 4); // 45 degrees
    gp3.setMinimumAngle(M_PI / 18); // 10 degrees
    gp3.setMaximumAngle(2 * M_PI / 3); // 120 degrees
    gp3.setNormalConsistency(true);

    // Get result
    gp3.setInputCloud(surfels);
    gp3.setSearchMethod(tree);
    gp3.reconstruct(*triangles);

    return triangles;
}

int main(int argc, char **argv) {

    // Load the points
    PointCloudPtr cloud(new PointCloud);
    if (argc == 0 || pcl::io::loadPCDFile(argv[1], *cloud)) {
        cout << "failed to load point cloud!";
        return 1;
    }
    cout << "point cloud loaded, points: " << cloud->points.size() << endl;

    // Compute surface elements
    cout << "computing normals ... " << endl;
    double mls_radius = 0.05, polynomial_order = 2;
    auto surfels = reconstructSurface(cloud, mls_radius, polynomial_order);

    // Compute a greedy surface triangulation
    cout << "computing mesh ... " << endl;
    pcl::PolygonMeshPtr mesh = triangulateMesh(surfels);

    cout << "display mesh ... " << endl;
    pcl::visualization::PCLVisualizer vis;
    vis.addPolylineFromPolygonMesh(*mesh, "mesh frame");
    vis.addPolygonMesh(*mesh, "mesh");
    vis.resetCamera();
    vis.spin();
}

```

##  Octo-Mapping:

OctoMapping is a probabilistic framework that combines occupancy grids with octrees, providing a dynamic and memory-efficient approach for representing and updating 3D maps of complex environments. The technique leverages sensor data, such as lidar or depth cameras, to construct high-resolution 3D maps while estimating the likelihood of occupancy at each voxel. This probabilistic approach enables reliable obstacle detection, mapping, and path planning for autonomous robots and vehicles.

<img src="./octo2.png" width=100%>

```cpp

#include <iostream>
#include <fstream>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <octomap/octomap.h> // for octomap

#include <Eigen/Geometry>
#include <boost/format.hpp> // for formatting strings

int main(int argc, char **argv) {
     vector<cv::Mat> colorImgs, depthImgs; // color image and depth image
     vector<Eigen::Isometry3d> poses; // camera pose

     ifstream fin("./data/pose.txt");
     if (!fin) {
         cerr << "cannot find pose file" << endl;
         return 1;
     }

     for (int i = 0; i < 5; i++) {
         boost::format fmt("./data/%s/%d.%s"); //image file format
         colorImgs.push_back(cv::imread((fmt % "color" % (i + 1) % "png").str()));
         depthImgs.push_back(cv::imread((fmt % "depth" % (i + 1) % "png").str(), -1)); // use -1 to read the original image

         double data[7] = {0};
         for (int i = 0; i < 7; i++) {
             fin >> data[i];
         }
         Eigen::Quaterniond q(data[6], data[3], data[4], data[5]);
         Eigen::Isometry3d T(q);
         T.pretranslate(Eigen::Vector3d(data[0], data[1], data[2]));
         poses. push_back(T);
     }

     // Calculate the point cloud and stitch it
     // internal parameters of the camera
     double cx = 319.5;
     double cy = 239.5;
     double fx = 481.2;
     double fy = -480.0;
     double depthScale = 5000.0;

     cout << "Converting image to Octomap..." << endl;

     // octomap tree
     octomap::OcTree tree(0.01); // The parameter is the resolution

     for (int i = 0; i < 5; i++) {
         cout << "Conversion image: " << i + 1 << endl;
         cv::Mat color = colorImgs[i];
         cv::Mat depth = depthImgs[i];
         Eigen::Isometry3d T = poses[i];

         octomap::Pointcloud cloud; // the point cloud in octomap

         for (int v = 0; v < color.rows; v++)
             for (int u = 0; u < color.cols; u++) {
                 unsigned int d = depth.ptr<unsigned short>(v)[u]; // depth value
                 if (d == 0) continue; // 0 means not measured
                 Eigen::Vector3d point;
                 point[2] = double(d) / depthScale;
                 point[0] = (u - cx) * point[2] / fx;
                 point[1] = (v - cy) * point[2] / fy;
                 Eigen::Vector3d pointWorld = T * point;
                 // Put the points in the world coordinate system into the point cloud
                 cloud.push_back(pointWorld[0], pointWorld[1], pointWorld[2]);
             }

         // Store the point cloud into the octree map, given the origin, so that the projection line can be calculated
         tree.insertPointCloud(cloud, octomap::point3d(T(0, 3), T(1, 3), T(2, 3)));
     }

     // Update the occupancy information of the intermediate node and write it to disk
     tree.updateInnerOccupancy();
     cout << "saving octomap ... " << endl;
     tree.writeBinary("octomap.bt");
     return 0;
}

```
## Real-time 3D reconstruction:

+ [NeuralRecon: Real-Time Coherent 3D Reconstruction from Monocular Video](https://arxiv.org/abs/2104.00681)
+ [ReFusion: 3D Reconstruction in Dynamic Environments for RGB-D Cameras Exploiting Residuals](https://www.ipb.uni-bonn.de/pdfs/palazzolo2019iros.pdf)
+ [DFusion: Denoised TSDF Fusion of Multiple Depth Maps with Sensor Pose Noises](https://www.mdpi.com/1424-8220/22/4/1631)
+ [3 Dimensional Dense Reconstruction: A Review of Algorithms and Dataset](https://arxiv.org/ftp/arxiv/papers/2304/2304.09371.pdf)
+ [State of the Art in Dense Monocular Non-Rigid 3D Reconstruction](https://arxiv.org/pdf/2210.15664.pdf)

*TSDF and RGB-D Fusion Series | [Real-time on-device 3D reconstruction](https://youtu.be/PxjjYCxGUR8?si=Py0fyShqpUMV7Tpl) | [dense_3d_reconstruction.pdf](./lecture_10.pdf)