# Optical Flow | Structure from Motion (SfM)

Optical Flow and Structure from Motion (SfM) are two essential computer vision techniques used for motion analysis and 3D reconstruction from images or video sequences.

<b>Optical Flow:</b> Optical Flow is the estimation of the motion of objects between consecutive frames in a video sequence. It calculates the displacement of pixels from one frame to another. The optical flow field provides information about the speed and direction of motion in an image or video.

<b>Key Concepts:</b>

+ <b>Lucas-Kanade Optical Flow:</b> The Lucas-Kanade method is a popular optical flow algorithm that assumes that the motion between frames is small and can be approximated as translational motion. It computes the flow vector for each pixel by solving a system of linear equations.

+ <b>Horn-Schunck Optical Flow:</b> The Horn-Schunck method is another optical flow algorithm that computes a global flow field across the entire image. It assumes that the brightness of the image is constant over time.

```cpp

#include <opencv2/opencv.hpp>

int main() {
    cv::VideoCapture cap("video.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file." << std::endl;
        return -1;
    }

    cv::Mat prevFrame, nextFrame;
    cap >> prevFrame;

    cv::Mat flow;
    cv::Ptr<cv::DenseOpticalFlow> opticalFlow = cv::createOptFlow_DualTVL1();

    while (true) {
        cap >> nextFrame;
        if (nextFrame.empty()) {
            break;
        }

        opticalFlow->calc(prevFrame, nextFrame, flow);

        // Display optical flow visualization
        cv::Mat flowVis;
        cv::cvtColor(prevFrame, flowVis, cv::COLOR_BGR2GRAY);
        cv::imshow("Optical Flow", flowVis);
        
        prevFrame = nextFrame;

        if (cv::waitKey(30) == 27) {
            break; // Press 'Esc' to exit
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}


```

## Structure from Motion (SfM):

<b>Structure from Motion (SfM):</b> SfM is a technique for estimating the 3D structure of a scene from a collection of 2D images. It involves the following steps:

Key Concepts:

+ <b>Feature Detection and Matching:</b> Detect distinctive features (e.g., keypoints) in each image and match corresponding features between images.

+ <b>Camera Pose Estimation:</b> Estimate the camera's pose (position and orientation) for each image in a common coordinate system.

+ <b>Triangulation:</b> Calculate the 3D position of matched feature points using the camera poses.

+ <b>Bundle Adjustment:</b> Refine camera poses and 3D points to minimize reprojection errors.

SfM is a complex process and typically requires the use of specialized libraries like OpenMVG (Open Multiple View Geometry). Here's a simplified example using the OpenCV library for feature detection and matching:

```cpp

#include <opencv2/opencv.hpp>

int main() {
    // Load a set of images
    std::vector<cv::Mat> images;
    // Load images into the 'images' vector

    // Initialize feature detector and descriptor
    cv::Ptr<cv::Feature2D> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    std::vector<std::vector<cv::KeyPoint>> keypoints(images.size());
    std::vector<cv::Mat> descriptors(images.size());

    // Detect keypoints and compute descriptors for each image
    for (int i = 0; i < images.size(); ++i) {
        detector->detect(images[i], keypoints[i]);
        detector->compute(images[i], keypoints[i], descriptors[i]);
    }

    // Perform feature matching between images
    std::vector<std::vector<cv::DMatch>> matches;
    for (int i = 0; i < images.size() - 1; ++i) {
        matcher->knnMatch(descriptors[i], descriptors[i + 1], matches, 2);

        // Apply ratio test to select good matches
        std::vector<cv::DMatch> goodMatches;
        for (const cv::DMatch& match : matches) {
            if (match[0].distance < 0.7 * match[1].distance) {
                goodMatches.push_back(match[0]);
            }
        }

        // Use goodMatches to estimate camera poses and 3D structure (not shown in this example)
    }

    return 0;
}


```

### Tomasi-Kanade Factorization:

Tomasi-Kanade Factorization is a technique used in computer vision and structure from motion (SfM) to estimate the 3D structure and motion from a sequence of 2D images. It's particularly useful for reconstructing scenes from multiple images, such as those captured by a camera moving through a scene. 

+ <b>Observation Matrix:</b> The observation matrix, often denoted as A, represents the correspondence between 3D points and 2D image points across multiple images. Each row of this matrix typically corresponds to a single point in one image and its projections in other images.

+ <b>Factorization:</b> Tomasi-Kanade Factorization decomposes the observation matrix into two matrices:

+ <b>Structure Matrix (3D Points):</b> Represents the 3D coordinates of the points in the scene.
+ <b>Motion Matrix (Camera Poses):</b> Represents the camera poses (rotations and translations) for each image.
+ <b>Linear Approximation:</b> The method assumes that the 3D structure and motion are linearly related to the 2D observations. This is valid when the scene undergoes small motions.

In this simplified example, we'll use OpenCV to perform Tomasi-Kanade Factorization on synthetic data. In practice, real-world applications involve the use of feature points, camera calibration, and bundle adjustment for improved accuracy.

```cpp

#include <opencv2/opencv.hpp>

int main() {
    // Create synthetic data (for simplicity)
    int num_images = 3;  // Number of images
    int num_points = 10; // Number of 3D points

    // Generate random 3D points
    cv::Mat structure = cv::Mat::randn(3, num_points, CV_64F);
    
    // Generate random camera poses (rotation and translation matrices)
    std::vector<cv::Mat> motion;
    for (int i = 0; i < num_images; ++i) {
        cv::Mat rotation, translation;
        cv::RNG rng;
        rng.fill(rotation, cv::RNG::UNIFORM, 0, 1);
        rng.fill(translation, cv::RNG::UNIFORM, 0, 1);
        motion.push_back(rotation);
        motion.push_back(translation);
    }
    
    // Generate synthetic observations (2D image points)
    cv::Mat observations(num_images * 2, num_points, CV_64F);
    observations = motion[0] * structure;
    
    // Perform Tomasi-Kanade Factorization
    cv::Mat structure_estimate, motion_estimate;
    cv::decomposeProjectionMatrix(observations, structure_estimate, motion_estimate);
    
    // Output the estimated structure and motion matrices
    std::cout << "Estimated Structure (3D Points):\n" << structure_estimate << std::endl;
    std::cout << "Estimated Motion (Camera Poses):\n" << motion_estimate << std::endl;

    return 0;
}


```

### Change detection and object tracking:

Change detection and object tracking are fundamental tasks in computer vision and image processing. These tasks involve identifying changes between consecutive frames of a video sequence or tracking objects as they move through a scene. One commonly used technique for change detection and object tracking is the Gaussian Mixture Model (GMM). 

+ <b>Gaussian Mixture Model (GMM):</b> GMM is a probabilistic model used for representing a mixture of multiple Gaussian distributions. In the context of change detection and object tracking, each Gaussian component models the pixel intensity distribution in a background or foreground region. GMM is particularly useful when dealing with dynamic backgrounds and lighting changes.

+ <b>Background Subtraction:</b> is the process of separating the background (static scene elements) from the foreground (moving objects) in a video sequence. GMM-based background subtraction models the background as a mixture of Gaussians and updates the model over time to adapt to changes.

In this example, we'll use OpenCV's BackgroundSubtractorMOG2, which is an implementation of a GMM-based background subtraction method.

```cpp

#include <opencv2/opencv.hpp>

int main() {
    cv::VideoCapture cap("video.mp4");
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video file." << std::endl;
        return -1;
    }

    cv::Ptr<cv::BackgroundSubtractorMOG2> bgSubtractor = cv::createBackgroundSubtractorMOG2();
    cv::Mat frame, fgMask;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        // Apply background subtraction
        bgSubtractor->apply(frame, fgMask);

        // Threshold the foreground mask to obtain binary segmentation
        cv::threshold(fgMask, fgMask, 128, 255, cv::THRESH_BINARY);

        // Display the result
        cv::imshow("Foreground Mask", fgMask);
        
        if (cv::waitKey(30) == 27) {
            break; // Press 'Esc' to exit
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}


```

In this code:

+ We open a video capture stream from a video file.
+ We initialize the BackgroundSubtractorMOG2 object, which models the background using GMM.
+ For each frame in the video, we apply background subtraction to obtain a foreground mask.
+ We threshold the foreground mask to create a binary segmentation, where + white pixels represent the detected foreground.
Finally, we display the binary segmentation result.