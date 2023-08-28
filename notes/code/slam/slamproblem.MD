# The SLAM Problem:

Simultaneous Localization and Mapping (SLAM) is a critical problem in robotics and computer vision. It involves estimating the trajectory of a robot (or camera) while simultaneously building a map of its environment. SLAM can be approached in various ways, including probabilistic models, and can be formulated as a factor graph optimization problem.

### Motion Model:

The motion model describes how the robot's pose (position and orientation) changes over time. A commonly used model is odometry, which estimates the robot's motion based on wheel encoders or inertial sensors.

```cpp

struct OdometryModel {
    double x; // Change in x position
    double y; // Change in y position
    double theta; // Change in orientation
};

// Update robot pose using odometry data
void updatePose(Pose& currentPose, const OdometryModel& odometry) {
    currentPose.x += odometry.x;
    currentPose.y += odometry.y;
    currentPose.theta += odometry.theta;
}


```

### Observation Model:

The observation model relates sensor measurements to the map. For example, in visual SLAM, this might involve estimating the 3D positions of landmarks (features) in the environment from 2D image coordinates.

```cpp

struct Landmark {
    double x; // X-coordinate of the landmark
    double y; // Y-coordinate of the landmark
};

// Predict observed landmark position from the current robot pose
Landmark predictLandmark(const Pose& currentPose, const Landmark& landmarkInMap) {
    // Apply the robot's pose transformation to the landmark
    Landmark predictedLandmark;
    predictedLandmark.x = currentPose.x + cos(currentPose.theta) * landmarkInMap.x - sin(currentPose.theta) * landmarkInMap.y;
    predictedLandmark.y = currentPose.y + sin(currentPose.theta) * landmarkInMap.x + cos(currentPose.theta) * landmarkInMap.y;
    return predictedLandmark;
}


```

### Factor Graph Optimization (MLE and MAP):

Factor graph optimization is used to estimate the robot's trajectory and the map by optimizing a graph of factors that represent constraints from motion and observation models. Maximum Likelihood Estimation - MLE and Maximum A Posteriori - MAP.

For MLE, you can use a least-squares optimization approach:

```cpp

#include <Eigen/Dense>
using namespace Eigen;

// Define a factor graph
typedef std::vector<Factor> FactorGraph;

// Define a factor (error term) between predicted and observed landmarks
struct Factor {
    Landmark observedLandmark;
    Matrix2d measurementCovariance;
};

// MLE optimization using Gauss-Newton
void optimizeMLE(FactorGraph& graph, Pose& trajectory, std::vector<Landmark>& map) {
    for (Factor& factor : graph) {
        // Calculate the error (residual) between predicted and observed landmark
        Landmark predictedLandmark = predictLandmark(trajectory, map[factor.landmarkIndex]);
        Vector2d error = predictedLandmark - factor.observedLandmark;

        // Update the trajectory and map using Gauss-Newton
        // (This involves updating the state variables using the Jacobian and Hessian matrices)
        // ...
    }
}


```

For MAP optimization, you'd typically use a probabilistic framework, such as Bayesian filtering or optimization, to estimate the trajectory and map while considering prior information and measurement uncertainties.

Factor graph optimization is a powerful technique used in various fields such as robotics, computer vision, and machine learning to solve estimation problems. It involves representing a complex optimization problem as a graph, where nodes represent variables and factors represent constraints between these variables.

Factor Graph Basics: A factor graph consists of two types of nodes:

+ <b>Variable Nodes:</b> These nodes represent the variables we want to estimate. In SLAM or other estimation problems, these could be robot poses, landmark positions, etc.

+ <b>Factor Nodes:</b> These nodes represent constraints between variables. Factors encode information from measurements or priors.

<b>🌸 Maximum Likelihood Estimation (MLE):</b> In MLE, we want to find the values of variables that maximize the likelihood of the measurements given our model. Here's an example using a simple linear factor graph:

```cpp

#include <iostream>
#include <Eigen/Dense>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>

using namespace std;
using namespace Eigen;
using namespace gtsam;

// Define the factor graph
NonlinearFactorGraph graph;

// Create variable symbols
Symbol x1('x', 1);
Symbol x2('x', 2);

// Define measurement data
Vector2d measurement(2.0, 3.0);
Matrix2d measurementCovariance = Matrix2d::Identity();

// Create measurement factors
graph.add(BetweenFactor<Vector2d>(x1, x2, measurement, measurementCovariance));

// Create an initial estimate
Values initialEstimate;
initialEstimate.insert(x1, Vector2d(0.0, 0.0));
initialEstimate.insert(x2, Vector2d(0.0, 0.0));

// Optimize the factor graph using Levenberg-Marquardt
LevenbergMarquardtOptimizer optimizer(graph, initialEstimate);
Values result = optimizer.optimize();
cout << "MLE Estimated Values:" << endl;
cout << result << endl;


```

This code defines a simple factor graph with two variables `x1` and `x2`, connected by a measurement factor. It uses the GTSAM library for factor graph optimization.

<b>🌸 Maximum A Posteriori (MAP):</b> In MAP estimation, we want to find the values of variables that maximize the posterior probability, incorporating prior information. Here's an example:

```cpp

// Define the factor graph (same as above)
NonlinearFactorGraph graph;

// Define prior information (prior on x1)
Vector2d priorMean(1.0, 2.0);
Matrix2d priorCovariance = Matrix2d::Identity();
graph.add(PriorFactor<Vector2d>(x1, priorMean, priorCovariance));

// Optimize the factor graph using Levenberg-Marquardt
LevenbergMarquardtOptimizer optimizer(graph, initialEstimate);
Values mapResult = optimizer.optimize();
cout << "MAP Estimated Values:" << endl;
cout << mapResult << endl;


```

In this code, we've added a prior factor to `x1`, which represents prior information about its value.

Note that in real-world applications, factor graphs can be much larger and more complex, involving multiple types of factors and variables. Libraries like GTSAM, Ceres Solver, and others are commonly used for factor graph optimization due to their efficiency and flexibility.

## Mapping:

Mapping is a fundamental task in robotics and computer vision, which involves building representations of an environment using sensor data. Here, we'll discuss several mapping techniques: point cloud mapping, occupancy grid mapping (specifically, voxel mapping), Truncated Signed Distance Function (TSDF) mapping, and surfel mapping. 


<table width=100%>
<tr>
<th>Point Cloud Mapping</th>
<th>Occupancy Grid Mapping (Voxel Map)</th>
</tr>

<tr>
<td>

Point cloud mapping represents an environment by a collection of 3D points captured by sensors like LiDAR or RGB-D cameras.

```cpp

#include <iostream>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

int main() {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    
    // Populate the point cloud with data (e.g., from a sensor)
    for (float x = -1.0; x <= 1.0; x += 0.01) {
        for (float y = -1.0; y <= 1.0; y += 0.01) {
            pcl::PointXYZ point;
            point.x = x;
            point.y = y;
            point.z = 0.0;
            cloud->push_back(point);
        }
    }
    
    // Perform further processing or visualization with the point cloud
    // ...
    
    return 0;
}  
```


</td>

<td>


Occupancy grid mapping divides the environment into a grid of voxels and assigns each voxel a binary value (occupied or unoccupied).

```cpp

#include <iostream>
#include <vector>

// Define a simple voxel structure
struct Voxel {
    bool occupied;
};

int main() {
    // Create a 2D grid of voxels
    int grid_size_x = 100;
    int grid_size_y = 100;
    std::vector<std::vector<Voxel>> occupancy_grid(grid_size_x, std::vector<Voxel>(grid_size_y));

    // Populate the occupancy grid with data (e.g., from sensor readings)
    // ...

    // Perform mapping updates based on sensor data
    // ...

    return 0;
}


```

</td>

</tr>




<tr>
<th>Truncated Signed Distance Function (TSDF) Mapping</th>
<th>Surfel Mapping</th>
</tr>

<tr>
<td>

TSDF mapping represents the environment using a continuous, volumetric function that measures the signed distance from surfaces. It's commonly used in 3D reconstruction.

```cpp

#include <iostream>
#include <Eigen/Dense>

// Define a simple TSDF voxel structure
struct TSDFVoxel {
    float tsdf_value;
    float weight;
};

int main() {
    // Create a 3D grid of TSDF voxels
    int grid_size_x = 100;
    int grid_size_y = 100;
    int grid_size_z = 100;
    std::vector<std::vector<std::vector<TSDFVoxel>>> tsdf_grid(
        grid_size_x, 
        std::vector<std::vector<TSDFVoxel>>(
            grid_size_y, 
            std::vector<TSDFVoxel>(grid_size_z)
        )
    );

    // Populate the TSDF grid with data (e.g., from sensor readings)
    // ...

    // Perform TSDF updates based on sensor data
    // ...

    return 0;
}


```



</td>

<td>

Surfel mapping represents the environment using surfels, which are small oriented disk-like elements that approximate the surfaces in the environment.

```cpp

#include <iostream>
#include <vector>
#include <Eigen/Dense>

// Define a simple surfel structure
struct Surfel {
    Eigen::Vector3f position;
    Eigen::Vector3f normal;
    float radius;
    // Additional attributes
};

int main() {
    std::vector<Surfel> surfel_map;

    // Populate the surfel map with data (e.g., from sensor readings)
    // ...

    // Perform surfel updates based on sensor data
    // ...

    return 0;
}


```

</td>

</tr>
</table>




