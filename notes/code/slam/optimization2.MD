# Filters and Optimizations (Backend) II

Sliding Window Optimization is a key technique used in Visual Simultaneous Localization and Mapping (SLAM), an essential technology in robotics and computer vision. It plays a critical role in improving the efficiency and accuracy of SLAM systems. Here's an overview of sliding window optimization in the context of SLAM:


+ <b>SLAM Overview:</b> Visual SLAM is the process of a robot or a camera simultaneously estimating its own motion (localization) and creating a map of its environment (mapping) using visual sensor data, typically from cameras. This is vital for tasks like autonomous navigation and 3D reconstruction.

+ <b>Temporal Window:</b> In SLAM, the system typically maintains a temporal window of recent sensor measurements and estimated poses. This window contains a limited number of keyframes, which are the most important frames for mapping and localization. This window helps strike a balance between computational efficiency and maintaining enough historical data for optimization.

+ <b>Optimization Objective:</b> Sliding Window Optimization involves optimizing the poses of the keyframes and the map points (3D landmarks) within the temporal window. The objective is to minimize the error between predicted and observed feature points in the images. This is usually formulated as a nonlinear least squares problem.

+ <b>Sliding Window Approach:</b> Instead of optimizing over the entire history of poses and map points, which can be computationally expensive, sliding window optimization restricts optimization to the most recent keyframes and associated map points within the temporal window. This sliding window is updated as new frames are processed, effectively shifting it through time.

Goal :
+ Learn the principles of sliding window optimization;
+ Learn the basic knowledge about pose graph optimization.
+ Pose graph optimization with g2o.


##  Sliding Window Filter and Optimization :


[ Controlling the Structure of BA ] : 

The graph optimization with camera pose and spatial points is called BA, which can
effectively solve large-scale positioning and mapping problems. This is very useful
in SfM, but in the real-time SLAM process, we often need to control the problem’s
scale to maintain real-time calculations. If the computing power is unlimited, we
might calculate the entire BA every moment-but that does not meet the reality. The
reality is that we must limit the calculation time of the backend. For example, the
BA scale cannot exceed 10,000 landmarks, the iterations cannot exceed 20 times,
and the time used does not exceed 0.5 s, and so on. An algorithm that takes a week
to reconstruct a city map like SfM is not necessarily effective in SLAM.
There are many ways to control the calculation scale, such as extracting keyframes
from the continuous video, only constructing the BA between the keyframe and
the landmarks. 

### Sliding Window

## Pose Graph Optimization

We can construct a graph optimization with only pose variables. The edge between the pose vertices can be set with measurement by the ego-motion estimation obtained from feature matching. The difference is that once the initial estimation is completed, we no longer optimize the positions of those landmark points but only care about the connections between all camera poses. In this way, we save a lot of calculations for landmarks optimization and only keep the keyframe’s trajectory, thus constructing the so-called pose graph (Pose Graph).

[ [Graph-based SLAM using Pose Graphs (Cyrill Stachniss)](https://youtu.be/uHbRKvD8TWg?si=tiK6PrmoBrPapraD) , [Graph-based SLAM in 20 Minutes](https://youtu.be/Alu59K8zvYs?si=GTd_CEeswBr2Ll6K)]

Pose Graph Optimization is a fundamental technique in robotics and autonomous systems, particularly in the field of Simultaneous Localization and Mapping (SLAM). It plays a pivotal role in improving the accuracy and consistency of robot pose estimation and map building. Here's an overview of Pose Graph Optimization:

+ SLAM and Robotics Context: In robotics, especially in scenarios involving autonomous navigation, a robot needs to estimate its own position and orientation (pose) within an environment and simultaneously build a map of that environment. This is the core problem addressed by SLAM.

+ Pose Graph Representation: Pose Graph Optimization deals with the representation of the robot's trajectory and the observed features (landmarks) in the form of a graph. In this graph:

        Nodes represent robot poses at different points in time.
        Edges represent constraints between these poses, which encode the relative motion information derived from sensors such as odometry, GPS, or IMUs.
        Landmarks, if present, are additional nodes that represent observed features in the environment.
+ Optimization Objective: The primary objective of Pose Graph Optimization is to refine the estimated poses (position and orientation) and the landmarks to minimize the error in the graph. This error is typically measured as the discrepancy between predicted and observed measurements, which could be derived from various sensors, such as cameras, LIDAR, or range finders.

+ Global Consistency: Pose Graph Optimization helps maintain global consistency in the map and pose estimates. It ensures that the entire trajectory of the robot is considered when making adjustments, preventing drift and inconsistencies that can occur when only local information is used.

+ Nonlinear Optimization: Solving the Pose Graph Optimization problem is a nontrivial task, often tackled with nonlinear optimization techniques such as Gauss-Newton or Levenberg-Marquardt. These iterative algorithms iteratively update the poses and landmarks to minimize the error in the graph.

+ Data Association: Accurate data association, linking observed features to landmarks and robot poses, is crucial for the success of Pose Graph Optimization. This is often done using sensor measurements and sensor fusion techniques.

+ Loop Closure Detection: Handling loop closures, instances where the robot revisits a previously explored area, is a critical aspect of Pose Graph Optimization. Detecting loop closures and accurately incorporating them into the optimization process helps correct accumulated errors.

### Residuals and Jacobians

## Pose Graph Using Sophus :

pose graph : [sphere.g2o](https://github.com/gaoxiang12/slambook2/blob/master/ch10/sphere.g2o)

```cpp

#include <g2o/core/base_vertex.h>
#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel_impl.h>
#include <iostream>

#include "common.h"
#include "sophus/se3.hpp"

using namespace Sophus;
using namespace Eigen;
using namespace std;

/// Structure of posture and internal parameters
struct PoseAndIntrinsics {
    PoseAndIntrinsics() {}

    /// set from given data address
    explicit PoseAndIntrinsics(double *data_addr) {
        rotation = SO3d::exp(Vector3d(data_addr[0], data_addr[1], data_addr[2]));
        translation = Vector3d(data_addr[3], data_addr[4], data_addr[5]);
        focal = data_addr[6];
        k1 = data_addr[7];
        k2 = data_addr[8];
    }

    /// Put estimated value into memory
    void set_to(double *data_addr) {
        auto r = rotation.log();
        for (int i = 0; i < 3; ++i) data_addr[i] = r[i];
        for (int i = 0; i < 3; ++i) data_addr[i + 3] = translation[i];
        data_addr[6] = focal;
        data_addr[7] = k1;
        data_addr[8] = k2;
    }

    SO3d rotation;
    Vector3d translation = Vector3d::Zero();
    double focal = 0;
    double k1 = 0, k2 = 0;
};

/// The vertex of the pose plus camera internal parameters, 9 dimensions, the first three dimensions are 
/// so3, and the next are t, f, k1, k2
class VertexPoseAndIntrinsics : public g2o::BaseVertex<9, PoseAndIntrinsics> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexPoseAndIntrinsics() {}

    virtual void setToOriginImpl() override {
        _estimate = PoseAndIntrinsics();
    }

    virtual void oplusImpl(const double *update) override {
        _estimate.rotation = SO3d::exp(Vector3d(update[0], update[1], update[2])) * _estimate.rotation;
        _estimate.translation += Vector3d(update[3], update[4], update[5]);
        _estimate.focal += update[6];
        _estimate.k1 += update[7];
        _estimate.k2 += update[8];
    }

/// Project a point based on the estimated value
    Vector2d project(const Vector3d &point) {
        Vector3d pc = _estimate.rotation * point + _estimate.translation;
        pc = -pc / pc[2];
        double r2 = pc.squaredNorm();
        double distortion = 1.0 + r2 * (_estimate.k1 + _estimate.k2 * r2);
        return Vector2d(_estimate.focal * distortion * pc[0],
                        _estimate.focal * distortion * pc[1]);
    }

    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}
};

class VertexPoint : public g2o::BaseVertex<3, Vector3d> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    VertexPoint() {}

    virtual void setToOriginImpl() override {
        _estimate = Vector3d(0, 0, 0);
    }

    virtual void oplusImpl(const double *update) override {
        _estimate += Vector3d(update[0], update[1], update[2]);
    }

    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}
};

class EdgeProjection :
    public g2o::BaseBinaryEdge<2, Vector2d, VertexPoseAndIntrinsics, VertexPoint> {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

    virtual void computeError() override {
        auto v0 = (VertexPoseAndIntrinsics *) _vertices[0];
        auto v1 = (VertexPoint *) _vertices[1];
        auto proj = v0->project(v1->estimate());
        _error = proj - _measurement;
    }

    // use numeric derivatives
    virtual bool read(istream &in) {}

    virtual bool write(ostream &out) const {}

};

void SolveBA(BALProblem &bal_problem);

int main(int argc, char **argv) {

    if (argc != 2) {
        cout << "usage: bundle_adjustment_g2o bal_data.txt" << endl;
        return 1;
    }

    BALProblem bal_problem(argv[1]);
    bal_problem.Normalize();
    bal_problem.Perturb(0.1, 0.5, 0.5);
    bal_problem.WriteToPLYFile("initial.ply");
    SolveBA(bal_problem);
    bal_problem.WriteToPLYFile("final.ply");

    return 0;
}

void SolveBA(BALProblem &bal_problem) {
    const int point_block_size = bal_problem.point_block_size();
    const int camera_block_size = bal_problem.camera_block_size();
    double *points = bal_problem.mutable_points();
    double *cameras = bal_problem.mutable_cameras();

    // pose dimension 9, landmark is 3
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<9, 3>> BlockSolverType;
    typedef g2o::LinearSolverCSparse<BlockSolverType::PoseMatrixType> LinearSolverType;
    // use LM
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
        g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    /// build g2o problem
    const double *observations = bal_problem.observations();
    // vertex
    vector<VertexPoseAndIntrinsics *> vertex_pose_intrinsics;
    vector<VertexPoint *> vertex_points;

    for (int i = 0; i < bal_problem.num_cameras(); ++i) {
        VertexPoseAndIntrinsics *v = new VertexPoseAndIntrinsics();
        double *camera = cameras + camera_block_size * i;
        v->setId(i);
        v->setEstimate(PoseAndIntrinsics(camera));
        optimizer.addVertex(v);
        vertex_pose_intrinsics.push_back(v);
    }
    for (int i = 0; i < bal_problem.num_points(); ++i) {
        VertexPoint *v = new VertexPoint();
        double *point = points + point_block_size * i;
        v->setId(i + bal_problem.num_cameras());
        v->setEstimate(Vector3d(point[0], point[1], point[2]));
// g2o needs to manually set the vertex to be Marg in BA
        v->setMarginalized(true);
        optimizer.addVertex(v);
        vertex_points.push_back(v);
    }

    // edge
    for (int i = 0; i < bal_problem.num_observations(); ++i) {
        EdgeProjection *edge = new EdgeProjection;
        edge->setVertex(0, vertex_pose_intrinsics[bal_problem.camera_index()[i]]);
        edge->setVertex(1, vertex_points[bal_problem.point_index()[i]]);
        edge->setMeasurement(Vector2d(observations[2 * i + 0], observations[2 * i + 1]));
        edge->setInformation(Matrix2d::Identity());
        edge->setRobustKernel(new g2o::RobustKernelHuber());
        optimizer.addEdge(edge);
    }

    optimizer.initializeOptimization();
    optimizer.optimize(40);

    // set to bal problem
    for (int i = 0; i < bal_problem.num_cameras(); ++i) {
        double *camera = cameras + camera_block_size * i;
        auto vertex = vertex_pose_intrinsics[i];
        auto estimate = vertex->estimate();
        estimate.set_to(camera);
    }
    for (int i = 0; i < bal_problem.num_points(); ++i) {
        double *point = points + point_block_size * i;
        auto vertex = vertex_points[i];
        for (int k = 0; k < 3; ++k) point[k] = vertex->estimate()[k];
    }
}

```