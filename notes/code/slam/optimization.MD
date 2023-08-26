# Filters and Optimizations (Backend)

Backend optimization in Simultaneous Localization and Mapping (SLAM) is a critical aspect of improving the performance and accuracy of SLAM systems. SLAM is a fundamental technology used in robotics and autonomous systems to map an unknown environment while simultaneously estimating the robot's pose within that environment. The backend optimization phase plays a crucial role in refining the map and pose estimates based on the collected sensor data. Here are some key points to consider regarding backend optimization in SLAM:

+ <b>Data Fusion:</b> Backend optimization involves fusing data from various sensors, such as LIDAR, cameras, IMUs (Inertial Measurement Units), and odometry, to create a consistent and accurate map of the environment. This fusion process helps reduce sensor noise and corrects for discrepancies in sensor readings.

+ <b>Graph Optimization:</b> SLAM is often formulated as a graph optimization problem. A graph is constructed, where nodes represent robot poses at different time steps, and edges represent measurements or constraints between these poses. Backend optimization techniques optimize this graph to find the most likely trajectory and map that explains the sensor data.

+ <b>Bundle Adjustment:</b> Bundle adjustment is a common technique used in backend optimization. It iteratively refines the robot's trajectory and the map to minimize the error between predicted and observed sensor measurements. This process enhances the accuracy of the map and the robot's pose estimate.

+ <b>Loop Closure Detection:</b> Backend optimization also includes the detection of loop closures, which are points in the robot's trajectory where it revisits a previously visited location. Detecting and correcting for loop closures is essential for preventing drift in the map and pose estimates.

+ <b>Global Consistency:</b> Backend optimization aims to ensure global consistency in the map and pose estimates. This means that the entire map should be consistent, and the robot's estimated trajectory should align with the observed data across the entire mission.

+ <b>Sparse vs. Dense Approaches:</b> Depending on the complexity of the SLAM problem and available computational resources, backend optimization can be implemented using sparse or dense techniques. Sparse methods focus on optimizing key poses and landmarks, while dense methods optimize a more detailed representation of the map.

+ <b>Computational Efficiency:</b> Achieving real-time performance is often a challenge in SLAM systems. Backend optimization algorithms need to be computationally efficient to process large volumes of sensor data in real-time.

Goal :
+ Learn how to formulate the backend problem into a filter or least-square
optimization problem.
+ Learn how to use the sparse structure in the bundle adjustment problem.
+ Solve a BA problem with g2o and Ceres.

State Estimation from Probabilistic Perspective: I have state estimation notes at @github/[state-estimate](https://github.com/florist-notes/aicore_s/blob/main/notes/navself.MD).

### Linear Systems and the Kalman Filter

I have brief notes on linear systems and the Kalman Filter family at @github/[lin-kf](https://github.com/florist-notes/aicore_s/blob/main/notes/perc.MD). 

### Nonlinear Systems and the EKF

I have good notes on optimization at @github/[optimize](https://github.com/florist-notes/alg0rith.math/blob/main/mathematics/op.MD) and nonlinear optimization at @github/[nonlin-opt](./nonlinop.MD).

## Bundle Adjustment and Graph Optimization : 

Bundle Adjustment refers to optimizing both camera parameters (intrinsic and extrinsic) and 3D landmarks with images. Consider the bundles of light rays emitted from 3D points. They are projected into the image planes of several cameras and then detected as feature points. The purpose of optimization can be explained as to adjust the camera poses and the 3D points, to ensure the projected 2D features (bundles) match the detected results.

Bundle Adjustment (BA) is a fundamental technique in the field of computer vision and 3D reconstruction, essential for refining the accuracy of camera pose estimates and 3D point cloud reconstructions in the presence of noisy or inaccurate data. It plays a pivotal role in applications such as Structure from Motion (SfM), Simultaneous Localization and Mapping (SLAM), and photogrammetry, where 3D scenes are reconstructed from a collection of 2D images or video frames.

Here are the key aspects of Bundle Adjustment:

+ <b>Global Optimization:</b> Bundle Adjustment is a global optimization technique. It aims to simultaneously refine the camera poses (positions and orientations) and the 3D coordinates of scene points to minimize the reprojection error, which is the discrepancy between the observed image points and their corresponding projections from the estimated 3D points.

+ <b>Nonlinear Optimization:</b> BA employs nonlinear optimization algorithms, often Levenberg-Marquardt or Gauss-Newton methods. These iterative techniques adjust the camera parameters and 3D point coordinates incrementally to reduce the error.

+ <b>Batch and Incremental Modes:</b> Bundle Adjustment can be performed in batch mode, where all the data (images and corresponding 2D-3D correspondences) are processed at once, or in incremental mode, which updates the estimates as new data becomes available. Batch BA tends to produce more accurate results but can be computationally intensive, while incremental BA is useful for real-time applications like SLAM.

+ <b>Sparsity Exploitation:</b> To manage the computational complexity, BA algorithms take advantage of the sparsity of the Jacobian matrices involved. This is particularly important when dealing with a large number of cameras and 3D points.

+ <b>Loop Closure and Global Consistency:</b> In SLAM and large-scale reconstruction scenarios, loop closures (when the camera revisits a previously observed area) can introduce inconsistencies. BA helps correct these inconsistencies, ensuring global consistency in the reconstructed scene.

+ <b>Noise Handling:</b> BA is capable of handling various sources of noise, such as camera calibration errors, image feature detection errors, and inaccuracies in camera poses. It iteratively refines the parameters to mitigate the effects of noise and uncertainty in the input data.

+ <b>Applications:</b> Bundle Adjustment is widely used in computer vision for applications like 3D modeling, autonomous navigation, augmented reality, and cultural heritage preservation, among others. It is instrumental in creating highly accurate 3D reconstructions from images and videos.

[ The Projection Model and Cost Function, Sparsity, Schur Trick, Robust Kernels, [The Basics about Bundle Adjustment](https://youtu.be/sobyKHwgB0Y?si=NixfXMeDrlXPWL1B), [Structure-from-Motion: Bundle Adjustment](https://youtu.be/MyrVDUnaqUs?si=bdePwLjf1gdBR-Mh), [The Numerics of Bundle Adjustment](https://youtu.be/LKDLcKrWOIU?si=5vhsCxIdjAmg_VWB) ]

### Solve the BA problem : [Bundle Adjustment in the Large (BAL) dataset](https://grail.cs.washington.edu/projects/bal/)

+++ Solving BA in Ceres : 

Define the projection error model:

```cpp

#ifndef SnavelyReprojection_H
#define SnavelyReprojection_H

#include <iostream>
#include "ceres/ceres.h"
#include "rotation.h"

class SnavelyReprojectionError {
public:
    SnavelyReprojectionError(double observation_x, double observation_y) : observed_x(observation_x),
                                                                           observed_y(observation_y) {}

    template<typename T>
    bool operator()(const T *const camera,
                    const T *const point,
                    T *residuals) const {
        // camera[0,1,2] are the angle-axis rotation
        T predictions[2];
        CamProjectionWithDistortion(camera, point, predictions);
        residuals[0] = predictions[0] - T(observed_x);
        residuals[1] = predictions[1] - T(observed_y);

        return true;
    }

    // camera : 9 dims array
    // [0-2] : angle-axis rotation
    // [3-5] : translateion
    // [6-8] : camera parameter, [6] focal length, [7-8] second and forth order radial distortion
    // point : 3D location.
    // predictions : 2D predictions with center of the image plane.
    template<typename T>
    static inline bool CamProjectionWithDistortion(const T *camera, const T *point, T *predictions) {
        // Rodrigues' formula
        T p[3];
        AngleAxisRotatePoint(camera, point, p);
        // camera[3,4,5] are the translation
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        // Compute the center fo distortion
        T xp = -p[0] / p[2];
        T yp = -p[1] / p[2];

        // Apply second and fourth order radial distortion
        const T &l1 = camera[7];
        const T &l2 = camera[8];

        T r2 = xp * xp + yp * yp;
        T distortion = T(1.0) + r2 * (l1 + l2 * r2);

        const T &focal = camera[6];
        predictions[0] = focal * distortion * xp;
        predictions[1] = focal * distortion * yp;

        return true;
    }

    static ceres::CostFunction *Create(const double observed_x, const double observed_y) {
        return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
            new SnavelyReprojectionError(observed_x, observed_y)));
    }

private:
    double observed_x;
    double observed_y;
};

#endif // SnavelyReprojection.h

```

BA with Ceres :

```cpp

#include <iostream>
#include <ceres/ceres.h>
#include "common.h"
#include "SnavelyReprojectionError.h"

using namespace std;

void SolveBA(BALProblem &bal_problem);

int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "usage: bundle_adjustment_ceres bal_data.txt" << endl;
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

    // Observations is 2 * num_observations long array observations
    // [u_1, u_2, ... u_n], where each u_i is two dimensional, the x
    // and y position of the observation.
    const double *observations = bal_problem.observations();
    ceres::Problem problem;

    for (int i = 0; i < bal_problem.num_observations(); ++i) {
        ceres::CostFunction *cost_function;

        // Each Residual block takes a point and a camera as input
        // and outputs a 2 dimensional Residual
        cost_function = SnavelyReprojectionError::Create(observations[2 * i + 0], observations[2 * i + 1]);

        // If enabled use Huber's loss function.
        ceres::LossFunction *loss_function = new ceres::HuberLoss(1.0);

        // Each observation corresponds to a pair of a camera and a point
        // which are identified by camera_index()[i] and point_index()[i]
        // respectively.
        double *camera = cameras + camera_block_size * bal_problem.camera_index()[i];
        double *point = points + point_block_size * bal_problem.point_index()[i];

        problem.AddResidualBlock(cost_function, loss_function, camera, point);
    }

    // show some information here ...
    std::cout << "bal problem file loaded..." << std::endl;
    std::cout << "bal problem have " << bal_problem.num_cameras() << " cameras and "
              << bal_problem.num_points() << " points. " << std::endl;
    std::cout << "Forming " << bal_problem.num_observations() << " observations. " << std::endl;

    std::cout << "Solving ceres BA ... " << endl;
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::LinearSolverType::SPARSE_SCHUR;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << "\n";
}

```

BA with g2o :

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

resources: [Simultaneous Localization and Mapping: Through the Lens of Nonlinear Optimization](https://arxiv.org/pdf/2112.05921.pdf), [A Comparison of Particle Filter and Graph-based Optimization for Localization with Landmarks in Automated Vehicles ](https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/wilbers2019irc-cpfg.pdf), [A comparison of different approaches to solve the SLAM problem on a Formula Student Driverless race car](https://www.mrt.kit.edu/z/publ/download/LeLarge_Thesis_SLAM_FormulaStudent.pdf), [SLAM Back-End Optimization Algorithm Based on Vision Fusion IPS](https://www.mdpi.com/1424-8220/22/23/9362), [A Review of Visual-Inertial Simultaneous Localization and Mapping from Filtering-Based and Optimization-Based Perspectives](https://www.mdpi.com/2218-6581/7/3/45), [A Hybrid Visual-Based SLAM Architecture: Local Filter-Based SLAM with KeyFrame-Based Global Mapping](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8749760/), [A Review of Optimisation Strategies used in Simultaneous Localisation and Mapping](https://opus.lib.uts.edu.au/bitstream/10453/130036/1/JCD_v2.pdf), [Robust, Visual-Inertial State Estimation: from Frame-based to Event-based Cameras](https://youtu.be/5-fqEGgt-b8?si=TQT8yXr1LAoGloaP).