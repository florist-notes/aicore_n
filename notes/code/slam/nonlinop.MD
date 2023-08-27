# Nonlinear Optimization :

Nonlinear optimization is a powerful mathematical framework employed to address complex real-world problems across various domains, including engineering, economics, machine learning, and scientific research. Unlike linear optimization, which deals with linear relationships between variables, nonlinear optimization tackles situations where the relationships between variables are nonlinear, often exhibiting intricate and non-trivial behavior.

Nonlinear optimization involves the search for the optimal values of decision variables that maximize or minimize a given objective function while satisfying a set of constraints. The challenges in this domain arise from the nonlinearity of the objective function and constraints, which can lead to multiple local optima, discontinuities, and computational complexity.

Various algorithms and techniques have been developed to address these challenges, ranging from classical methods like gradient-based optimization, the Newton-Raphson method, and the simplex method, to more modern approaches like genetic algorithms, particle swarm optimization, and simulated annealing. The choice of method often depends on the problem's characteristics, such as its dimensionality, smoothness, and convexity.

Nonlinear optimization in SLAM focuses on finding the optimal trajectory and map that best explain the observed sensor data, while respecting the motion dynamics and sensor characteristics of the robot. This problem is inherently nonlinear due to the nonlinear relationships between robot poses, map features, and sensor measurements. Furthermore, SLAM operates in real-time, requiring computationally efficient solutions to handle large-scale, high-dimensional optimization problems.

Several nonlinear optimization techniques have been applied to SLAM, including the Extended Kalman Filter (EKF), the Graph-based SLAM (G-SLAM) framework, and more recently, nonlinear optimization solvers like the Gauss-Newton and Levenberg-Marquardt algorithms. These approaches iteratively refine the estimates of robot poses and map features, leveraging the mathematical structure of SLAM problems to converge towards an accurate solution.

The applications of nonlinear optimization in SLAM are extensive, spanning from indoor robotics for navigation and mapping to autonomous vehicles, where precise real-time localization and mapping are imperative for safe and efficient operation. Additionally, SLAM techniques are indispensable in fields like augmented reality, where they enable the overlay of virtual objects onto the real world accurately.

### State Estimation: From Batch State Estimation to Least-Square:

Forming the batch state estimation problem into a least-squares framework is a common approach used in various fields, including robotics, navigation, and control. This method is particularly useful when you have multiple measurements collected over time and want to estimate the optimal state vector by minimizing the sum of squared errors. Here's a brief overview of the steps involved in converting the state estimation problem into a least-squares problem:

+ <b>Define the State Vector (x):</b> Begin by defining the state vector, which contains all the variables you want to estimate. This vector may include the robot's pose, velocities, sensor biases, or any other relevant parameters.

+ <b>Formulate the Measurement Model (h(x)):</b> Create a measurement model that relates the state vector to the measurements collected over time. This model should capture how each measurement depends on the state variables. It is typically nonlinear due to sensor characteristics or system dynamics.

+ <b>Gather Measurements (z):</b> Collect all the measurements you have from various sensors and time instances into a vector, denoted as 'z.'

+ <b>Define the Residuals (r):</b> Calculate the residuals, which are the differences between the actual measurements (z) and the predictions made by the measurement model (h(x)). The residuals vector, denoted as 'r,' represents the discrepancies between the measurements and the current state estimate.

      r = z - h(x)

+ <b>Formulate the Cost Function (J)</b>: Construct a cost function that quantifies the total error between the measurements and the state predictions. The cost function is defined as the sum of squared residuals:

      J = ∑(rᵀ * r)

      Here, 'rᵀ' denotes the transpose of the residuals vector 'r.'

+ <b>Minimize the Cost Function:</b> The objective is to find the state vector 'x' that minimizes the cost function 'J.' This can be achieved using optimization techniques such as the Gauss-Newton method, Levenberg-Marquardt algorithm, or gradient descent. These methods iteratively update the state vector 'x' to reduce the cost function until convergence is achieved.

+ <b>Iterate:</b> Repeat the optimization process until the cost function converges to a minimum value, indicating that the estimated state vector 'x' provides the best fit to the measurements.

I have my state estimation notes @github/[state-estimation](https://github.com/florist-notes/aicore_s/blob/main/notes/navself.MD) and optimization notes @github/[math-optimize](https://github.com/florist-notes/alg0rith.math/blob/main/mathematics/README.MD).

Goal : 
+ Understand how to form the batch state estimation problem into a leastsquare and how to solve the least-square problem as well as use the Google Ceres and g2o library to solve a least-square problem.  
+ Understand the Gauss-Newton and Levenburg-Marquardt method and implement them.


Nonlinear Least-Square Problem : The First and Second-Order Method, The Gauss-Newton Method,  The Levernberg-Marquatdt Method [[optimization.pdf](./optimization.pdf)]

### Curve Fitting :

In SLAM, the primary objective is to estimate the robot's trajectory and create a map of the environment by integrating sensor measurements. However, the environment often contains continuous structures like walls, roads, or other features that are best represented as curves rather than discrete points. Accurately modeling these curves is essential for improving both mapping and localization accuracy.

+ <b>Curve Representation:</b> To address this challenge, SLAM optimization aims to represent curves in a parametric form, typically using mathematical equations. Common representations include:

    - <b>Line Segments:</b> For straight features like walls, line segments (linear curves) can be used to represent them efficiently.

    - <b>Polynomial Curves:</b> Higher-order polynomials (quadratic, cubic, etc.) can represent more complex curves in the environment.

    - <b>Splines:</b> Splines, such as B-splines or cubic splines, provide a flexible way to model curves and smooth transitions.

+ <b>Optimization Objective:</b> The curve fitting problem in SLAM optimization involves finding the parameters of these curve representations that best fit the sensor measurements while minimizing the overall error. This is typically achieved through nonlinear optimization techniques.

+ <b>Data Association:</b> One of the challenges in curve fitting within SLAM is associating sensor measurements with the appropriate curve representations. This is often done by detecting features in sensor data, such as corners or edge points, and then fitting curves to these feature points.

+ <b>Continuous-Time SLAM:</b> In some advanced SLAM approaches, particularly in visual SLAM or LiDAR-based mapping, continuous-time SLAM methods are used. These methods leverage differential equations to model the continuous motion and continuous feature representations more accurately. This is particularly beneficial for curve-like features that evolve over time.

+ Applications: Curve fitting in SLAM optimization finds applications in various domains:

    - <b>Robotics:</b> Autonomous vehicles and robots can use curve fitting to accurately map and navigate environments with curved features like roads or tunnels.

    - <b>Augmented Reality:</b> In AR applications, curve fitting can help align virtual objects with real-world curves, enhancing the user's experience.

    - <b>Geospatial Mapping:</b> In mapping and geospatial applications, curve fitting helps represent natural terrain features like rivers and coastlines.

Curve Fitting with Gauss-Newton:  optimize a simple fitting problem iteratively

 ```cpp

#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main(int argc, char **argv) {
   double ar = 1.0, br = 2.0, cr = 1.0; // real parameter value
   double ae = 2.0, be = -1.0, ce = 5.0; // estimated parameter values
   int N = 100; // data points
   double w_sigma = 1.0; // noise Sigma value
   double inv_sigma = 1.0 / w_sigma;
   cv::RNG rng; // OpenCV random number generator

   vector<double> x_data, y_data; // data
   for (int i = 0; i < N; i++) {
     double x = i / 100.0;
     x_data.push_back(x);
     y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
   }

   // Start Gauss-Newton iteration
   int iterations = 100; // number of iterations
   double cost = 0, lastCost = 0; // The cost of this iteration and the cost of the previous iteration

   chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
   for (int iter = 0; iter < iterations; iter++) {

     Matrix3d H = Matrix3d::Zero(); // Hessian = J^T W^{-1} J in Gauss-Newton
     Vector3d b = Vector3d::Zero(); // bias
     cost = 0;

     for (int i = 0; i < N; i++) {
       double xi = x_data[i], yi = y_data[i]; // i-th data point
       double error = yi - exp(ae * xi * xi + be * xi + ce);
       Vector3d J; // Jacobian matrix
       J[0] = -xi * xi * exp(ae * xi * xi + be * xi + ce); // de/da
       J[1] = -xi * exp(ae * xi * xi + be * xi + ce); // de/db
       J[2] = -exp(ae * xi * xi + be * xi + ce); // de/dc

       H += inv_sigma * inv_sigma * J * J.transpose();
       b += -inv_sigma * inv_sigma * error * J;

       cost += error * error;
     }

     // Solve the linear equation Hx=b
     Vector3d dx = H.ldlt().solve(b);
     if (isnan(dx[0])) {
       cout << "result is nan!" << endl;
       break;
     }

     if (iter > 0 && cost >= lastCost) {
       cout << "cost: " << cost << ">= last cost: " << lastCost << ", break." << endl;
       break;
     }

     ae += dx[0];
     be += dx[1];
     ce += dx[2];

     lastCost = cost;

     cout << "total cost: " << cost << ", \t\tupdate: " << dx.transpose() <<
          "\t\testtimated params: " << ae << "," << be << "," << ce << endl;
   }

   chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
   chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
   cout << "solve time cost = " << time_used. count() << " seconds." << endl;

   cout << "estimated abc = " << ae << ", " << be << ", " << ce << endl;
   return 0;
}
 
 ```

Google [Ceres](https://github.com/ceres-solver/ceres-solver) is a widely used optimization library for least-square problems. In
Ceres, as users, we only need to define the optimization problem to be solved according to specific steps and then hand it over to the solver for calculation. Ceres for Curve Fitting:

```cpp

#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>

using namespace std;

// Calculation model of the cost function
struct CURVE_FITTING_COST {
   CURVE_FITTING_COST(double x, double y) : _x(x), _y(y) {}

   // Calculation of residuals
   template<typename T>
   bool operator()(
     const T *const abc, // model parameters, with 3 dimensions
     T *residual) const {
     residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1] * T(_x) + abc[2]); // y- exp(ax^2+bx+c)
     return true;
   }

   const double _x, _y; // x, y data
};

int main(int argc, char **argv) {
   double ar = 1.0, br = 2.0, cr = 1.0; // real parameter value
   double ae = 2.0, be = -1.0, ce = 5.0; // estimated parameter values
   int N = 100; // data points
   double w_sigma = 1.0; // noise Sigma value
   double inv_sigma = 1.0 / w_sigma;
   cv::RNG rng; // OpenCV random number generator

   vector<double> x_data, y_data; // data
   for (int i = 0; i < N; i++) {
     double x = i / 100.0;
     x_data.push_back(x);
     y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
   }

   double abc[3] = {ae, be, ce};

   // Construct the least squares problem
   ceres::Problem problem;
   for (int i = 0; i < N; i++) {
     problem.AddResidualBlock( // add error term to the problem
       // Use automatic derivation, template parameters: error type, output dimension, 
       // input dimension, dimension must be consistent with the previous struct
       new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3>(
         new CURVE_FITTING_COST(x_data[i], y_data[i])
       ),
       nullptr, // Kernel function, not used here, is empty
       abc // parameters to be estimated
     );
   }

   // configure the solver
   ceres::Solver::Options options; // There are many configuration items here that can be filled
   options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY; // How to solve the incremental equation
   options.minimizer_progress_to_stdout = true; // output to cout

   ceres::Solver::Summary summary; // optimization information
   chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
   ceres::Solve(options, &problem, &summary); // start optimization
   chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
   chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
   cout << "solve time cost = " << time_used. count() << " seconds." << endl;

   // output result
   cout << summary. BriefReport() << endl;
   cout << "estimated a,b,c = ";
   for (auto a:abc) cout << a << " ";
   cout << endl;

   return 0;
}

```

### Graph Optimization Theory : 

Graph optimization is a theory that combines nonlinear optimization with graph theory. Graph optimization theory plays a central role in addressing one of the most fundamental challenges in robotics and computer vision: Simultaneous Localization and Mapping (SLAM). SLAM is the problem of a robot or a sensor system navigating an unknown environment while simultaneously building a map of that environment and estimating its own position within it. Graph optimization theory provides a powerful framework to tackle the intricacies of this problem. Here's a note on how graph optimization theory is applied in the context of SLAM:

+ Graph Representation: In SLAM, the world and the robot's movements are represented as a graph. This graph consists of nodes and edges, where:

    - Nodes: Represent states or poses of the robot at different points in time (e.g., position and orientation).

    - Edges: Capture the relationships or constraints between these poses, typically derived from sensor measurements. These constraints can include relative motion between poses and relative observations of landmarks in the environment.

+ Optimization Objective: The goal of SLAM is to find the most likely configuration of robot poses and landmarks (the optimal state) that best explains the observed data while satisfying all the constraints in the graph. This is achieved by minimizing an objective function, often referred to as the "error" or "cost" function. The error function quantifies the discrepancy between predicted and observed measurements. The optimization problem can be mathematically expressed as:

    Minimize J(x) = Σ(rᵀ * Ω * r)

    - J(x): The objective function to minimize.
    - r: Residuals representing the difference between predicted and observed measurements.
    - Ω: A weight matrix that accounts for the uncertainty of measurements.
    - Σ: Summation over all constraints in the graph.

+ Optimization Techniques: Graph optimization in SLAM is typically solved using iterative optimization techniques such as:

    - Gauss-Newton Method: A gradient-based optimization method that iteratively refines the robot's pose estimates by linearizing the constraints.

    - Levenberg-Marquardt Algorithm: An extension of Gauss-Newton that adds a regularization term to improve robustness and handling of ill-conditioned problems.

    - Sparse Bundle Adjustment: Optimizes the graph structure by exploiting the sparsity of the constraints, making it computationally efficient for large-scale SLAM problems.


In order to use [g2o](https://github.com/RainerKuemmerle/g2o), the curve fitting problem must first be constructed into graph
optimization. In this process, just remember that nodes are the optimization variables,
and edges are the error terms. 

```cpp

#include <iostream>
#include <g2o/core/g2o_core_api.h>
#include <g2o/core/base_vertex.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/core/optimization_algorithm_dogleg.h>
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>
#include <opencv2/core/core.hpp>
#include <cmath>
#include <chrono>

using namespace std;

// The vertices of the curve model, template parameters: optimize variable dimensions and data types
class CurveFittingVertex : public g2o::BaseVertex<3, Eigen::Vector3d> {
public:
   EIGEN_MAKE_ALIGNED_OPERATOR_NEW

   // reset
   virtual void setToOriginImpl() override {
     _estimate << 0, 0, 0;
   }

   // renew
   virtual void oplusImpl(const double *update) override {
     _estimate += Eigen::Vector3d(update);
   }

   // Save and read: leave blank
   virtual bool read(istream &in) {}

   virtual bool write(ostream &out) const {}
};

// Error model template parameters: observation dimension, type, connection vertex type
class CurveFittingEdge : public g2o::BaseUnaryEdge<1, double, CurveFittingVertex> {
public:
   EIGEN_MAKE_ALIGNED_OPERATOR_NEW

   CurveFittingEdge(double x) : BaseUnaryEdge(), _x(x) {}

   // Calculate curve model error
   virtual void computeError() override {
     const CurveFittingVertex *v = static_cast<const CurveFittingVertex *> (_vertices[0]);
     const Eigen::Vector3d abc = v->estimate();
     _error(0, 0) = _measurement - std::exp(abc(0, 0) * _x * _x + abc(1, 0) * _x + abc(2, 0));
   }

   // Compute the Jacobian matrix
   virtual void linearizeOplus() override {
     const CurveFittingVertex *v = static_cast<const CurveFittingVertex *> (_vertices[0]);
     const Eigen::Vector3d abc = v->estimate();
     double y = exp(abc[0] * _x * _x + abc[1] * _x + abc[2]);
     _jacobianOplusXi[0] = -_x * _x * y;
     _jacobianOplusXi[1] = -_x * y;
     _jacobianOplusXi[2] = -y;
   }

   virtual bool read(istream &in) {}

   virtual bool write(ostream &out) const {}

public:
   double _x; // x value, y value _measurement
};

int main(int argc, char **argv) {
   double ar = 1.0, br = 2.0, cr = 1.0; // real parameter value
   double ae = 2.0, be = -1.0, ce = 5.0; // estimated parameter values
   int N = 100; // data points
   double w_sigma = 1.0; // noise Sigma value
   double inv_sigma = 1.0 / w_sigma;
   cv::RNG rng; // OpenCV random number generator

   vector<double> x_data, y_data; // data
   for (int i = 0; i < N; i++) {
     double x = i / 100.0;
     x_data.push_back(x);
     y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
   }

   // Build graph optimization, first set g2o
   typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> BlockSolverType; 
   // The optimization variable dimension of each error term is 3, and the error value dimension is 1
   typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType; // linear solver type

   // Gradient descent method, you can choose from GN, LM, DogLeg
   auto solver = new g2o::OptimizationAlgorithmGaussNewton(
     g2o::make_unique<BlockSolverType>(g2o::make_unique<LinearSolverType>()));
   g2o::SparseOptimizer optimizer; // graph model
   optimizer.setAlgorithm(solver); // set the solver
   optimizer.setVerbose(true); // turn on debug output

   // Add vertices to the graph
   CurveFittingVertex *v = new CurveFittingVertex();
   v->setEstimate(Eigen::Vector3d(ae, be, ce));
   v->setId(0);
   optimizer. addVertex(v);

   // Add edges to the graph
   for (int i = 0; i < N; i++) {
     CurveFittingEdge *edge = new CurveFittingEdge(x_data[i]);
     edge->setId(i);
     edge->setVertex(0, v); // set the connected vertices
     edge->setMeasurement(y_data[i]); // observation value
     edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 / (w_sigma * w_sigma)); 
     // information matrix: inverse of covariance matrix
     optimizer. addEdge(edge);
   }

   // perform optimization
   cout << "start optimization" << endl;
   chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
   optimizer.initializeOptimization();
   optimizer. optimize(10);
   chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
   chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
   cout << "solve time cost = " << time_used. count() << " seconds." << endl;

   // output optimized value
   Eigen::Vector3d abc_estimate = v->estimate();
   cout << "estimated model: " << abc_estimate.transpose() << endl;

   return 0;
}

```