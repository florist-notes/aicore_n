# Nonlinear Optimization :

Goal : Understand how to form the batch state estimation problem into a leastsquare and how to solve the least-square problem as well as use the Google Ceres and g2o library to solve a least-square problem.  Understand the Gauss-Newton and Levenburg-Marquardt method and implement them.

State Estimation: From Batch State Estimation to Least-Square

Nonlinear Least-Square Problem : The First and Second-Order Method, The Gauss-Newton Method,  The Levernberg-Marquatdt Method

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
       // Use automatic derivation, template parameters: error type, output dimension, input dimension, dimension must be consistent with the previous struct
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

Graph Optimization Theory : Graph optimization is a theory that combines nonlinear optimization with graph theory.


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
   typedef g2o::BlockSolver<g2o::BlockSolverTraits<3, 1>> BlockSolverType; // The optimization variable dimension of each error term is 3, and the error value dimension is 1
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
     edge->setInformation(Eigen::Matrix<double, 1, 1>::Identity() * 1 / (w_sigma * w_sigma)); // information matrix: inverse of covariance matrix
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