# Lie Group and Lie Algebra :

Goal : Learn the concept of Lie group, Lie algebra, and their applications of
SO(3), SE(3) and the corresponding Lie algebras. Learn the meaning and usage of the BCH (Baker-Campbell-Hausdorff) formula. Learn the perturbation model on Lie algebra. Use Sophus to perform operations on Lie algebras.

Thee rotation matrix (we have to estimate and optimize them in addition to the representation) itself is a constrained (orthogonal, and
the determinant is 1) matrix. When used as optimization variables, it introduces additional constraints on matrices that make optimization difficult. Through the transformation relationship between Lie group and Lie algebra, we can turn the pose
estimation into an unconstrained optimization problem and simplify the solution.


Three-dimensional rotation matrix constitutes the special orthogonal group SO(3), and the transformation matrix constitutes the special Euclidean group SE(3). They are written like this:


$$ SO (3) = { R ∈ R^{3x3} | RR^T = I, det(R) = 1}. $$

and

$$ SE(3) = \begin{cases} T =  \begin{bmatrix} R & t \\[0.3em] 0^T & 1 \end{bmatrix} ∈  R^{4x4} | R ∈  SO(3), t ∈  R^3 \end{cases}  $$


Lie Algebra : [Why study Lie theory? | Lie groups, algebras, brackets](https://youtu.be/IlqVo3sJFLE), [Lie Groups and Lie Algebra](https://www.youtube.com/watch?v=kN-LZvrKJck&list=PLN_4R2IuNuuRgJb00X2J53Iq9qe7k1nyr), [Lie Groups and Lie Algebra](https://www.youtube.com/watch?v=fXJcoXeiPD8&list=PLRlVmXqzHjURZO0fviJuyikvKlGS6rXrb)

Exponential and Logarithmic Mapping :  Exponential Map of SO(3) and Exponential Map of SE(3)

 Lie Algebra Derivation and Perturbation Model : BCH Formula and Its Approximation, Derivative on SO(3), Derivative Model, Perturbation Model and Derivative on SE(3).

A good Lie algebra library is the [Sophus library](https://github.com/strasdat/Sophus) maintained by Strasdat.

```cpp

#include <iostream>
#include <cmath>
using namespace std;

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "sophus/so3.h"
#include "sophus/se3.h"

int main( int argc, char** argv )
{
     // Rotation matrix for 90 degrees along the Z axis
     Eigen::Matrix3d R = Eigen::AngleAxisd(M_PI/2, Eigen::Vector3d(0,0,1)).toRotationMatrix();
    
     Sophus::SO3 SO3_R(R); // Sophus::SO(3) can be constructed directly from the rotation matrix
     Sophus::SO3 SO3_v( 0, 0, M_PI/2 ); // can also be constructed from a rotation vector
     Eigen::Quaterniond q(R); // or quaternion
     Sophus::SO3 SO3_q( q );
     // The above expressions are all equivalent
     // When SO(3) is output, output in the form of so(3)
     cout<<"SO(3) from matrix: "<<SO3_R<<endl;
     cout<<"SO(3) from vector: "<<SO3_v<<endl;
     cout<<"SO(3) from quaternion :"<<SO3_q<<endl;
    
     // Get its Lie algebra using the logarithmic map
     Eigen::Vector3d so3 = SO3_R.log();
     cout<<"so3 = "<<so3.transpose()<<endl;
     // hat is vector to anti-symmetric matrix
     cout<<"so3 hat=\n"<<Sophus::SO3::hat(so3)<<endl;
     // Relatively, vee is antisymmetric to vector
     cout<<"so3 hat vee= "<<Sophus::SO3::vee( Sophus::SO3::hat(so3) ).transpose()<<endl; // transpose is purely for beautiful output
    
     // update of the incremental perturbation model
     Eigen::Vector3d update_so3(1e-4, 0, 0); //Assume that the update amount is so much
     Sophus::SO3 SO3_updated = Sophus::SO3::exp(update_so3)*SO3_R;
     cout<<"SO3 updated = "<<SO3_updated<<endl;
    
     /******************** Adorable dividing line************************ *****/
     cout<<"************I am the dividing line************"<<endl;
     // The operations on SE(3) are similar
     Eigen::Vector3d t(1,0,0); // translate 1 along the X axis
     Sophus::SE3 SE3_Rt(R, t); // Construct SE(3) from R, t
     Sophus::SE3 SE3_qt(q,t); // Construct SE(3) from q,t
     cout<<"SE3 from R,t="<<endl<<SE3_Rt<<endl;
     cout<<"SE3 from q,t= "<<endl<<SE3_qt<<endl;
     // Lie algebra se(3) is a six-dimensional vector, for convenience, first typedef it
     typedef Eigen::Matrix<double,6,1> Vector6d;
     Vector6d se3 = SE3_Rt. log();
     cout<<"se3 = "<<se3.transpose()<<endl;
     // Observe the output, you will find that in Sophus, the translation of se(3) is in front, and the rotation is in the back.
     // Similarly, there are two operators hat and vee
     cout<<"se3 hat = "<<endl<<Sophus::SE3::hat(se3)<<endl;
     cout<<"se3 hat vee = "<<Sophus::SE3::vee( Sophus::SE3::hat(se3) ).transpose()<<endl;
    
     // Finally, demonstrate the update
     Vector6d update_se3; //update amount
     update_se3.setZero();
     update_se3(0,0) = 1e-4d;
     Sophus::SE3 SE3_updated = Sophus::SE3::exp(update_se3)*SE3_Rt;
     cout<<"SE3 updated = "<<endl<<SE3_updated.matrix()<<endl;
    
     return 0;
}

```

In CMAkeLists.txt we add : `find_package( Sophus REQUIRED )`, `include_directories( ${Sophus_INCLUDE_DIRS} )` and `add_executable( useSophus useSophus.cpp )`.

Evaluating the Trajectory :  Difference between the estimated trajectory of an algorithm and the real trajectory to evaluate the algorithm’s accuracy. Calculate the absolute trajectory error and display it in a 3D window. There are many kinds of error indicators. The common used one is absolute
trajectory error, which is like:

$$ ATE_{all} =  \sqrt{ \dfrac{1}{N} \sum_{i=1}^{N} || log(T^{-1}_{realtraj,i} T_{estimat,i})^v ||  \dfrac{2}{2}, } $$

This is actually the root-mean-squared error (RMSE) for each pose in Lie algebra.
This error can describe both the rotation and translation errors

$$ ATE_{trans} =  \sqrt{ \dfrac{1}{N} \sum_{i=1}^{N} || trans(T^{-1}_{realtraj,i} T_{estimat,i}) ||  \dfrac{2}{2}, } $$


```cpp

#include <iostream>
#include <fstream>
#include <unistd.h>
#include <pangolin/pangolin.h>
#include <sophus/se3.hpp>

using namespace Sophus;
using namespace std;

string groundtruth_file = "./example/groundtruth.txt";
string estimated_file = "./example/estimated.txt";

typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;

void DrawTrajectory(const TrajectoryType &gt, const TrajectoryType &esti);

TrajectoryType ReadTrajectory(const string &path);

int main(int argc, char **argv) {
  TrajectoryType groundtruth = ReadTrajectory(groundtruth_file);
  TrajectoryType estimated = ReadTrajectory(estimated_file);
  assert(!groundtruth.empty() && !estimated.empty());
  assert(groundtruth.size() == estimated.size());

  // compute rmse
  double rmse = 0;
  for (size_t i = 0; i < estimated.size(); i++) {
    Sophus::SE3d p1 = estimated[i], p2 = groundtruth[i];
    double error = (p2.inverse() * p1).log().norm();
    rmse += error * error;
  }
  rmse = rmse / double(estimated.size());
  rmse = sqrt(rmse);
  cout << "RMSE = " << rmse << endl;

  DrawTrajectory(groundtruth, estimated);
  return 0;
}

TrajectoryType ReadTrajectory(const string &path) {
  ifstream fin(path);
  TrajectoryType trajectory;
  if (!fin) {
    cerr << "trajectory " << path << " not found." << endl;
    return trajectory;
  }

  while (!fin.eof()) {
    double time, tx, ty, tz, qx, qy, qz, qw;
    fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
    Sophus::SE3d p1(Eigen::Quaterniond(qw, qx, qy, qz), Eigen::Vector3d(tx, ty, tz));
    trajectory.push_back(p1);
  }
  return trajectory;
}

void DrawTrajectory(const TrajectoryType &gt, const TrajectoryType &esti) {
  // create pangolin window and plot the trajectory
  pangolin::CreateWindowAndBind("Trajectory Viewer", 1024, 768);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrix(1024, 768, 500, 500, 512, 389, 0.1, 1000),
      pangolin::ModelViewLookAt(0, -0.1, -1.8, 0, 0, 0, 0.0, -1.0, 0.0)
  );

  pangolin::View &d_cam = pangolin::CreateDisplay()
      .SetBounds(0.0, 1.0, pangolin::Attach::Pix(175), 1.0, -1024.0f / 768.0f)
      .SetHandler(new pangolin::Handler3D(s_cam));


  while (pangolin::ShouldQuit() == false) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    d_cam.Activate(s_cam);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);

    glLineWidth(2);
    for (size_t i = 0; i < gt.size() - 1; i++) {
      glColor3f(0.0f, 0.0f, 1.0f);  // blue for ground truth
      glBegin(GL_LINES);
      auto p1 = gt[i], p2 = gt[i + 1];
      glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
      glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
      glEnd();
    }

    for (size_t i = 0; i < esti.size() - 1; i++) {
      glColor3f(1.0f, 0.0f, 0.0f);  // red for estimated
      glBegin(GL_LINES);
      auto p1 = esti[i], p2 = esti[i + 1];
      glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
      glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
      glEnd();
    }
    pangolin::FinishFrame();
    usleep(5000);   // sleep 5 ms
  }

}

```