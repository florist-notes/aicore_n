# 3D Rigid Body Motion :

In Simultaneous Localization and Mapping (SLAM), understanding and modeling 3D rigid body motion is essential for accurate localization of a robot or sensor system within its environment and for building a coherent 3D map. Here's a note on the significance of 3D rigid body motion in the context of SLAM:

#### Definition of 3D Rigid Body Motion:

3D rigid body motion refers to the transformation of an object in three-dimensional space without any deformation. It is characterized by translation and rotation. In the context of SLAM:

  - Translation: This represents the linear movement of the robot or sensor system from one point to another in space. It includes changes in position along the X, Y, and Z axes.

  - Rotation: This signifies the angular change in orientation of the robot or sensor system about its center. It includes roll, pitch, and yaw rotations.

#### Localization and Mapping:

In SLAM, the task is to simultaneously estimate the robot's pose (position and orientation) and build a map of its environment. Understanding 3D rigid body motion is crucial for both aspects:

+ Localization: Accurate localization relies on knowing how the robot has moved from one pose to another. This is accomplished by tracking the transformations (translations and rotations) between consecutive poses, often through odometry or sensor measurements.

+ Mapping: Accurate 3D mapping involves aligning sensor measurements (e.g., point clouds from LiDAR or depth data from RGB-D cameras) to the robot's pose. Knowing the robot's 3D motion allows for the correct alignment of these measurements to build a coherent and accurate 3D map of the environment.


Goal :  Study the rigid body geometry in three-dimensional space: rotation matrix,
transformation matrix, quaternion, and Euler angle. Learn the usage of the Eigen library’s matrix and geometry module.

Rotation Matrix : Points, Vectors, and Coordinate Systems. Most 3D libraries use right-handed coordinates (such as OpenGL, 3DS Max, etc.), and some libraries use left-handed coordinates (such as Unity, Direct3D, etc.).

 [ Euclidean Transforms Between Coordinate Systems & Transform Matrix and Homogeneous Coordinates ]

 Practice: Use Eigen : ` $  sudo apt−get install libeigen3−dev `. The Eigen header file’s default location should be `/usr/include/eigen3`  (find out :  ` $ sudo locate eigen3  `)

```cpp
#include <iostream>

using namespace std;

#include <ctime>
// Eigen core part
#include <Eigen/Core>
// Algebraic operations on dense matrices (inverse, eigenvalues, etc.)
#include <Eigen/Dense>

using namespace Eigen;

#define MATRIX_SIZE 50

/****************************
* This program demonstrates the use of Eigen primitive types
*****************************/

int main(int argc, char **argv) {
   // All vectors and matrices in Eigen are Eigen::Matrix, which is a template class.
   // Its first three parameters are: data type, row, column
   // Declare a 2*3 float matrix
   Matrix<float, 2, 3> matrix_23;

   // At the same time, Eigen provides many built-in types through typedef, but the bottom layer is still Eigen::Matrix
   // For example, Vector3d is essentially Eigen::Matrix<double, 3, 1>, which is a three-dimensional vector
   Vector3d v_3d;
   // this is the same
   Matrix<float, 3, 1> vd_3d;

   // Matrix3d is essentially Eigen::Matrix<double, 3, 3>
   Matrix3d matrix_33 = Matrix3d::Zero(); //initialize to zero
   // If you are not sure about the matrix size, you can use a dynamically sized matrix
   Matrix<double, Dynamic, Dynamic> matrix_dynamic;
   // simpler
   MatrixXd matrix_x;
   // There are many more of this type, we will not list them one by one

   // The following is the operation on the Eigen array
   // input data (initialization)
   matrix_23 << 1, 2, 3, 4, 5, 6;
   // output
   cout << "matrix 2x3 from 1 to 6: \n" << matrix_23 << endl;

   // Use () to access elements in the matrix
   cout << "print matrix 2x3: " << endl;
   for (int i = 0; i < 2; i++) {
     for (int j = 0; j < 3; j++) cout << matrix_23(i, j) << "\t";
     cout << endl;
   }

   // matrix and vector multiplication (actually still matrix and matrix)
   v_3d << 3, 2, 1;
   vd_3d << 4, 5, 6;

   // But in Eigen you can't mix two different types of matrices, it's wrong like this
   // Matrix<double, 2, 1> result_wrong_type = matrix_23 * v_3d;
   // should be explicitly converted
   Matrix<double, 2, 1> result = matrix_23.cast<double>() * v_3d;
   cout << "[1,2,3;4,5,6]*[3,2,1]=" << result.transpose() << endl;

   Matrix<float, 2, 1> result2 = matrix_23 * vd_3d;
   cout << "[1,2,3;4,5,6]*[4,5,6]: " << result2.transpose() << endl;

   // Also you can't get the dimensions of the matrix wrong
   // Try to uncomment the following to see what error Eigen will report
   // Eigen::Matrix<double, 2, 3> result_wrong_dimension = matrix_23.cast<double>() * v_3d;

   // some matrix operations
   // The four arithmetic operations will not be demonstrated, just use +-*/ directly.
   matrix_33 = Matrix3d::Random(); // random number matrix
   cout << "random matrix: \n" << matrix_33 << endl;
   cout << "transpose: \n" << matrix_33.transpose() << endl; // transpose
   cout << "sum: " << matrix_33.sum() << endl; // the sum of each element
   cout << "trace: " << matrix_33.trace() << endl; // trace
   cout << "times 10: \n" << 10 * matrix_33 << endl; // number multiplication
   cout << "inverse: \n" << matrix_33.inverse() << endl; // inverse
   cout << "det: " << matrix_33.determinant() << endl; // determinant

   // Eigenvalues
   // Real symmetric matrix can guarantee successful diagonalization
   SelfAdjointEigenSolver<Matrix3d> eigen_solver(matrix_33.transpose() * matrix_33);
   cout << "Eigen values = \n" << eigen_solver.eigenvalues() << endl;
   cout << "Eigen vectors = \n" << eigen_solver.eigenvectors() << endl;

   // We solve the equation matrix_NN * x = v_Nd
   // The size of N is defined in the previous macro, it is generated by a random number
   // Direct inversion is naturally the most direct, but the amount of inversion calculation is large

   Matrix<double, MATRIX_SIZE, MATRIX_SIZE> matrix_NN
       = MatrixXd::Random(MATRIX_SIZE, MATRIX_SIZE);
   matrix_NN = matrix_NN * matrix_NN.transpose(); // guarantee positive semi-definite
   Matrix<double, MATRIX_SIZE, 1> v_Nd = MatrixXd::Random(MATRIX_SIZE, 1);

   clock_t time_stt = clock(); // timing
   // direct inversion
   Matrix<double, MATRIX_SIZE, 1> x = matrix_NN.inverse() * v_Nd;
   cout << "time of normal inverse is "
        << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
   cout << "x = " << x.transpose() << endl;

   //Usually use matrix decomposition, such as QR decomposition, the speed will be much faster
   time_stt = clock();
   x = matrix_NN.colPivHouseholderQr().solve(v_Nd);
   cout << "time of Qr decomposition is "
        << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
   cout << "x = " << x.transpose() << endl;

   // For positive definite matrices, you can also use cholesky decomposition to solve the equation
   time_stt = clock();
   x = matrix_NN.ldlt().solve(v_Nd);
   cout << "time of ldlt decomposition is "
        << 1000 * (clock() - time_stt) / (double) CLOCKS_PER_SEC << "ms" << endl;
   cout << "x = " << x.transpose() << endl;

   return 0;
}


```

This example demonstrates the basic operations and operations of the Eigen
matrix. To compile it, we need to specify the header file directory of Eigen in
the CMakeLists.txt:  `include_directories( "/usr/include/eigen3" )`. Because the Eigen library only has header files, we don’t need to link the program to the library with the `target_link_libraries` statement.

## Rotation Vectors, Euler Angles and Quaternions :

In the field of computer graphics, computer vision, robotics, and many other disciplines, understanding and representing spatial rotations are essential. Four common mathematical representations for representing rotations are rotation matrices, transformation matrices, quaternions, and Euler angles. Here's a brief overview of each:

### Rotation Matrix: A rotation matrix is a 3x3 matrix that describes a rotation in three-dimensional space. It is orthogonal, meaning its columns (or rows) are orthogonal unit vectors.

    Representation: A typical 3x3 rotation matrix looks like this:

```
      | R11 R12 R13 |
      | R21 R22 R23 |
      | R31 R32 R33 |
```

  Advantages: Rotation matrices are intuitive and straightforward to work with. They are commonly used in computer graphics and computer vision.

  Limitations: They can suffer from numerical instability, especially near singularities, and they are less memory-efficient than quaternions.

### Transformation Matrix: A transformation matrix combines a rotation matrix and a translation vector to represent both a rotation and translation in 3D space. It is a 4x4 matrix.

    Representation: A typical 4x4 transformation matrix looks like this:

```
    | R11 R12 R13 Tx |
    | R21 R22 R23 Ty |
    | R31 R32 R33 Tz |
    | 0   0   0   1  |
```

  Advantages: Transformation matrices can represent both rotation and translation in a single matrix, which is useful in computer graphics and robotics.

  Limitations: They can be computationally expensive to work with, especially when dealing with large datasets.

### Quaternion: A quaternion is a four-element mathematical structure that represents rotations in 3D space. Quaternions are compact and can efficiently represent rotations without singularities.

  Representation: A quaternion is typically written as `q = (w, x, y, z)`.

  Advantages: Quaternions are numerically stable, efficient for interpolation, and widely used in robotics, game development, and orientation sensor fusion.

  Limitations: They are less intuitive to understand geometrically than rotation matrices or Euler angles.

### Euler Angle: Euler angles represent rotations as a combination of three separate angles, typically represented as roll, pitch, and yaw.

  Representation: Euler angles are usually represented as (ϕ, θ, ψ), where ϕ represents rotation about the X-axis (roll), θ about the Y-axis (pitch), and ψ about the Z-axis (yaw).

  Advantages: Euler angles are intuitive for certain applications and easy for humans to visualize.

  Limitations: They suffer from the problem of gimbal lock, where certain combinations of rotations can lead to a loss of one degree of freedom, making them less suitable for some applications.

[ [Quaternions and 3d rotation, explained interactively](https://youtu.be/zjMuIxRvygQ), [Visualizing quaternions (4d numbers) with stereographic projection](https://youtu.be/d4EgbgTm0Bg), Affine and Projective Transformation : [What are affine transformations?](https://youtu.be/E3Phj6J287o) ]

Data Structure of the Eigen Geometry Module:


```cpp

#include <iostream>
#include <cmath>

using namespace std;

#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace Eigen;

// This program demonstrates the use of the Eigen geometry module

int main(int argc, char **argv) {

   // The Eigen/Geometry module provides various representations of rotation and translation
   // 3D rotation matrix directly use Matrix3d or Matrix3f
   Matrix3d rotation_matrix = Matrix3d::Identity();
   // The rotation vector uses AngleAxis, its bottom layer is not directly a Matrix,
   //  but the operation can be used as a matrix (because the operator is overloaded)
   AngleAxisd rotation_vector(M_PI / 4, Vector3d(0, 0, 1)); //rotate 45 degrees along the Z axis
   cout. precision(3);
   cout << "rotation matrix =\n" << rotation_vector.matrix() << endl; //Convert to matrix with matrix()
   // can also be assigned directly
   rotation_matrix = rotation_vector.toRotationMatrix();
   // Use AngleAxis to perform coordinate transformation
   Vector3d v(1, 0, 0);
   Vector3d v_rotated = rotation_vector * v;
   cout << "(1,0,0) after rotation (by angle axis) = " << v_rotated.transpose() << endl;
   // or use a rotation matrix
   v_rotated = rotation_matrix * v;
   cout << "(1,0,0) after rotation (by matrix) = " << v_rotated.transpose() << endl;

   // Euler angle: the rotation matrix can be directly converted into Euler angle
   Vector3d euler_angles = rotation_matrix.eulerAngles(2, 1, 0); // ZYX order, ie roll pitch yaw order
   cout << "yaw pitch roll = " << euler_angles.transpose() << endl;

   // Euclidean transformation matrix using Eigen::Isometry
   Isometry3d T = Isometry3d::Identity(); // Although it is called 3d, it is actually a 4*4 matrix
   T.rotate(rotation_vector); // Rotate according to rotation_vector
   T.pretranslate(Vector3d(1, 3, 4)); // Set the translation vector to (1,3,4)
   cout << "Transform matrix = \n" << T.matrix() << endl;

   // use transformation matrix for coordinate transformation
   Vector3d v_transformed = T * v; // equivalent to R*v+t
   cout << "v transformed = " << v_transformed.transpose() << endl;

   // For affine and projective transformations, use Eigen::Affine3d and Eigen::Projective3d, slightly

   // Quaternion
   // You can directly assign AngleAxis to a quaternion, and vice versa
   Quaterniond q = Quaterniond(rotation_vector);
   cout << "quaternion from rotation vector = " << q.coeffs().transpose()
        << endl; // Please note that the order of coeffs is (x, y, z, w), w is the real part,
        //  and the first three are the imaginary part
   // You can also assign the rotation matrix to it
   q = Quaterniond(rotation_matrix);
   cout << "quaternion from rotation matrix = " << q. coeffs(). transpose() << endl;
   // Use quaternion to rotate a vector, just use overloaded multiplication
   v_rotated = q * v; // Note that it is qvq^{-1} in mathematics
   cout << "(1,0,0) after rotation = " << v_rotated.transpose() << endl;
   // Expressed by conventional vector multiplication, it should be calculated as follows
   cout << "should be equal to " << (q * Quaterniond(0, 1, 0, 0) * q. inverse()). coeffs(). transpose() << endl;

   return 0;
}

```

• Rotation matrix ( 3 × 3 ): `Eigen::Matrix3d`.
• Rotation vector ( 3 × 1 ): `Eigen::AngleAxisd`.
• Euler angle ( 3 × 1 ): `Eigen::Vector3d`.
• Quaternion ( 4 × 1 ): `Eigen::Quaterniond`.
• Euclidean transformation matrix ( 4 × 4 ): `Eigen::Isometry3d`.
• Affine transform ( 4 × 4 ): `Eigen::Affine3d`.
• Perspective transformation ( 4 × 4 ): `Eigen::Projective3d`

 This program demonstrates how to use the rotation matrix, rotation
vectors (AngleAxis), Euler angles, and quaternions in Eigen. 

Coordinate Transformation Example:

The pose of the robot 1 is 

$$ q_1 = [0.35, 0.2, 0.3, 0.1]^T ,t_1 = [0.3, 0.1, 0.1]^T . $$

The pose of the robot 2 is 

$$ q_2 = [−0.5, 0.4, −0.1, 0.2]^T ,t_2 = [−0.1, 0.5, 0.3]^T $$

Here q and t express T_{R_k} ,W , k = 1, 2, which is the world to the robot transform matrix.

```cpp

#include <iostream>
#include <vector>
#include <algorithm>
#include <Eigen/Core>
#include <Eigen/Geometry>

using namespace std;
using namespace Eigen;

int main(int argc, char** argv) {
  Quaterniond q1(0.35, 0.2, 0.3, 0.1), q2(-0.5, 0.4, -0.1, 0.2);
  q1.normalize();
  q2.normalize();
  Vector3d t1(0.3, 0.1, 0.1), t2(-0.1, 0.5, 0.3);
  Vector3d p1(0.5, 0, 0.2);

  Isometry3d T1w(q1), T2w(q2);
  T1w.pretranslate(t1);
  T2w.pretranslate(t2);

  Vector3d p2 = T2w * T1w.inverse() * p1;
  cout << endl << p2.transpose() << endl;
  return 0;
}

```

In Linux,
a widely used library in SLAM is the OpenGL-based [Pangolin](https://github.com/stevenlovegrove/Pangolin) library which provides simple OpenGL drawing operations in a window.

```cpp

#include <pangolin/pangolin.h>
#include <Eigen/Core>
#include <unistd.h>


using namespace std;
using namespace Eigen;

// path to trajectory file
string trajectory_file = "./examples/trajectory.txt";

void DrawTrajectory(vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>>);

int main(int argc, char **argv) {

  vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> poses;
  ifstream fin(trajectory_file);
  if (!fin) {
    cout << "cannot find trajectory file at " << trajectory_file << endl;
    return 1;
  }

  while (!fin.eof()) {
    double time, tx, ty, tz, qx, qy, qz, qw;
    fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
    Isometry3d Twr(Quaterniond(qw, qx, qy, qz));
    Twr.pretranslate(Vector3d(tx, ty, tz));
    poses.push_back(Twr);
  }
  cout << "read total " << poses.size() << " pose entries" << endl;

  // draw trajectory in pangolin
  DrawTrajectory(poses);
  return 0;
}

/*******************************************************************************************/
void DrawTrajectory(vector<Isometry3d, Eigen::aligned_allocator<Isometry3d>> poses) {
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
    .SetBounds(0.0, 1.0, 0.0, 1.0, -1024.0f / 768.0f)
    .SetHandler(new pangolin::Handler3D(s_cam));

  while (pangolin::ShouldQuit() == false) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    d_cam.Activate(s_cam);
    glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
    glLineWidth(2);
    for (size_t i = 0; i < poses.size(); i++) {
      Vector3d Ow = poses[i].translation();
      Vector3d Xw = poses[i] * (0.1 * Vector3d(1, 0, 0));
      Vector3d Yw = poses[i] * (0.1 * Vector3d(0, 1, 0));
      Vector3d Zw = poses[i] * (0.1 * Vector3d(0, 0, 1));
      glBegin(GL_LINES);
      glColor3f(1.0, 0.0, 0.0);
      glVertex3d(Ow[0], Ow[1], Ow[2]);
      glVertex3d(Xw[0], Xw[1], Xw[2]);
      glColor3f(0.0, 1.0, 0.0);
      glVertex3d(Ow[0], Ow[1], Ow[2]);
      glVertex3d(Yw[0], Yw[1], Yw[2]);
      glColor3f(0.0, 0.0, 1.0);
      glVertex3d(Ow[0], Ow[1], Ow[2]);
      glVertex3d(Zw[0], Zw[1], Zw[2]);
      glEnd();
    }
    for (size_t i = 0; i < poses.size(); i++) {
      glColor3f(0.0, 0.0, 0.0);
      glBegin(GL_LINES);
      auto p1 = poses[i], p2 = poses[i + 1];
      glVertex3d(p1.translation()[0], p1.translation()[1], p1.translation()[2]);
      glVertex3d(p2.translation()[0], p2.translation()[1], p2.translation()[2]);
      glEnd();
    }
    pangolin::FinishFrame();
    usleep(5000);   // sleep 5 ms
  }
}

```