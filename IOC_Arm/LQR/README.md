# SO-100 2DOF LQR Control (ROS 2)

This folder contains a complete Linear Quadratic Regulator (LQR) control setup
for a reduced 2-DOF SO-100 robotic arm in ROS 2 and Gazebo.

---

## Purpose

Implement and run an outer-loop LQR controller for a 2-DOF subset of the SO-100
robot arm using ROS 2 control interfaces and Gazebo simulation.

---

## Folder Structure

### `so_100_arm/`
Original SO-100 robot assets.

Definition:
Contains meshes, models, and original URDF files for the full 5-DOF SO-100 arm.
These assets are reused without modification for visualization and collision.

---

### `so100_2dof_bringup/`
2-DOF simulation bringup package.

Definition:
Provides a reduced 2-DOF URDF, controller configuration, and Gazebo launch file
to simulate only the selected joints of the SO-100 arm.

Contents:
- `urdf/so_100_arm_2dof.urdf`  
  Reduced URDF defining only Shoulder_Pitch and Elbow joints.
- `config/controllers_2dof.yaml`  
  ROS 2 control configuration for joint trajectory control.
- `launch/gz_2dof.launch.py`  
  Gazebo launch file for spawning the 2-DOF arm with controllers loaded.

---

### `so100_lqr/`
LQR outer-loop controller package.

Definition:
Implements a ROS 2 node that computes LQR control actions from joint state
feedback and sends commands to the joint trajectory controller.

Main file:
- `lqr_outer_loop.py`  
  Subscribes to `/joint_states`, computes LQR feedback, and publishes
  `JointTrajectory` commands.

---

## Control Architecture

### Inner Loop
ROS 2 `JointTrajectoryController`.

Definition:
Tracks desired joint positions using internal PID control at the joint level.

---

### Outer Loop
LQR controller (`so100_lqr`).

Definition:
Computes optimal joint position updates based on a linearized joint-space model
and quadratic cost on state and control.

---

## Interfaces

### Subscribed Topics
- `/joint_states` (`sensor_msgs/JointState`)  
  Provides joint positions and velocities.

---

### Published Topics
- `/joint_trajectory_controller/joint_trajectory`
  (`trajectory_msgs/JointTrajectory`)  
  Sends desired joint positions to the inner controller.

---

## Controlled Joints

- `Shoulder_Pitch`
- `Elbow`

Definition:
Only these two joints are actuated and controlled.
All other joints remain fixed.

---

## Parameters (LQR Node)

- `u_max`  
  Maximum allowed control magnitude.
- `dq_max`  
  Maximum allowed joint velocity.
- `rate`  
  Control loop frequency.

---

## Runtime Requirements

- ROS 2 Jazzy
- Gazebo (gz-sim)
- Ubuntu 24.04
- `ros2_control` with `JointTrajectoryController`

---

## Typical Usage

1. Build workspace with `colcon`.
2. Launch 2-DOF Gazebo simulation.
3. Run LQR outer-loop controller.

---

## Notes

- The LQR controller does not modify URDF or inner controllers.
- Stability depends on correct joint ordering and controller gains.
- The system runs entirely in joint space.
