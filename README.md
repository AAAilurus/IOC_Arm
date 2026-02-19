# Model-Free Inverse Optimal Control — SO-100 Dual Arm

Leader-follower IOC system for dual SO-100 robotic arms in Gazebo.
Leader demonstrates expert LQR behavior. Follower infers the hidden
cost matrix Q using SPSA, then replicates the behavior autonomously.
Follower never receives the goal — it estimates it by watching the leader.

## Environment
- ROS2 Jazzy
- Gazebo Harmonic
- Python 3, numpy, scipy

## Quick Setup
```bash
# 1. Clone
git clone https://github.com/AAAilurus/IOC_Arm.git
cd IOC_Arm

# 2. Create workspace
mkdir -p so100_ws/src
cp -r freemodel so100_2dof_bringup so100_lqr \
      so101_2dof_bringup so101_ioc so100_ioc_pipeline \
      so100_ws/src/

# 3. Build
cd so100_ws
source /opt/ros/jazzy/setup.bash
colcon build
source install/setup.bash
```

## Run — Full Pipeline

### Terminal A: Gazebo
```bash
source /opt/ros/jazzy/setup.bash && source so100_ws/install/setup.bash
ros2 launch so101_2dof_bringup gz_2dof.launch.py
```

### Terminal B: Leader (change goal as needed)
```bash
ros2 run freemodel fm_leader --ros-args \
  -p q_des:="[0.7, -1.2]" \
  -p n_samples:=60000
```
Wait for: `PHASE 2: Holding goal`

### Terminal C: Learn Q with SPSA
```bash
python3 so100_ws/src/freemodel/freemodel/fm_offline_spsa.py \
  --out_dir /tmp/freemodel_out
```
Wait for: `Phase 2 done — now start follower`

### Terminal C: Follower
```bash
ros2 run freemodel fm_follower --ros-args \
  -p q_learned_path:="/tmp/freemodel_out/Q_learned.npy"
```

## How It Works
```
Phase 1  Leader (SO100) runs expert LQR with noise
         Cycles random waypoints to generate rich data
         Records 60000 (e, u) pairs
         Switches to hold mode at final goal

Phase 2  SPSA learns diagonal Q offline (pure math, no Gazebo)
         Searches for Q such that K(Q) matches K_star
         Converges within 0.4% of true Q = [100, 100, 10, 10]

Phase 3  Follower (SO101) watches SO100
         Estimates goal from leader settled position
         Computes K from learned Q via DARE
         Moves to estimated goal and auto-stops
```

## Results
| Goal | Q learned | pos_err |
|------|-----------|---------|
| [-0.8, 0.5] | [99.61, 99.61, 10.03, 10.03] | 0.00111 rad |
| [0.7, -1.2] | [99.59, 99.60, 10.03, 10.03] | 0.00904 rad |
