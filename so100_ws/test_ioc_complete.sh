#!/bin/bash

echo "=========================================="
echo "Complete IOC Pipeline Test"
echo "=========================================="

# Source ROS
source /opt/ros/jazzy/setup.bash
source /root/so100_ws/install/setup.bash

# Start follower in background
echo ""
echo "Step 1: Starting follower (120 seconds)..."
ros2 run so100_ioc_pipeline follower_run \
  --ros-args \
  -p ns:=/so101 \
  -p duration_s:=120.0 \
  -p kfile:=/tmp/ioc_result.npz \
  -r /so101/joint_states:=/so100/joint_states &

FOLLOWER_PID=$!
echo "Follower started (PID: $FOLLOWER_PID)"

# Wait for follower to initialize
sleep 3

echo ""
echo "Step 2: Moving SO100 (leader)..."
echo "Watch both arms in Gazebo!"
echo ""

# Movement 1
echo ">>> Move 1: [0.6, -0.4]"
ros2 topic pub --once /so100/arm_position_controller/commands \
  std_msgs/msg/Float64MultiArray "{data: [0.6, -0.4]}"
sleep 5

# Check positions
echo "SO100 position:"
ros2 topic echo /so100/joint_states --once | grep -A2 "position:" | head -3
echo "SO101 position:"
ros2 topic echo /so101/joint_states --once | grep -A2 "position:" | head -3

# Movement 2
echo ""
echo ">>> Move 2: [-0.4, 0.6]"
ros2 topic pub --once /so100/arm_position_controller/commands \
  std_msgs/msg/Float64MultiArray "{data: [-0.4, 0.6]}"
sleep 5

echo "SO100 position:"
ros2 topic echo /so100/joint_states --once | grep -A2 "position:" | head -3
echo "SO101 position:"
ros2 topic echo /so101/joint_states --once | grep -A2 "position:" | head -3

# Movement 3
echo ""
echo ">>> Move 3: [0.3, 0.3]"
ros2 topic pub --once /so100/arm_position_controller/commands \
  std_msgs/msg/Float64MultiArray "{data: [0.3, 0.3]}"
sleep 5

echo "SO100 position:"
ros2 topic echo /so100/joint_states --once | grep -A2 "position:" | head -3
echo "SO101 position:"
ros2 topic echo /so101/joint_states --once | grep -A2 "position:" | head -3

# Movement 4
echo ""
echo ">>> Move 4: Return to [0.0, 0.0]"
ros2 topic pub --once /so100/arm_position_controller/commands \
  std_msgs/msg/Float64MultiArray "{data: [0.0, 0.0]}"
sleep 5

echo "SO100 position:"
ros2 topic echo /so100/joint_states --once | grep -A2 "position:" | head -3
echo "SO101 position:"
ros2 topic echo /so101/joint_states --once | grep -A2 "position:" | head -3

echo ""
echo "=========================================="
echo "Test complete! Press Ctrl+C to stop follower"
echo "=========================================="

# Wait for user to stop
wait $FOLLOWER_PID
