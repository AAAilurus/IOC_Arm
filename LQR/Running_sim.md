````md
# IOC_Arm / LQR — Full Run Instructions (ROS 2 Jazzy + Gazebo, Docker)

## 0) Open the container

### If container is already created and running
```bash
docker ps
docker exec -it <CONTAINER_NAME> bash
````

### If container exists but is stopped

```bash
docker ps -a
docker start <CONTAINER_NAME>
docker exec -it <CONTAINER_NAME> bash
```

---

## 1) Required environment (every new terminal inside container)

```bash
source /opt/ros/jazzy/setup.bash
source /root/so100_ws/install/setup.bash
export ROS2CLI_DISABLE_DAEMON=1

export GZ_SIM_SYSTEM_PLUGIN_PATH="/opt/ros/jazzy/lib:$GZ_SIM_SYSTEM_PLUGIN_PATH"
export GZ_SIM_RESOURCE_PATH="$(ros2 pkg prefix so_100_arm)/share/so_100_arm/models:$GZ_SIM_RESOURCE_PATH"
```

---

## 2) Build packages (only if you changed code)

```bash
cd /root/so100_ws
colcon build --packages-select so100_2dof_bringup so100_lqr
source /root/so100_ws/install/setup.bash
```

---

## 3) Launch Gazebo 2DOF simulation

Terminal A:

```bash
source /opt/ros/jazzy/setup.bash
source /root/so100_ws/install/setup.bash
export ROS2CLI_DISABLE_DAEMON=1
export GZ_SIM_SYSTEM_PLUGIN_PATH="/opt/ros/jazzy/lib:$GZ_SIM_SYSTEM_PLUGIN_PATH"
export GZ_SIM_RESOURCE_PATH="$(ros2 pkg prefix so_100_arm)/share/so_100_arm/models:$GZ_SIM_RESOURCE_PATH"

ros2 launch so100_2dof_bringup gz_2dof.launch.py
```

---

## 4) Confirm controllers are active

Terminal B:

```bash
source /opt/ros/jazzy/setup.bash
source /root/so100_ws/install/setup.bash
export ROS2CLI_DISABLE_DAEMON=1

ros2 control list_controllers
```

Expected:

* `joint_state_broadcaster` = `active`
* `joint_trajectory_controller` = `active`

---

## 5) Confirm joint state data is publishing

```bash
ros2 topic info /joint_states
ros2 topic echo /joint_states --once
```

---

## 6) Send a manual joint trajectory command (position control)

```bash
ros2 topic pub --once /joint_trajectory_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory "{
  joint_names: ['Shoulder_Pitch','Elbow'],
  points: [
    { positions: [0.2, -0.6], time_from_start: {sec: 2} }
  ]
}"
```

---

## 7) Monitor controller state (optional)

```bash
ros2 topic echo /joint_trajectory_controller/controller_state --once
```

---

## 8) Run the LQR outer-loop node

Terminal C:

```bash
source /opt/ros/jazzy/setup.bash
source /root/so100_ws/install/setup.bash
export ROS2CLI_DISABLE_DAEMON=1

ros2 run so100_lqr lqr_outer_loop
```

With limits:

```bash
ros2 run so100_lqr lqr_outer_loop --ros-args -p u_max:=0.8 -p dq_max:=0.4
```

---

## 9) Verify LQR node is publishing

```bash
ros2 topic info /joint_trajectory_controller/joint_trajectory
ros2 topic hz /joint_trajectory_controller/joint_trajectory
```

---

## 10) Stop everything

### Stop LQR node

Press:

```text
Ctrl+C
```

### Stop Gazebo launch

In Terminal A press:

```text
Ctrl+C
```

---

## 11) Folder layout (GitHub)

Inside `IOC_Arm/LQR/`:

* `so_100_arm/` = original assets (models/meshes)
* `so100_2dof_bringup/` = URDF + controllers + Gazebo launch
* `so100_lqr/` = LQR outer-loop node

---

## 12) Controlled joints

* `Shoulder_Pitch`
* `Elbow`

---

## 13) If there is no motion

### Check controller is active

```bash
ros2 control list_controllers
```

### Check correct joint names were used

```bash
ros2 topic echo /joint_states --once
```

### Send a small command within limits

```bash
ros2 topic pub --once /joint_trajectory_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory "{
  joint_names: ['Shoulder_Pitch','Elbow'],
  points: [{ positions: [0.5, -0.6], time_from_start: {sec: 2} }]
}"
```

```
```

------------------------------------------------------------------------
```INST=/root/so100_ws/install/so100_2dof_bringup/share/so100_2dof_bringup/launch/gz_2dof.launch.py

python3 - <<'PY'
from pathlib import Path
import re

p = Path("/root/so100_ws/install/so100_2dof_bringup/share/so100_2dof_bringup/launch/gz_2dof.launch.py")
txt = p.read_text().splitlines()

out = []
inserted = False

for line in txt:
    # before the first model_path that uses pkg_share_models, insert definition
    if (not inserted) and ("model_path" in line) and ("pkg_share_models" in line):
        out.append("    pkg_share_models = FindPackageShare('so_100_arm').find('so_100_arm')")
        inserted = True

    out.append(line)

# also force model_path to use the correct path form
joined = "\n".join(out)
joined = re.sub(
    r"model_path\s*=\s*os\.path\.join\([^\n]*pkg_share_models[^\n]*\)",
    "model_path = os.path.join(pkg_share_models, 'models')",
    joined,
    count=1
)

p.write_text(joined)
print("patched:", p)
PY

# verify the missing variable now exists
grep -n "pkg_share_models" "$INST" | head -n 20
grep -n "model_path" "$INST" | head -n 20
---------------------------------------------------

```source /opt/ros/jazzy/setup.bash
source /root/so100_ws/install/setup.bash

export ROS2CLI_DISABLE_DAEMON=1
export GZ_SIM_SYSTEM_PLUGIN_PATH="/opt/ros/jazzy/lib:$GZ_SIM_SYSTEM_PLUGIN_PATH"
export GZ_SIM_RESOURCE_PATH="$(ros2 pkg prefix so_100_arm)/share/so_100_arm/models:$GZ_SIM_RESOURCE_PATH"

ros2 launch so100_2dof_bringup gz_2dof.launch.py




