from typing import Dict, List, Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import numpy as np

# Import python-control for LQR calculation
try:
    import control
    HAS_CONTROL = True
except ImportError:
    HAS_CONTROL = False
    print("WARNING: python-control not installed. Install with: pip install control --break-system-packages")


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def compute_lqr_gain(dt: float, Q_diag: List[float], R_diag: List[float]) -> np.ndarray:
    """
    Compute optimal LQR gain K using discrete-time Riccati equation.
    
    System model for 2-DOF arm:
    State: x = [q1, q2, dq1, dq2]  (4x1)
    Control: u = [u1, u2]          (2x1)
    
    Discrete dynamics: x[k+1] = A*x[k] + B*u[k]
    Cost: sum( x'Qx + u'Ru )
    
    Args:
        dt: Sample time (seconds)
        Q_diag: Diagonal of Q matrix [q1_cost, q2_cost, dq1_cost, dq2_cost]
        R_diag: Diagonal of R matrix [u1_cost, u2_cost]
    
    Returns:
        K: Optimal gain matrix (2x4)
    """
    if not HAS_CONTROL:
        raise RuntimeError("python-control library required. Install with: pip install control --break-system-packages")
    
    # Discrete-time state-space model
    # Position updates: q[k+1] = q[k] + dq[k]*dt
    # Velocity updates: dq[k+1] = dq[k] + u[k]*dt
    A = np.array([
        [1, 0, dt, 0],   # q1[k+1] = q1[k] + dq1[k]*dt
        [0, 1, 0, dt],   # q2[k+1] = q2[k] + dq2[k]*dt
        [0, 0, 1, 0],    # dq1[k+1] = dq1[k] + u1[k]*dt
        [0, 0, 0, 1]     # dq2[k+1] = dq2[k] + u2[k]*dt
    ])
    
    B = np.array([
        [0, 0],     # q1 not directly affected by u
        [0, 0],     # q2 not directly affected by u
        [dt, 0],    # dq1[k+1] += u1[k]*dt
        [0, dt]     # dq2[k+1] += u2[k]*dt
    ])
    
    # Cost matrices
    Q = np.diag(Q_diag)
    R = np.diag(R_diag)
    
    # Solve discrete-time algebraic Riccati equation
    K, P, eigVals = control.dlqr(A, B, Q, R)
    
    return K, P, eigVals


class LqrOuterLoop(Node):
    def __init__(self):
        super().__init__("so100_lqr_outer_loop")

        self.declare_parameter("joints", ["Shoulder_Pitch", "Elbow"])
        self.declare_parameter("state_topic", "/joint_states")
        self.declare_parameter("cmd_topic", "/joint_trajectory_controller/joint_trajectory")
        self.declare_parameter("rate_hz", 100.0)

        self.declare_parameter("q_des", [0.5, -0.6])
        self.declare_parameter("dq_des", [0.0, 0.0])

        # NEW: LQR cost matrix parameters instead of hardcoded K
        self.declare_parameter("Q_diag", [10.0, 10.0, 1.0, 1.0])  # State cost [q1, q2, dq1, dq2]
        self.declare_parameter("R_diag", [1.0, 1.0])               # Control cost [u1, u2]
        self.declare_parameter("compute_K", True)                  # Auto-compute K using LQR?
        
        # Fallback: manual K (only used if compute_K=False)
        self.declare_parameter("K", [
            20.0, 0.0, 6.0, 0.0,
            0.0, 20.0, 0.0, 6.0
        ])

        self.declare_parameter("u_max", 2.0)
        self.declare_parameter("dq_max", 1.0)
        self.declare_parameter("point_dt", 0.05)

        self.joints: List[str] = list(self.get_parameter("joints").value)
        self.state_topic = str(self.get_parameter("state_topic").value)
        self.cmd_topic = str(self.get_parameter("cmd_topic").value)
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.dt = 1.0 / self.rate_hz

        self.q_des = [float(v) for v in self.get_parameter("q_des").value]
        self.dq_des = [float(v) for v in self.get_parameter("dq_des").value]

        self.u_max = float(self.get_parameter("u_max").value)
        self.dq_max = float(self.get_parameter("dq_max").value)
        self.point_dt = float(self.get_parameter("point_dt").value)

        # Compute or load K matrix
        compute_K = bool(self.get_parameter("compute_K").value)
        
        if compute_K:
            Q_diag = [float(v) for v in self.get_parameter("Q_diag").value]
            R_diag = [float(v) for v in self.get_parameter("R_diag").value]
            
            self.get_logger().info(f"Computing LQR gain with Q_diag={Q_diag}, R_diag={R_diag}")
            K_np, P, eigVals = compute_lqr_gain(self.dt, Q_diag, R_diag)
            
            self.K = K_np.tolist()  # Convert to list[list]
            
            self.get_logger().info(f"Computed K =\n{K_np}")
            self.get_logger().info(f"Eigenvalues: {eigVals}")
            
            # Check stability
            if np.max(np.abs(eigVals)) >= 1.0:
                self.get_logger().warn(f"System may be unstable! Max |eigenvalue| = {np.max(np.abs(eigVals)):.4f}")
            else:
                self.get_logger().info(f"System is stable. Max |eigenvalue| = {np.max(np.abs(eigVals)):.4f}")
        else:
            flatK = list(self.get_parameter("K").value)
            if len(flatK) != 8:
                raise RuntimeError("K must have length 8 (2x4 flattened).")
            self.K = [flatK[0:4], flatK[4:8]]
            self.get_logger().info(f"Using manual K = {self.K}")

        self.q: Optional[List[float]] = None
        self.dq: Optional[List[float]] = None
        self.dq_cmd = [0.0, 0.0]

        self.sub = self.create_subscription(JointState, self.state_topic, self._on_joint_state, 10)
        self.pub = self.create_publisher(JointTrajectory, self.cmd_topic, 10)

        self.timer = self.create_timer(self.dt, self._step)

        self.get_logger().info(f"[LQR] joints={self.joints}")
        self.get_logger().info(f"[LQR] sub={self.state_topic}")
        self.get_logger().info(f"[LQR] pub={self.cmd_topic}")
        self.get_logger().info(f"[LQR] rate={self.rate_hz:.1f}Hz dt={self.dt:.4f}s")

    def _on_joint_state(self, msg: JointState):
        name_to_i: Dict[str, int] = {n: i for i, n in enumerate(msg.name)}
        if any(j not in name_to_i for j in self.joints):
            return

        q = []
        dq = []
        for j in self.joints:
            i = name_to_i[j]
            q.append(float(msg.position[i]))
            dq.append(float(msg.velocity[i]) if len(msg.velocity) > i else 0.0)

        self.q = q
        self.dq = dq

    def _step(self):
        if self.q is None or self.dq is None:
            return

        # Compute error state
        e = [
            self.q[0] - self.q_des[0],
            self.q[1] - self.q_des[1],
            self.dq[0] - self.dq_des[0],
            self.dq[1] - self.dq_des[1],
        ]

        # LQR control law: u = -K*e
        u = [0.0, 0.0]
        for r in range(2):
            u[r] = -(self.K[r][0]*e[0] + self.K[r][1]*e[1] + self.K[r][2]*e[2] + self.K[r][3]*e[3])
            u[r] = clamp(u[r], -self.u_max, self.u_max)

        # Integrate control to velocity command
        self.dq_cmd[0] = clamp(self.dq_cmd[0] + u[0]*self.dt, -self.dq_max, self.dq_max)
        self.dq_cmd[1] = clamp(self.dq_cmd[1] + u[1]*self.dt, -self.dq_max, self.dq_max)

        # Integrate velocity to position command
        q_cmd = [
            self.q[0] + self.dq_cmd[0]*self.dt,
            self.q[1] + self.dq_cmd[1]*self.dt,
        ]

        # Publish trajectory
        traj = JointTrajectory()
        traj.joint_names = self.joints

        pt = JointTrajectoryPoint()
        pt.positions = q_cmd
        pt.time_from_start = Duration(
            sec=int(self.point_dt),
            nanosec=int((self.point_dt % 1.0) * 1e9),
        )
        traj.points = [pt]
        self.pub.publish(traj)


def main():
    rclpy.init()
    node = LqrOuterLoop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
