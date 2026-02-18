#!/usr/bin/env python3
import os
import numpy as np
import scipy.linalg

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


class FMFollower(Node):
    """
    SO101 follower:
      - Loads learned Q
      - Computes LQR gain K
      - Controls SO101 to q_des
      - Logs q, e, u to npz on exit
    """

    def __init__(self):
        super().__init__('fm_follower')

        self.declare_parameter('rate_hz', 100.0)
        self.declare_parameter('q_des', [0.5, -0.6])
        self.declare_parameter('q_learned_path', '/root/so100_ws/freemodel_out/Q_learned.npy')
        self.declare_parameter('R_diag', [0.5, 0.5])
        self.declare_parameter('joints', ['Shoulder_Pitch', 'Elbow'])
        self.declare_parameter('u_max', 1.0)
        self.declare_parameter('dq_cmd_max', 0.5)
        self.declare_parameter('log_dir', '/root/so100_ws/freemodel_follow_logs')

        self.dt = 1.0 / float(self.get_parameter('rate_hz').value)
        self.q_des = np.array(self.get_parameter('q_des').value, dtype=float)
        self.q_path = str(self.get_parameter('q_learned_path').value)
        self.R = np.diag(np.array(self.get_parameter('R_diag').value, dtype=float))
        self.joints = list(self.get_parameter('joints').value)
        self.u_max = float(self.get_parameter('u_max').value)
        self.dq_cmd_max = float(self.get_parameter('dq_cmd_max').value)
        self.log_dir = str(self.get_parameter('log_dir').value)

        if not os.path.exists(self.q_path):
            raise RuntimeError(f"Q file not found: {self.q_path}")
        self.Q = np.load(self.q_path)
        self.get_logger().info(f"[fm_follower] Loaded Q from {self.q_path}:\n{self.Q}")

        Ts = self.dt
        self.Ad = np.array([[1,0,Ts,0],
                            [0,1,0,Ts],
                            [0,0,1,0],
                            [0,0,0,1]], dtype=float)
        self.Bd = np.array([[0.5*Ts**2, 0],
                            [0, 0.5*Ts**2],
                            [Ts, 0],
                            [0, Ts]], dtype=float)

        P = scipy.linalg.solve_discrete_are(self.Ad, self.Bd, self.Q, self.R)
        self.K = np.linalg.solve(self.R + self.Bd.T @ P @ self.Bd, self.Bd.T @ P @ self.Ad)
        self.get_logger().info(f"[fm_follower] K from learned Q:\n{self.K}")

        self.have_state = False
        self.q = np.zeros(2)
        self.dq = np.zeros(2)

        self.log_q = []
        self.log_e = []
        self.log_u = []

        self.sub = self.create_subscription(JointState, '/so101/joint_states', self.cb_js, 10)
        self.pub = self.create_publisher(Float64MultiArray, '/so101/arm_position_controller/commands', 10)
        self.timer = self.create_timer(self.dt, self.step)

    def cb_js(self, msg: JointState):
        try:
            idx = [msg.name.index(j) for j in self.joints]
        except ValueError:
            return
        self.q = np.array([msg.position[i] for i in idx], dtype=float)
        self.dq = np.array([msg.velocity[i] for i in idx], dtype=float)
        self.have_state = True

    def step(self):
        if not self.have_state:
            return

        e = np.hstack([self.q - self.q_des, self.dq])
        u = -self.K @ e
        u = np.clip(u, -self.u_max, self.u_max)

        dq_cmd = self.dq + u * self.dt
        dq_cmd = np.clip(dq_cmd, -self.dq_cmd_max, self.dq_cmd_max)
        q_cmd = self.q + dq_cmd * self.dt

        msg = Float64MultiArray()
        msg.data = q_cmd.tolist()
        self.pub.publish(msg)

        self.log_q.append(self.q.copy())
        self.log_e.append(e.copy())
        self.log_u.append(u.copy())

    def save_logs(self):
        os.makedirs(self.log_dir, exist_ok=True)
        path = os.path.join(self.log_dir, "follower_run.npz")
        np.savez(path, q=np.array(self.log_q), e=np.array(self.log_e), u=np.array(self.log_u),
                 q_des=self.q_des, joints=np.array(self.joints, dtype=object), dt=float(self.dt))
        print(f"[fm_follower] Saved logs to {path}")
def main():
    rclpy.init()
    node = FMFollower()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
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


if __name__ == '__main__':
    main()
