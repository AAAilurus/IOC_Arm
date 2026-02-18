#!/usr/bin/env python3
import numpy as np
import scipy.linalg
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

class FMLeader(Node):
    """
    SO100 leader:
      - subscribes /so100/joint_states
      - computes error e = [q-q_des, dq-0]
      - computes u = -K* e + noise
      - publishes /so100/lqr_u (u)
      - publishes /so100/arm_position_controller/commands (position cmd)
    """
    def __init__(self):
        super().__init__('fm_leader')

        self.declare_parameter('rate_hz', 100.0)   # Ts=0.01
        self.declare_parameter('q_des', [0.5, -0.6])
        self.declare_parameter('noise_std', 0.05)
        self.declare_parameter('u_max', 5.0)
        self.declare_parameter('dq_max', 2.0)
        self.declare_parameter('q_min', -1.5)
        self.declare_parameter('q_max',  1.5)
        self.declare_parameter('Q_star_diag', [100.0, 100.0, 10.0, 10.0])
        self.declare_parameter('R_star_diag', [0.5, 0.5])
        self.declare_parameter('joint_names', ['Shoulder_Pitch', 'Elbow'])  # order for command + error

        self.rate_hz  = float(self.get_parameter('rate_hz').value)
        self.dt       = 1.0 / self.rate_hz
        self.q_des    = np.array(self.get_parameter('q_des').value, dtype=float)
        self.noise_std= float(self.get_parameter('noise_std').value)
        self.u_max    = float(self.get_parameter('u_max').value)
        self.dq_max   = float(self.get_parameter('dq_max').value)
        self.q_min    = float(self.get_parameter('q_min').value)
        self.q_max    = float(self.get_parameter('q_max').value)
        self.joints   = list(self.get_parameter('joint_names').value)

        Qd = np.array(self.get_parameter('Q_star_diag').value, dtype=float)
        Rd = np.array(self.get_parameter('R_star_diag').value, dtype=float)
        Q_star = np.diag(Qd)
        R_star = np.diag(Rd)

        Ts = self.dt
        self.Ad = np.array([[1,0,Ts,0],
                            [0,1,0,Ts],
                            [0,0,1, 0],
                            [0,0,0, 1]], dtype=float)
        self.Bd = np.array([[0.5*Ts**2, 0],
                            [0, 0.5*Ts**2],
                            [Ts, 0],
                            [0, Ts]], dtype=float)

        P = scipy.linalg.solve_discrete_are(self.Ad, self.Bd, Q_star, R_star)
        self.K_star = np.linalg.solve(R_star + self.Bd.T @ P @ self.Bd, self.Bd.T @ P @ self.Ad)

        self.q  = np.zeros(2)
        self.dq = np.zeros(2)
        self.dq_cmd = np.zeros(2)

        self.sub = self.create_subscription(JointState, '/so100/joint_states', self.cb_js, 10)
        self.pub_u   = self.create_publisher(Float64MultiArray, '/so100/lqr_u', 10)
        self.pub_cmd = self.create_publisher(Float64MultiArray, '/so100/arm_position_controller/commands', 10)
        self.timer = self.create_timer(self.dt, self.step)

        self.get_logger().info(f"[fm_leader] Ts={Ts:.4f}, joints={self.joints}")
        self.get_logger().info(f"[fm_leader] K_star=\n{self.K_star}")

    def cb_js(self, msg: JointState):
        n2i = {n:i for i,n in enumerate(msg.name)}
        if any(j not in n2i for j in self.joints):
            return
        for k,j in enumerate(self.joints):
            i = n2i[j]
            self.q[k]  = msg.position[i]
            self.dq[k] = msg.velocity[i] if len(msg.velocity) > i else 0.0

    def step(self):
        e = np.hstack([self.q - self.q_des, self.dq])  # dq_des=0
        eta = self.noise_std * np.random.randn(2)
        u = np.clip(-(self.K_star @ e) + eta, -self.u_max, self.u_max)

        # publish u for model-free IOC recording
        mu = Float64MultiArray()
        mu.data = u.tolist()
        self.pub_u.publish(mu)

        # integrate accel->vel->pos (position controller expects q_cmd)
        self.dq_cmd = np.clip(self.dq_cmd + u*self.dt, -self.dq_max, self.dq_max)
        q_cmd = np.clip(self.q + self.dq_cmd*self.dt, self.q_min, self.q_max)

        mc = Float64MultiArray()
        mc.data = q_cmd.tolist()
        self.pub_cmd.publish(mc)

def main():
    rclpy.init()
    node = FMLeader()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass

if __name__ == '__main__':
    main()
