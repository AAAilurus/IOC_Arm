#!/usr/bin/env python3
import csv, numpy as np, scipy.linalg
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

class MFLeader(Node):
    def __init__(self):
        super().__init__("mf_leader")
        self.declare_parameter("q_des", [0.5, -0.6])
        self.joints = ["Shoulder_Pitch", "Elbow"]
        self.q_des = np.array(self.get_parameter("q_des").value)
        self.dq_des = np.zeros(2)
        self.noise_std = 0.05
        self.dt = 0.02
        self.u_max = 5.0
        self.dq_max = 2.0
        Ts = self.dt
        Ad = np.array([[1,0,Ts,0],[0,1,0,Ts],[0,0,1,0],[0,0,0,1]])
        Bd = np.array([[0.5*Ts**2,0],[0,0.5*Ts**2],[Ts,0],[0,Ts]])
        Q_star = np.diag([100.0,100.0,10.0,10.0])
        R_star = np.diag([0.5,0.5])
        P = scipy.linalg.solve_discrete_are(Ad, Bd, Q_star, R_star)
        self.K_star = np.linalg.solve(R_star + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
        self.get_logger().info(f"K_star:\n{self.K_star}")
        np.random.seed(42)
        self.targets = [self.q_des]  # single target for deployment
        self.target_idx = 0
        self.hold_steps = 0
        self.steps_per_target = 999999  # never switch
        self.q = np.zeros(2)
        self.dq = np.zeros(2)
        self.q_cmd = self.q_des.copy()
        self.dq_cmd = np.zeros(2)
        self.prev_e = None
        self.prev_u = None
        self.csv_path = "/root/so100_ws/mf_data.csv"
        self.f = open(self.csv_path, "w", newline="")
        self.w = csv.writer(self.f)
        self.w.writerow(["eq1","eq2","edq1","edq2","u1","u2","eq1_next","eq2_next","edq1_next","edq2_next","u1_next","u2_next"])
        self.sample_count = 0
        self.sub = self.create_subscription(JointState, "/so100/joint_states", self.cb, 10)
        self.pub = self.create_publisher(Float64MultiArray, "/so100/arm_position_controller/commands", 10)
        self.timer = self.create_timer(self.dt, self.step)

    def cb(self, msg):
        n2i = {n:i for i,n in enumerate(msg.name)}
        if any(j not in n2i for j in self.joints): return
        for k,j in enumerate(self.joints):
            i = n2i[j]
            self.q[k] = msg.position[i]
            self.dq[k] = msg.velocity[i] if len(msg.velocity)>i else 0.0

    def step(self):
        self.hold_steps += 1
        if self.hold_steps >= self.steps_per_target:
            self.hold_steps = 0
            self.target_idx = (self.target_idx+1) % len(self.targets)
            self.dq_cmd = np.zeros(2)
        tgt = self.targets[self.target_idx]
        e = np.hstack([self.q - tgt, self.dq - self.dq_des])
        eta = self.noise_std * np.random.randn(2)
        u = np.clip(-self.K_star @ e + eta, -self.u_max, self.u_max)
        if self.prev_e is not None:
            row = list(self.prev_e) + list(self.prev_u) + list(e) + list(u)
            self.w.writerow(row)
            self.sample_count += 1
            if self.sample_count % 500 == 0:
                self.f.flush()
                self.get_logger().info(f"Logged {self.sample_count} samples")
        self.prev_e = e.copy()
        self.prev_u = u.copy()
        self.dq_cmd = np.clip(self.dq_cmd + u*self.dt, -self.dq_max, self.dq_max)
        self.q_cmd = np.clip(self.q + self.dq_cmd*self.dt, -1.5, 1.5)
        msg = Float64MultiArray()
        msg.data = self.q_cmd.tolist()
        self.pub.publish(msg)

    def destroy_node(self):
        self.f.flush(); self.f.close()
        self.get_logger().info(f"Total samples: {self.sample_count}")
        super().destroy_node()

def main():
    rclpy.init()
    node = MFLeader()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__": main()
