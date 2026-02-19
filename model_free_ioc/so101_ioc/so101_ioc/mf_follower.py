#!/usr/bin/env python3
import json, numpy as np, scipy.linalg
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

class MFFollower(Node):
    def __init__(self):
        super().__init__("mf_follower")
        self.declare_parameter("q_des", [0.5, -0.6])
        self.declare_parameter("json_path", "/root/so100_ws/mf_learned_Q.json")
        self.declare_parameter("R_diag", [0.5, 0.5])
        self.declare_parameter("rate_hz", 50.0)
        self.declare_parameter("u_max", 5.0)
        self.declare_parameter("dq_max", 2.0)
        self.q_des = np.array(self.get_parameter("q_des").value)
        self.dq_des = np.zeros(2)
        json_path = self.get_parameter("json_path").value
        R_diag = self.get_parameter("R_diag").value
        self.u_max = float(self.get_parameter("u_max").value)
        self.dq_max = float(self.get_parameter("dq_max").value)
        self.dt = 1.0 / float(self.get_parameter("rate_hz").value)
        self.joints = ["Shoulder_Pitch", "Elbow"]
        Ts = self.dt
        Ad = np.array([[1,0,Ts,0],[0,1,0,Ts],[0,0,1,0],[0,0,0,1]])
        Bd = np.array([[0.5*Ts**2,0],[0,0.5*Ts**2],[Ts,0],[0,Ts]])
        R = np.diag(R_diag)
        with open(json_path) as f: data = json.load(f)
        Q = np.diag(data["Q_diag"])
        P = scipy.linalg.solve_discrete_are(Ad, Bd, Q, R)
        self.K = np.linalg.solve(R + Bd.T @ P @ Bd, Bd.T @ P @ Ad)
        self.get_logger().info(f"Loaded Q_diag={data[chr(81)+'_diag']}")
        self.get_logger().info(f"K:\n{self.K}")
        self.q = np.zeros(2)
        self.dq = np.zeros(2)
        self.q_cmd = self.q_des.copy()
        self.dq_cmd = np.zeros(2)
        self.sub = self.create_subscription(JointState, "/so101/joint_states", self.cb, 10)
        self.pub = self.create_publisher(Float64MultiArray, "/so101/arm_position_controller/commands", 10)
        self.timer = self.create_timer(self.dt, self.step)

    def cb(self, msg):
        n2i = {n:i for i,n in enumerate(msg.name)}
        if any(j not in n2i for j in self.joints): return
        for k,j in enumerate(self.joints):
            i = n2i[j]
            self.q[k] = msg.position[i]
            self.dq[k] = msg.velocity[i] if len(msg.velocity)>i else 0.0

    def step(self):
        e = np.hstack([self.q - self.q_des, self.dq - self.dq_des])
        u = np.clip(-self.K @ e, -self.u_max, self.u_max)
        self.dq_cmd = np.clip(self.dq_cmd + u*self.dt, -self.dq_max, self.dq_max)
        self.q_cmd = np.clip(self.q + self.dq_cmd*self.dt, -1.5, 1.5)
        msg = Float64MultiArray()
        msg.data = self.q_cmd.tolist()
        self.pub.publish(msg)

def main():
    rclpy.init()
    node = MFFollower()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__": main()
