#!/usr/bin/env python3
import os, json
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException

from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray


class FMPipeline(Node):
    """
    Case A (record-only logger for MATLAB):
      - Subscribe leader state (/so100/joint_states)
      - Subscribe leader control (/so100/lqr_u) published by fm_leader
      - Record consecutive pairs: (e_k,u_k,e_{k+1},u_{k+1})
      - Save Ek, Uk, Ek1, Uk1 as .npy and meta as json

    e = [q - q_des, dq]   (dq_des = 0)
    """

    def __init__(self):
        super().__init__('fm_pipeline')

        # -------- parameters --------
        self.declare_parameter('rate_hz', 100.0)
        self.declare_parameter('q_des', [0.5, -0.6])
        self.declare_parameter('joints', ['Shoulder_Pitch', 'Elbow'])

        self.declare_parameter('n_samples', 60000)
        self.declare_parameter('save_dir', '/root/so100_ws/freemodel_out')

        # record-only by default
        self.declare_parameter('do_learn', False)
        self.declare_parameter('do_control', False)

        # start recording when ||e|| > threshold AND u exists
        self.declare_parameter('start_threshold', 1e-2)

        self.dt = 1.0 / float(self.get_parameter('rate_hz').value)
        self.q_des = np.array(self.get_parameter('q_des').value, dtype=float)
        self.joints = list(self.get_parameter('joints').value)

        self.N = int(self.get_parameter('n_samples').value)
        self.save_dir = str(self.get_parameter('save_dir').value)

        self.do_learn = bool(self.get_parameter('do_learn').value)
        self.do_control = bool(self.get_parameter('do_control').value)
        self.start_threshold = float(self.get_parameter('start_threshold').value)

        self.get_logger().info(f"[fm_pipeline] WAIT. Will record {self.N} samples. save_dir={self.save_dir}")
        self.get_logger().info(f"[fm_pipeline] Target q_des={self.q_des}, joints={self.joints}")
        self.get_logger().info(f"[fm_pipeline] do_learn={self.do_learn}, do_control={self.do_control}")

        # -------- data buffers --------
        self.Ek  = np.zeros((self.N, 4), dtype=float)
        self.Uk  = np.zeros((self.N, 2), dtype=float)
        self.Ek1 = np.zeros((self.N, 4), dtype=float)
        self.Uk1 = np.zeros((self.N, 2), dtype=float)

        self.phase = "WAIT"
        self.count = 0

        # latest leader signals
        self.leader_q = np.zeros(2)
        self.leader_dq = np.zeros(2)
        self.leader_u = None
        self.have_state = False

        # previous (e,u) to build pairs
        self.prev_e = None
        self.prev_u = None

        # -------- ROS I/O --------
        self.sub_js = self.create_subscription(
            JointState, '/so100/joint_states', self.cb_js, 10
        )
        self.sub_u = self.create_subscription(
            Float64MultiArray, '/so100/lqr_u', self.cb_u, 10
        )

        self.timer = self.create_timer(self.dt, self.step)

    def cb_js(self, msg: JointState):
        try:
            idx = [msg.name.index(j) for j in self.joints]
        except ValueError:
            return
        self.leader_q = np.array([msg.position[i] for i in idx], dtype=float)
        self.leader_dq = np.array([msg.velocity[i] for i in idx], dtype=float)
        self.have_state = True

    def cb_u(self, msg: Float64MultiArray):
        if len(msg.data) < 2:
            return
        self.leader_u = np.array(msg.data[:2], dtype=float)

    def step(self):
        if self.count >= self.N:
            return

        if not self.have_state or self.leader_u is None:
            return

        e = np.hstack([self.leader_q - self.q_des, self.leader_dq])
        u = self.leader_u.copy()

        if self.phase == "WAIT":
            if np.linalg.norm(e) > self.start_threshold:
                self.phase = "RECORD"
                self.prev_e = e.copy()
                self.prev_u = u.copy()
                self.get_logger().info("[fm_pipeline] -> RECORD (leader is moving)")
            return

        # RECORD phase: build consecutive pairs
        if self.prev_e is None or self.prev_u is None:
            self.prev_e = e.copy()
            self.prev_u = u.copy()
            return

        k = self.count
        self.Ek[k, :]  = self.prev_e
        self.Uk[k, :]  = self.prev_u
        self.Ek1[k, :] = e
        self.Uk1[k, :] = u
        self.count += 1

        self.prev_e = e.copy()
        self.prev_u = u.copy()

        if self.count % 5000 == 0:
            self.get_logger().info(f"[fm_pipeline] recorded {self.count}/{self.N}")

        if self.count >= self.N:
            self.get_logger().info("[fm_pipeline] dataset complete -> saving")
            self.save_and_exit()

    def save_and_exit(self):
        os.makedirs(self.save_dir, exist_ok=True)
        np.save(os.path.join(self.save_dir, "Ek.npy"), self.Ek)
        np.save(os.path.join(self.save_dir, "Uk.npy"), self.Uk)
        np.save(os.path.join(self.save_dir, "Ek1.npy"), self.Ek1)
        np.save(os.path.join(self.save_dir, "Uk1.npy"), self.Uk1)

        meta = {
            "q_des": self.q_des.tolist(),
            "joints": self.joints,
            "dt": float(self.dt),
            "N_pairs": int(self.N),
            "topics": {
                "joint_states": "/so100/joint_states",
                "u": "/so100/lqr_u"
            }
        }
        with open(os.path.join(self.save_dir, "record_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        self.get_logger().info(f"[fm_pipeline] Saved Ek/Uk/Ek1/Uk1 + meta to: {self.save_dir}")
        raise SystemExit


def main():
    rclpy.init()
    node = None
    try:
        node = FMPipeline()
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, SystemExit):
        pass
    finally:
        if node is not None:
            node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
