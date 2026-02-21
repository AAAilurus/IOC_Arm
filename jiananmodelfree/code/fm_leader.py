#!/usr/bin/env python3
import os, csv
import numpy as np
import scipy.linalg
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

class FMLeader(Node):
    """
    TRAINING ONLY:
      - Random initial error per trajectory (like MATLAB).
      - Record Ek,Uk,Ek1,Uk1 for SPSA (model-free LS/H).
    """
    def __init__(self):
        super().__init__('fm_leader')

        self.declare_parameter('rate_hz',   100.0)         # Ts=0.01
        self.declare_parameter('q_des',     [0.5, -0.6])
        self.declare_parameter('noise_std', 0.05)

        # MATLAB-style: many short trajectories
        self.declare_parameter('num_traj',  50)           # e.g., 50
        self.declare_parameter('T_each',    300)          # e.g., 300 steps => N=15000

        # random start distribution (MATLAB: [0.4;0.1] on q,dq)
        self.declare_parameter('sigma_q',   0.4)
        self.declare_parameter('sigma_dq',  0.1)

        self.declare_parameter('joint_names', ['Shoulder_Pitch','Elbow'])
        self.declare_parameter('save_dir',    '/root/so100_ws/freemodel_out')

        self.dt        = 1.0 / float(self.get_parameter('rate_hz').value)
        self.q_des     = np.array(self.get_parameter('q_des').value, dtype=float)
        self.noise_std = float(self.get_parameter('noise_std').value)
        self.num_traj  = int(self.get_parameter('num_traj').value)
        self.T_each    = int(self.get_parameter('T_each').value)
        self.sigma_q   = float(self.get_parameter('sigma_q').value)
        self.sigma_dq  = float(self.get_parameter('sigma_dq').value)
        self.joints    = list(self.get_parameter('joint_names').value)
        self.save_dir  = str(self.get_parameter('save_dir').value)

        # expert K* (only for data generation)
        Ts = self.dt
        Q_star = np.diag([100.,100.,10.,10.])
        R_star = np.diag([0.5,0.5])
        self.Ad = np.array([[1,0,Ts,0],[0,1,0,Ts],[0,0,1,0],[0,0,0,1]], dtype=float)
        self.Bd = np.array([[0.5*Ts**2,0],[0,0.5*Ts**2],[Ts,0],[0,Ts]], dtype=float)
        P = scipy.linalg.solve_discrete_are(self.Ad, self.Bd, Q_star, R_star)
        self.K_star = np.linalg.solve(R_star + self.Bd.T@P@self.Bd, self.Bd.T@P@self.Ad)

        # ROS state
        self.have_js = False
        self.q  = np.zeros(2)
        self.dq = np.zeros(2)

        # internal simulation state for training rollouts (error-state)
        self.e   = None
        self.u   = None
        self.e2  = None
        self.u2  = None

        # dataset
        self.Ek=[]; self.Uk=[]; self.Ek1=[]; self.Uk1=[]
        self.traj_idx=0
        self.step_idx=0

        # seed like MATLAB rng(42)
        np.random.seed(42)

        # ROS I/O
        self.sub = self.create_subscription(JointState, '/so100/joint_states', self.cb_js, 10)
        self.pub = self.create_publisher(Float64MultiArray, '/so100/arm_position_controller/commands', 10)
        self.timer = self.create_timer(self.dt, self.step)

        N = self.num_traj * self.T_each
        self.get_logger().info(f"[fm_leader] TRAINING: num_traj={self.num_traj}, T_each={self.T_each}, N={N}")
        self.get_logger().info(f"[fm_leader] q_des={self.q_des}, sigma_q={self.sigma_q}, sigma_dq={self.sigma_dq}")
        self.get_logger().info(f"[fm_leader] K_star:\n{np.round(self.K_star,4)}")

    def cb_js(self, msg):
        try:
            idx = [msg.name.index(j) for j in self.joints]
        except ValueError:
            return
        self.q  = np.array([msg.position[i] for i in idx], dtype=float)
        self.dq = np.array([msg.velocity[i] if len(msg.velocity)>i else 0.0 for i in idx], dtype=float)
        self.have_js = True

    def _send(self, q_cmd):
        m = Float64MultiArray()
        m.data = q_cmd.tolist()
        self.pub.publish(m)

    def _start_new_traj(self):
        # random initial error (MATLAB)
        self.e = np.array([
            self.sigma_q*np.random.randn(),
            self.sigma_q*np.random.randn(),
            self.sigma_dq*np.random.randn(),
            self.sigma_dq*np.random.randn()
        ], dtype=float)
        self.step_idx = 0

        # place robot at x = x_eq + e (use position controller)
        q0  = self.q_des + self.e[0:2]
        self._send(q0)

        self.get_logger().info(f"[fm_leader] traj {self.traj_idx+1}/{self.num_traj} start q0={np.round(q0,4)}")

    def step(self):
        if not self.have_js:
            return

        if self.traj_idx >= self.num_traj:
            self._save_all()
            self.get_logger().info("[fm_leader] ✓ TRAINING DONE — run fm_offline_spsa next")
            raise SystemExit

        if self.e is None:
            self._start_new_traj()
            return

        # MATLAB: u = -K*e + eta
        eta  = self.noise_std * np.random.randn(2)
        u    = -self.K_star @ self.e + eta

        # error dynamics: e2 = Ad e + Bd u
        e2   = self.Ad @ self.e + self.Bd @ u

        # MATLAB: u2 = -K*e2 + eta2
        eta2 = self.noise_std * np.random.randn(2)
        u2   = -self.K_star @ e2 + eta2

        # record one sample
        self.Ek.append(self.e.copy())
        self.Uk.append(u.copy())
        self.Ek1.append(e2.copy())
        self.Uk1.append(u2.copy())

        # also drive the physical arm to follow x = x_eq + e2 (only positions)
        q_cmd = self.q_des + e2[0:2]
        self._send(q_cmd)

        self.e = e2
        self.step_idx += 1

        if self.step_idx % 50 == 0:
            self.get_logger().info(f"[fm_leader] traj {self.traj_idx+1} step {self.step_idx}/{self.T_each}")

        if self.step_idx >= self.T_each:
            self.traj_idx += 1
            self.e = None  # trigger new traj

    def _save_all(self):
        os.makedirs(self.save_dir, exist_ok=True)

        def save_csv(path, rows, header):
            with open(path,'w',newline='') as f:
                w = csv.writer(f); w.writerow(header); w.writerows(rows)

        save_csv(os.path.join(self.save_dir,'Ek.csv'),  self.Ek,  ['e1','e2','e3','e4'])
        save_csv(os.path.join(self.save_dir,'Uk.csv'),  self.Uk,  ['u1','u2'])
        save_csv(os.path.join(self.save_dir,'Ek1.csv'), self.Ek1, ['e1','e2','e3','e4'])
        save_csv(os.path.join(self.save_dir,'Uk1.csv'), self.Uk1, ['u1','u2'])

        with open(os.path.join(self.save_dir,'K_star.csv'),'w',newline='') as f:
            w = csv.writer(f)
            w.writerow(['k1','k2','k3','k4'])
            for r in self.K_star:
                w.writerow([float(x) for x in r])

        self.get_logger().info(f"[fm_leader] ✓ Saved Ek/Uk/Ek1/Uk1/K_star -> {self.save_dir}")

def main():
    rclpy.init()
    node = FMLeader()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, SystemExit):
        pass
    finally:
        try: node._save_all()
        except: pass
        try: node.destroy_node()
        except: pass
        try: rclpy.shutdown()
        except: pass

if __name__ == '__main__':
    main()
