#!/usr/bin/env python3
import numpy as np
import scipy.linalg
import os, json
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

class FMLeader(Node):
    """
    Phase 1: Leader demonstrates expert LQR behavior.
    - Cycles through random waypoints with noise to generate rich data
    - Records (Ek, Uk, Ek1, Uk1) pairs from its own motion
    - After n_samples collected, saves data and switches to HOLD mode
    - Holds final goal position so follower can estimate it
    """
    def __init__(self):
        super().__init__('fm_leader')
        self.declare_parameter('rate_hz',        100.0)
        self.declare_parameter('q_des',          [0.5, -0.6])
        self.declare_parameter('noise_std',      0.05)
        self.declare_parameter('u_max',          5.0)
        self.declare_parameter('dq_max',         2.0)
        self.declare_parameter('q_min',         -1.5)
        self.declare_parameter('q_max',          1.5)
        self.declare_parameter('Q_star_diag',    [100.0, 100.0, 10.0, 10.0])
        self.declare_parameter('R_star_diag',    [0.5, 0.5])
        self.declare_parameter('joint_names',    ['Shoulder_Pitch', 'Elbow'])
        self.declare_parameter('cycle_interval', 3.0)
        self.declare_parameter('n_samples',      60000)
        self.declare_parameter('save_dir',       '/root/so100_ws/freemodel_out')

        self.rate_hz     = float(self.get_parameter('rate_hz').value)
        self.dt          = 1.0 / self.rate_hz
        self.q_des_final = np.array(self.get_parameter('q_des').value, dtype=float)
        self.noise_std   = float(self.get_parameter('noise_std').value)
        self.u_max       = float(self.get_parameter('u_max').value)
        self.dq_max      = float(self.get_parameter('dq_max').value)
        self.q_min       = float(self.get_parameter('q_min').value)
        self.q_max_val   = float(self.get_parameter('q_max').value)
        self.joints      = list(self.get_parameter('joint_names').value)
        self.cycle_secs  = float(self.get_parameter('cycle_interval').value)
        self.N           = int(self.get_parameter('n_samples').value)
        self.save_dir    = str(self.get_parameter('save_dir').value)

        Ts     = self.dt
        Q_star = np.diag(np.array(self.get_parameter('Q_star_diag').value, dtype=float))
        R_star = np.diag(np.array(self.get_parameter('R_star_diag').value, dtype=float))
        self.Ad = np.array([[1,0,Ts,0],[0,1,0,Ts],[0,0,1,0],[0,0,0,1]], dtype=float)
        self.Bd = np.array([[0.5*Ts**2,0],[0,0.5*Ts**2],[Ts,0],[0,Ts]], dtype=float)
        P = scipy.linalg.solve_discrete_are(self.Ad, self.Bd, Q_star, R_star)
        self.K_star = np.linalg.solve(R_star + self.Bd.T@P@self.Bd, self.Bd.T@P@self.Ad)

        # robot state
        self.q      = np.zeros(2)
        self.dq     = np.zeros(2)
        self.dq_cmd = np.zeros(2)

        # waypoint cycling
        self.q_des          = self._new_waypoint()
        self._cycle_steps   = int(self.cycle_secs / self.dt)
        self._cycle_counter = 0

        # data recording
        self.Ek    = np.zeros((self.N, 4))
        self.Uk    = np.zeros((self.N, 2))
        self.Ek1   = np.zeros((self.N, 4))
        self.Uk1   = np.zeros((self.N, 2))
        self.count = 0
        self.prev_e = None
        self.prev_u = None

        # phases: DEMO -> HOLD
        self.phase = "DEMO"

        self.sub     = self.create_subscription(
            JointState, '/so100/joint_states', self.cb_js, 10)
        self.pub_u   = self.create_publisher(
            Float64MultiArray, '/so100/lqr_u', 10)
        self.pub_cmd = self.create_publisher(
            Float64MultiArray, '/so100/arm_position_controller/commands', 10)
        self.timer = self.create_timer(self.dt, self.step)

        self.get_logger().info(
            f"[fm_leader] PHASE 1: Demonstrating to collect {self.N} samples")
        self.get_logger().info(
            f"[fm_leader] final_goal={self.q_des_final}  "
            f"noise={self.noise_std}  K_star=\n{self.K_star}")

    def _new_waypoint(self):
        if np.random.rand() < 0.25:
            return self.q_des_final.copy()
        return np.random.uniform(self.q_min, self.q_max_val, size=2)

    def cb_js(self, msg):
        n2i = {n:i for i,n in enumerate(msg.name)}
        if any(j not in n2i for j in self.joints):
            return
        for k, j in enumerate(self.joints):
            i = n2i[j]
            self.q[k]  = msg.position[i]
            self.dq[k] = msg.velocity[i] if len(msg.velocity) > i else 0.0

    def step(self):
        if self.phase == "DEMO":
            self._demo_step()
        else:
            self._hold_step()

    def _demo_step(self):
        # cycle waypoints
        self._cycle_counter += 1
        if self._cycle_counter >= self._cycle_steps:
            self._cycle_counter = 0
            self.q_des = self._new_waypoint()

        e   = np.hstack([self.q - self.q_des, self.dq])
        eta = self.noise_std * np.random.randn(2)
        u   = np.clip(-(self.K_star @ e) + eta, -self.u_max, self.u_max)

        # publish u for reference
        mu = Float64MultiArray()
        mu.data = u.tolist()
        self.pub_u.publish(mu)

        # move arm
        self.dq_cmd = np.clip(self.dq_cmd + u*self.dt, -self.dq_max, self.dq_max)
        q_cmd = np.clip(self.q + self.dq_cmd*self.dt, self.q_min, self.q_max_val)
        mc = Float64MultiArray()
        mc.data = q_cmd.tolist()
        self.pub_cmd.publish(mc)

        # record data pairs
        if self.prev_e is not None:
            k = self.count
            self.Ek[k]  = self.prev_e
            self.Uk[k]  = self.prev_u
            self.Ek1[k] = e
            self.Uk1[k] = u
            self.count += 1

            if self.count % 5000 == 0:
                self.get_logger().info(
                    f"[fm_leader] Demo progress: {self.count}/{self.N}")

            if self.count >= self.N:
                self._save_data()
                # switch to hold final goal
                self.phase  = "HOLD"
                self.q_des  = self.q_des_final.copy()
                self.dq_cmd = np.zeros(2)
                self.get_logger().info(
                    f"[fm_leader] PHASE 2: Holding goal {self.q_des_final} "
                    f"— run SPSA now, then start follower")

        self.prev_e = e.copy()
        self.prev_u = u.copy()

    def _hold_step(self):
        # hold final goal with LQR, no noise, no cycling
        e = np.hstack([self.q - self.q_des_final, self.dq])
        u = np.clip(-(self.K_star @ e), -self.u_max, self.u_max)
        self.dq_cmd = np.clip(self.dq_cmd + u*self.dt, -self.dq_max, self.dq_max)
        q_cmd = np.clip(self.q + self.dq_cmd*self.dt, self.q_min, self.q_max_val)
        mc = Float64MultiArray()
        mc.data = q_cmd.tolist()
        self.pub_cmd.publish(mc)

    def _save_data(self):
        os.makedirs(self.save_dir, exist_ok=True)
        np.save(os.path.join(self.save_dir, 'Ek.npy'),  self.Ek)
        np.save(os.path.join(self.save_dir, 'Uk.npy'),  self.Uk)
        np.save(os.path.join(self.save_dir, 'Ek1.npy'), self.Ek1)
        np.save(os.path.join(self.save_dir, 'Uk1.npy'), self.Uk1)
        meta = {'q_des': self.q_des_final.tolist(),
                'joints': self.joints, 'dt': self.dt, 'N': self.N}
        with open(os.path.join(self.save_dir, 'record_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        self.get_logger().info(
            f"[fm_leader] ✓ Data saved -> {self.save_dir}  "
            f"({self.N} pairs)")

def main():
    rclpy.init()
    node = FMLeader()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        try: rclpy.shutdown()
        except: pass

if __name__ == '__main__':
    main()
