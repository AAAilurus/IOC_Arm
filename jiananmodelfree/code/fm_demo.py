#!/usr/bin/env python3
import os, csv
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def read_matrix_csv(path):
    rows = []
    with open(path, 'r') as f:
        r = csv.reader(f)
        header = next(r, None)
        for row in r:
            if row:
                rows.append([float(x) for x in row])
    return np.asarray(rows, dtype=float)

class FMDemo(Node):
    def __init__(self):
        super().__init__('fm_demo')

        self.declare_parameter('rate_hz',        100.0)
        self.declare_parameter('demo_time_s',    25.0)
        self.declare_parameter('zoom_time_s',    6.0)
        self.declare_parameter('q_des',          [0.8, -1.0])

        self.declare_parameter('q_init',         [0.0, 0.0])  # ADD: initial position from terminal

        self.declare_parameter('K_star_path',    '/root/so100_ws/freemodel_out/K_star.csv')
        self.declare_parameter('K_learned_path', '/root/so100_ws/freemodel_out/K_learned.csv')
        self.declare_parameter('joints',         ['Shoulder_Pitch', 'Elbow'])
        self.declare_parameter('log_dir',        '/root/so100_ws/freemodel_follow_logs')
        self.declare_parameter('plot_dir',       '/root/so100_ws/freemodel_out/plots')

        self.dt          = 1.0 / float(self.get_parameter('rate_hz').value)
        self.demo_time_s = float(self.get_parameter('demo_time_s').value)
        self.zoom_time_s = float(self.get_parameter('zoom_time_s').value)
        self.q_des       = np.array(self.get_parameter('q_des').value, dtype=float)

        self.q_init      = np.array(self.get_parameter('q_init').value, dtype=float)  # ADD

        self.joints      = list(self.get_parameter('joints').value)
        self.log_dir     = str(self.get_parameter('log_dir').value)
        self.plot_dir    = str(self.get_parameter('plot_dir').value)

        kstar_path  = str(self.get_parameter('K_star_path').value)
        klearn_path = str(self.get_parameter('K_learned_path').value)

        if not os.path.exists(kstar_path):
            raise RuntimeError(f"K_star.csv not found: {kstar_path}")
        if not os.path.exists(klearn_path):
            raise RuntimeError(f"K_learned.csv not found: {klearn_path}")

        self.K_star    = read_matrix_csv(kstar_path)
        self.K_learned = read_matrix_csv(klearn_path)

        self.get_logger().info(f"[fm_demo] q_des={self.q_des} demo_time_s={self.demo_time_s}")
        self.get_logger().info(f"[fm_demo] K_star:\n{np.round(self.K_star,4)}")
        self.get_logger().info(f"[fm_demo] K_learned:\n{np.round(self.K_learned,4)}")
        self.get_logger().info(f"[fm_demo] ||K*-K_learned||_F = {np.linalg.norm(self.K_star-self.K_learned,'fro'):.5f}")

        # states
        self.have_l = False; self.have_f = False
        self.q_l  = np.zeros(2); self.dq_l  = np.zeros(2)
        self.q_f  = np.zeros(2); self.dq_f  = np.zeros(2)

        self.t = 0.0
        self.log_t=[]; self.log_lq=[]; self.log_ldq=[]; self.log_lu=[]
        self.log_fq=[]; self.log_fdq=[]; self.log_fu=[]

        self.sub_l = self.create_subscription(JointState,'/so100/joint_states',self.cb_l,10)
        self.sub_f = self.create_subscription(JointState,'/so101/joint_states',self.cb_f,10)
        self.pub_l = self.create_publisher(Float64MultiArray,'/so100/arm_position_controller/commands',10)
        self.pub_f = self.create_publisher(Float64MultiArray,'/so101/arm_position_controller/commands',10)
        self.timer = self.create_timer(self.dt, self.step)

        # ADD: send initial position to both robots at startup
        self._send_pos(self.pub_l, self.q_init)
        self._send_pos(self.pub_f, self.q_init)
        self.get_logger().info(f"[fm_demo] ✓ q_init={self.q_init} sent to both robots")

        self.get_logger().info("[fm_demo] ✓ running SO100(K_star) vs SO101(K_learned)")

    def cb_l(self, msg):
        try: idx = [msg.name.index(j) for j in self.joints]
        except ValueError: return
        self.q_l  = np.array([msg.position[i] for i in idx], dtype=float)
        self.dq_l = np.array([msg.velocity[i] if len(msg.velocity)>i else 0.0 for i in idx], dtype=float)
        self.have_l = True

    def cb_f(self, msg):
        try: idx = [msg.name.index(j) for j in self.joints]
        except ValueError: return
        self.q_f  = np.array([msg.position[i] for i in idx], dtype=float)
        self.dq_f = np.array([msg.velocity[i] if len(msg.velocity)>i else 0.0 for i in idx], dtype=float)
        self.have_f = True

    def _send_pos(self, pub, q_cmd):  # ADD: direct position command
        msg = Float64MultiArray()
        msg.data = q_cmd.tolist()
        pub.publish(msg)

    def _send(self, pub, q, dq, u):   # FIX: correct discrete integration (use measured dq)
        q_cmd = q + dq * self.dt + 0.5 * u * (self.dt ** 2)
        msg = Float64MultiArray()
        msg.data = q_cmd.tolist()
        pub.publish(msg)

    def step(self):
        if not self.have_l or not self.have_f:
            return

        e_l = np.hstack([self.q_l - self.q_des, self.dq_l])
        u_l = -self.K_star @ e_l
        self._send(self.pub_l, self.q_l, self.dq_l, u_l)   # FIX: pass real dq

        e_f = np.hstack([self.q_f - self.q_des, self.dq_f])
        u_f = -self.K_learned @ e_f
        self._send(self.pub_f, self.q_f, self.dq_f, u_f)   # FIX: pass real dq

        self.log_t.append(self.t)
        self.log_lq.append(self.q_l.copy());  self.log_ldq.append(self.dq_l.copy()); self.log_lu.append(u_l.copy())
        self.log_fq.append(self.q_f.copy());  self.log_fdq.append(self.dq_f.copy()); self.log_fu.append(u_f.copy())
        self.t += self.dt

        if self.t >= self.demo_time_s:
            self.get_logger().info(f"[fm_demo] ✓ {self.demo_time_s}s done — saving")
            self.save_and_plot()
            raise SystemExit

    # plotting + save_and_plot unchanged ...
    def _plot_pair(self, t, yL, yF, ylabel, title1, title2, h1, h2, out_png, xlim):
        fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10,7), sharex=True)
        ax1.plot(t, yL[:,0], lw=2, label='SO100 (K_star)')
        ax1.plot(t, yF[:,0], lw=2, ls='--', label='SO101 (K_learned)')
        if h1 is not None: ax1.axhline(h1, ls=':', lw=1.0, color='k')
        ax1.set_ylabel(ylabel); ax1.set_title(title1)
        ax1.legend(); ax1.grid(True, alpha=0.4); ax1.set_xlim([0, xlim])

        ax2.plot(t, yL[:,1], lw=2, label='SO100 (K_star)')
        ax2.plot(t, yF[:,1], lw=2, ls='--', label='SO101 (K_learned)')
        if h2 is not None: ax2.axhline(h2, ls=':', lw=1.0, color='k')
        ax2.set_ylabel(ylabel); ax2.set_xlabel('Time (s)'); ax2.set_title(title2)
        ax2.legend(); ax2.grid(True, alpha=0.4); ax2.set_xlim([0, xlim])

        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()

    def save_and_plot(self):
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

        csv_path = os.path.join(self.log_dir, 'follower_trajectory.csv')
        with open(csv_path,'w',newline='') as f:
            w = csv.writer(f)
            w.writerow(['time',
                        'leader_q1','leader_q2','leader_dq1','leader_dq2','leader_u1','leader_u2',
                        'follower_q1','follower_q2','follower_dq1','follower_dq2','follower_u1','follower_u2'])
            for i in range(len(self.log_t)):
                w.writerow([self.log_t[i],
                            self.log_lq[i][0], self.log_lq[i][1],
                            self.log_ldq[i][0], self.log_ldq[i][1],
                            self.log_lu[i][0], self.log_lu[i][1],
                            self.log_fq[i][0], self.log_fq[i][1],
                            self.log_fdq[i][0], self.log_fdq[i][1],
                            self.log_fu[i][0], self.log_fu[i][1]])

        t   = np.array(self.log_t)
        lq  = np.array(self.log_lq);  fq  = np.array(self.log_fq)
        ldq = np.array(self.log_ldq); fdq = np.array(self.log_fdq)

        self._plot_pair(t, lq, fq, 'Position (rad)',
                        'Joint 1 — Position', 'Joint 2 — Position',
                        self.q_des[0], self.q_des[1],
                        os.path.join(self.plot_dir,'plot1_position_25s.png'),
                        self.demo_time_s)

        self._plot_pair(t, ldq, fdq, 'Velocity (rad/s)',
                        'Joint 1 — Velocity', 'Joint 2 — Velocity',
                        0.0, 0.0,
                        os.path.join(self.plot_dir,'plot2_velocity_25s.png'),
                        self.demo_time_s)

        zt   = self.zoom_time_s
        mask = t <= zt
        self._plot_pair(t[mask], lq[mask], fq[mask], 'Position (rad)',
                        f'Joint 1 — Position (0–{zt}s)', f'Joint 2 — Position (0–{zt}s)',
                        self.q_des[0], self.q_des[1],
                        os.path.join(self.plot_dir,'plot3_position_zoom.png'),
                        zt)

        self._plot_pair(t[mask], ldq[mask], fdq[mask], 'Velocity (rad/s)',
                        f'Joint 1 — Velocity (0–{zt}s)', f'Joint 2 — Velocity (0–{zt}s)',
                        0.0, 0.0,
                        os.path.join(self.plot_dir,'plot4_velocity_zoom.png'),
                        zt)

        self.get_logger().info(f"[fm_demo] ✓ CSV saved -> {csv_path}")
        self.get_logger().info(f"[fm_demo] ✓ Plots saved -> {self.plot_dir}")

def main():
    rclpy.init()
    node = FMDemo()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, SystemExit):
        pass
    finally:
        try: node.destroy_node()
        except: pass
        try: rclpy.shutdown()
        except: pass

if __name__ == '__main__':
    main()
