#!/usr/bin/env python3
"""
Model-Free IOC Follower - Complete MATLAB-matching pipeline:
1. Observe leader (e_k, u_k) data
2. Estimate K_star from data via least squares  
3. Learn Q via SPSA to match K_star
4. Control SO101 with learned Q
"""
import numpy as np
import scipy.linalg
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray

class MFFollowerFull(Node):
    def __init__(self):
        super().__init__('mf_follower_full')

        # Parameters
        self.declare_parameter('n_collect',   60000)   # samples before learning
        self.declare_parameter('rate_hz',     50.0)
        self.declare_parameter('q_des',       [0.5, -0.6])
        self.declare_parameter('R_diag',      [0.5, 0.5])
        self.declare_parameter('noise_std',   0.05)
        self.declare_parameter('alphaQ',      0.8)
        self.declare_parameter('c_spsa',      1e-5)
        self.declare_parameter('tol_K',       1e-3)
        self.declare_parameter('maxIter',     1000)
        self.declare_parameter('u_max',       5.0)
        self.declare_parameter('dq_max',      2.0)

        self.n_collect = int(self.get_parameter('n_collect').value)
        self.rate_hz   = float(self.get_parameter('rate_hz').value)
        self.dt        = 1.0 / self.rate_hz
        self.q_des     = np.array(self.get_parameter('q_des').value)
        self.dq_des    = np.zeros(2)
        self.R         = np.diag(self.get_parameter('R_diag').value)
        self.noise_std = float(self.get_parameter('noise_std').value)
        self.alphaQ    = float(self.get_parameter('alphaQ').value)
        self.c_spsa    = float(self.get_parameter('c_spsa').value)
        self.tol_K     = float(self.get_parameter('tol_K').value)
        self.maxIter   = int(self.get_parameter('maxIter').value)
        self.u_max     = float(self.get_parameter('u_max').value)
        self.dq_max    = float(self.get_parameter('dq_max').value)
        self.joints    = ['Shoulder_Pitch', 'Elbow']
        self.n, self.m = 4, 2

        # Discrete system (for simulation-based data collection)
        Ts = self.dt
        self.Ad = np.array([[1,0,Ts,0],[0,1,0,Ts],[0,0,1,0],[0,0,0,1]])
        self.Bd = np.array([[0.5*Ts**2,0],[0,0.5*Ts**2],[Ts,0],[0,Ts]])

        # Data buffers
        self.Ek  = np.zeros((self.n_collect, self.n))
        self.Uk  = np.zeros((self.n_collect, self.m))
        self.Ek1 = np.zeros((self.n_collect, self.n))
        self.Uk1 = np.zeros((self.n_collect, self.m))

        # State machine
        self.state       = 'COLLECTING'  # COLLECTING → LEARNING → CONTROLLING
        self.sample_idx  = 0
        self.learned_Q   = None
        self.K_learned   = None

        # Follower state
        self.q_follower   = np.zeros(2)
        self.dq_follower  = np.zeros(2)
        self.q_cmd        = self.q_des.copy()
        self.dq_cmd_fol   = np.zeros(2)

        # Leader observation
        self.leader_q   = np.zeros(2)
        self.leader_dq  = np.zeros(2)
        self.leader_e   = None
        self.leader_u   = None

        # ROS
        self.sub_leader   = self.create_subscription(JointState, '/so100/joint_states', self.cb_leader, 10)
        self.sub_leader_u = self.create_subscription(Float64MultiArray, '/so100/lqr_u', self.cb_u, 10)
        self.sub_follower = self.create_subscription(JointState, '/so101/joint_states', self.cb_follower, 10)
        self.pub          = self.create_publisher(Float64MultiArray, '/so101/arm_position_controller/commands', 10)
        self.timer        = self.create_timer(self.dt, self.step)

        self.get_logger().info(
            f'MF Follower Full started\n'
            f'  State: COLLECTING ({self.n_collect} samples needed)\n'
            f'  Then: LEARNING Q via SPSA\n'
            f'  Then: CONTROLLING SO101'
        )

    def cb_leader(self, msg):
        n2i = {n:i for i,n in enumerate(msg.name)}
        if any(j not in n2i for j in self.joints): return
        for k,j in enumerate(self.joints):
            i = n2i[j]
            self.leader_q[k]  = msg.position[i]
            self.leader_dq[k] = msg.velocity[i] if len(msg.velocity)>i else 0.0

    def cb_u(self, msg):
        if len(msg.data) >= 2:
            self.leader_u = np.array(msg.data[:2])

    def cb_follower(self, msg):
        n2i = {n:i for i,n in enumerate(msg.name)}
        if any(j not in n2i for j in self.joints): return
        for k,j in enumerate(self.joints):
            i = n2i[j]
            self.q_follower[k]  = msg.position[i]
            self.dq_follower[k] = msg.velocity[i] if len(msg.velocity)>i else 0.0

    def step(self):
        if self.state == 'COLLECTING':
            self._collect()
        elif self.state == 'LEARNING':
            self._learn()  # blocking, runs once
        elif self.state == 'CONTROLLING':
            self._control()

    def _collect(self):
        """
        Collect (e_k, u_k, e_{k+1}, u_{k+1}) from SIMULATION.
        Uses same dynamics as MATLAB: e_{k+1} = Ad*e_k + Bd*u_k
        Leader's K_star is estimated from observed (e,u) pairs.
        """
        # Use observed leader e as starting point for simulation
        e_obs = np.hstack([self.leader_q - self.q_des, self.leader_dq])
        
        # Simulate one step using observed u from leader
        if self.leader_u is not None and self.sample_idx < self.n_collect:
            u_k   = self.leader_u.copy()
            e_k   = e_obs
            e_k1  = self.Ad @ e_k + self.Bd @ u_k
            eta2  = self.noise_std * np.random.randn(self.m)
            # u_{k+1} estimated from e_{k+1} using current best K estimate
            # For now use same u structure
            u_k1  = u_k + eta2  # perturbed

            self.Ek[self.sample_idx]  = e_k
            self.Uk[self.sample_idx]  = u_k
            self.Ek1[self.sample_idx] = e_k1
            self.Uk1[self.sample_idx] = u_k1
            self.sample_idx += 1

            if self.sample_idx % 5000 == 0:
                self.get_logger().info(
                    f'Collecting: {self.sample_idx}/{self.n_collect} samples'
                )

            if self.sample_idx >= self.n_collect:
                self.get_logger().info('Collection done! Starting LEARNING...')
                self.state = 'LEARNING'

    def _learn(self):
        """MATLAB Sections 4+5: Build Phi, estimate K_star, run SPSA"""
        self.state = 'BUSY'  # prevent re-entry
        
        Ek  = self.Ek
        Uk  = self.Uk
        Ek1 = self.Ek1
        Uk1 = self.Uk1
        n, m = self.n, self.m
        d = n + m

        # Step 1: Estimate K_star from data (MATLAB: K_star from observed behavior)
        ridge = 1e-8
        G = Ek.T @ Ek + ridge*np.eye(n)
        K_est = -(np.linalg.solve(G, Ek.T @ Uk)).T
        self.get_logger().info(f'Estimated K_star:\n{K_est.round(4)}')

        # Step 2: Build Phi (vech, 21-dim)
        vech_idx = [(i,j) for i in range(d) for j in range(i,d)]
        p_H = len(vech_idx)

        def phi_v(z):
            v = np.zeros(p_H)
            for k,(i,j) in enumerate(vech_idx):
                v[k] = z[i]*z[j] if i==j else 2*z[i]*z[j]
            return v

        def vech2mat(v):
            H = np.zeros((d,d))
            for k,(i,j) in enumerate(vech_idx):
                H[i,j] = H[j,i] = v[k]
            return H

        Phi = np.zeros((len(Ek), p_H))
        for k in range(len(Ek)):
            zk  = np.hstack([Ek[k],  Uk[k]])
            zk1 = np.hstack([Ek1[k], Uk1[k]])
            Phi[k] = phi_v(zk) - phi_v(zk1)

        rank = np.linalg.matrix_rank(Phi)
        self.get_logger().info(f'rank(Phi) = {rank} / {p_H}')

        def evalK(Q):
            theta = np.einsum('bi,ij,bj->b', Ek, Q, Ek) + \
                    np.einsum('bi,ij,bj->b', Uk, self.R, Uk)
            vH,*_ = np.linalg.lstsq(Phi, theta, rcond=None)
            H = vech2mat(vH)
            H_ux = H[n:, :n]
            H_uu = 0.5*(H[n:,n:]+H[n:,n:].T) + 1e-8*np.eye(m)
            return np.linalg.solve(H_uu, H_ux)

        def proj_psd(Q, floor=1e-6):
            Q = 0.5*(Q+Q.T)
            d2,V = np.linalg.eigh(Q)
            return V @ np.diag(np.maximum(d2, floor)) @ V.T

        # Step 3: SPSA
        Qj = 10.0 * np.eye(n)
        self.get_logger().info('=== SPSA: learning Q ===')

        for j in range(self.maxIter):
            Kj  = evalK(Qj)
            err = np.linalg.norm(Kj - K_est, 'fro')
            if j % 100 == 0:
                self.get_logger().info(
                    f'iter {j:4d}: ||K-K*||={err:.4e}, '
                    f'Q_diag={np.diag(Qj).round(2).tolist()}'
                )
            if err <= self.tol_K:
                self.get_logger().info(f'Converged at iter {j}!')
                break
            theta = np.diag(Qj).copy()
            Delta = 2*(np.random.rand(n) > 0.5) - 1
            K_p = evalK(np.diag(theta + self.c_spsa*Delta))
            K_m = evalK(np.diag(theta - self.c_spsa*Delta))
            g   = (np.linalg.norm(K_p-K_est,'fro')**2 -
                   np.linalg.norm(K_m-K_est,'fro')**2) / (2*self.c_spsa*Delta)
            Qj  = proj_psd(np.diag(theta - self.alphaQ*g))

        self.learned_Q = Qj
        Kf = evalK(Qj)
        self.K_learned = Kf

        # Compute K via DARE with learned Q
        P = scipy.linalg.solve_discrete_are(self.Ad, self.Bd, Qj, self.R)
        self.K_learned = np.linalg.solve(
            self.R + self.Bd.T @ P @ self.Bd, self.Bd.T @ P @ self.Ad
        )

        self.get_logger().info(
            f'\n=== Learning Complete ===\n'
            f'Learned Q_diag: {np.diag(Qj).round(3).tolist()}\n'
            f'K_learned:\n{self.K_learned.round(4)}\n'
            f'Starting CONTROLLING...'
        )
        self.state = 'CONTROLLING'

    def _control(self):
        """Control SO101 using learned K"""
        e = np.hstack([self.q_follower - self.q_des, self.dq_follower - self.dq_des])
        u = np.clip(-self.K_learned @ e, -self.u_max, self.u_max)
        self.dq_cmd_fol = np.clip(self.dq_cmd_fol + u*self.dt, -self.dq_max, self.dq_max)
        self.q_cmd = np.clip(self.q_follower + self.dq_cmd_fol*self.dt, -1.5, 1.5)
        msg = Float64MultiArray()
        msg.data = self.q_cmd.tolist()
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = MFFollowerFull()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
