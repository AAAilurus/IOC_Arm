# Model-Free Inverse Optimal Control (IOC)

## Overview
Model-free IOC for dual SO-100 robotic arms in Gazebo simulation.
Learns cost matrix Q from observed leader behavior using SPSA optimization.

## Algorithm (MATLAB-exact)
```
OFFLINE TRAINING:
1. Generate 60,000 samples: 300 trajectories × 200 steps
   - Start: random initial error e0 = [0.4*randn(2), 0.1*randn(2)]
   - Policy: u = -K_star*e + noise (noisy LQR)
   - Dynamics: e_{k+1} = Ad*e_k + Bd*u_k (discrete double integrator)

2. Build Phi matrix (60000×21) using vech parameterization:
   Phi[k] = vech(zk⊗zk) - vech(zk1⊗zk1)
   where zk = [e_k, u_k]

3. SPSA optimization:
   - Evaluate K(Q) via least squares on Phi
   - Minimize ||K(Q) - K_star||²
   - Update Q using simultaneous perturbation gradient

ONLINE DEPLOYMENT:
4. Load learned Q → compute K via DARE
5. Control SO101 with learned K
```

## Results
- True Q*  = [100, 100, 10, 10]
- Learned Q = [100.037, 99.997, 10.008, 10.004]
- ||K - K*|| = 0.007 (converged at iter ~950)

## Key Insight
Gazebo position controller breaks Bellman equation.
Solution: simulate double integrator dynamics in Python (like MATLAB),
learn Q from simulation data, deploy K to Gazebo.

## Files
- `mf_leader.py` - Leader node (SO100): LQR with noise, logs to CSV
- `mf_follower.py` - Follower node (SO101): loads learned Q, controls arm
- `mf_follower_full.py` - Full online pipeline (collect→learn→control)
- `mf_learned_Q.json` - Pre-trained Q matrix
- `run_modelfree_ioc.sh` - Complete pipeline script

## Usage
```bash
# Train (offline, already done):
./run_modelfree_ioc.sh train

# Deploy:
./run_modelfree_ioc.sh deploy

# Custom target:
ros2 run so101_ioc mf_leader --ros-args -p q_des:='[0.8, -0.4]' &
ros2 run so101_ioc mf_follower --ros-args -p q_des:='[0.8, -0.4]' &
```

## Comparison: Model-Based vs Model-Free
| | Model-Based | Model-Free |
|---|---|---|
| Gradients | Exact (Lyapunov) | SPSA (gradient-free) |
| Model needed | Yes (A, B) | No (data only) |
| Convergence | Fast (~100 iter) | Slower (~1000 iter) |
| Samples | Few | 60,000 |
| Q accuracy | High | High |
