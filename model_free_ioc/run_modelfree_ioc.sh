#!/bin/bash
# Complete Model-Free IOC Pipeline
# Usage: ./run_modelfree_ioc.sh [train|deploy|all]

source /opt/ros/jazzy/setup.bash
source /root/so100_ws/install/setup.bash

MODE=${1:-all}
JSON=/root/so100_ws/mf_learned_Q.json

train() {
    echo "=========================================="
    echo "OFFLINE TRAINING"
    echo "=========================================="
    python3 << 'PYEOF'
import numpy as np, scipy.linalg, json

print("=== Model-Free IOC: MATLAB-exact ===")
Ts = 0.01
Ad = np.array([[1,0,Ts,0],[0,1,0,Ts],[0,0,1,0],[0,0,0,1]])
Bd = np.array([[0.5*Ts**2,0],[0,0.5*Ts**2],[Ts,0],[0,Ts]])
Q_star = np.diag([100.0,100.0,10.0,10.0])
R_star = np.diag([0.5,0.5])
R      = R_star
P = scipy.linalg.solve_discrete_are(Ad,Bd,Q_star,R_star)
K_star = np.linalg.solve(R_star+Bd.T@P@Bd, Bd.T@P@Ad)
print(f"K_star:\n{K_star}")

# Generate data (MATLAB Section 3)
num_traj,T_each,noise_std = 300,200,0.05
n,m = 4,2
N = num_traj*T_each
Ek=np.zeros((N,n)); Uk=np.zeros((N,m))
Ek1=np.zeros((N,n)); Uk1=np.zeros((N,m))
np.random.seed(42)
idx=0
for _ in range(num_traj):
    e = np.hstack([0.4*np.random.randn(2),0.1*np.random.randn(2)])
    for t in range(T_each):
        u  = -K_star@e + noise_std*np.random.randn(m)
        e1 = Ad@e + Bd@u
        u1 = -K_star@e1 + noise_std*np.random.randn(m)
        Ek[idx]=e; Uk[idx]=u; Ek1[idx]=e1; Uk1[idx]=u1
        e=e1; idx+=1
print(f"Generated {idx} samples")

# Build Phi vech (Section 4)
d=n+m
vi=[(i,j) for i in range(d) for j in range(i,d)]
pH=len(vi)
def pv(z):
    v=np.zeros(pH)
    for k,(i,j) in enumerate(vi):
        v[k]=z[i]*z[j] if i==j else 2*z[i]*z[j]
    return v
def v2m(v):
    H=np.zeros((d,d))
    for k,(i,j) in enumerate(vi): H[i,j]=H[j,i]=v[k]
    return H
Phi=np.zeros((N,pH))
for k in range(N):
    Phi[k]=pv(np.hstack([Ek[k],Uk[k]]))-pv(np.hstack([Ek1[k],Uk1[k]]))
print(f"rank(Phi)={np.linalg.matrix_rank(Phi)}/{pH}")

def evalK(Q):
    th=np.einsum('bi,ij,bj->b',Ek,Q,Ek)+np.einsum('bi,ij,bj->b',Uk,R,Uk)
    vH,*_=np.linalg.lstsq(Phi,th,rcond=None)
    H=v2m(vH); Hux=H[n:,:n]; Huu=0.5*(H[n:,n:]+H[n:,n:].T)+1e-8*np.eye(m)
    return np.linalg.solve(Huu,Hux)

def psd(Q,f=1e-6):
    Q=0.5*(Q+Q.T); d2,V=np.linalg.eigh(Q)
    return V@np.diag(np.maximum(d2,f))@V.T

# SPSA (Section 5)
Qj=10.0*np.eye(n)
for j in range(1000):
    Kj=evalK(Qj); err=np.linalg.norm(Kj-K_star,'fro')
    if j%100==0: print(f"iter {j:4d}: ||K-K*||={err:.4e} Q={np.diag(Qj).round(2)}")
    if err<=1e-3: print(f"Converged iter {j}!"); break
    th=np.diag(Qj); D=2*(np.random.rand(n)>0.5)-1
    Kp=evalK(np.diag(th+1e-5*D)); Km=evalK(np.diag(th-1e-5*D))
    g=(np.linalg.norm(Kp-K_star,'fro')**2-np.linalg.norm(Km-K_star,'fro')**2)/(2e-5*D)
    Qj=psd(np.diag(th-0.8*g))

Kf=evalK(Qj)
print(f"\nLearned Q: {np.diag(Qj).round(3)}")
print(f"True Q:    [100,100,10,10]")
print(f"||K-K*||:  {np.linalg.norm(Kf-K_star,'fro'):.4e}")
json.dump({"Q_diag":np.diag(Qj).tolist(),"K_final":Kf.tolist(),"K_star":K_star.tolist()},
          open('/root/so100_ws/mf_learned_Q.json','w'),indent=2)
print("Saved to mf_learned_Q.json")
PYEOF
}

deploy() {
    echo "=========================================="
    echo "ONLINE DEPLOYMENT"
    echo "=========================================="
    
    if [ ! -f "$JSON" ]; then
        echo "ERROR: $JSON not found! Run training first."
        exit 1
    fi

    Q=$(python3 -c "import json; print(json.load(open('$JSON'))['Q_diag'])")
    echo "Loaded Q=$Q"
    echo ""
    echo "Starting leader (SO100)..."
    ros2 run so101_ioc mf_leader > /tmp/leader.log 2>&1 &
    LEADER=$!
    sleep 2

    echo "Starting follower (SO101)..."
    ros2 run so101_ioc mf_follower > /tmp/follower.log 2>&1 &
    FOLLOWER=$!
    sleep 5

    echo ""
    echo "=== SO100 (Leader) position ==="
    ros2 topic echo /so100/joint_states --once | grep -A 4 "position"
    echo ""
    echo "=== SO101 (Follower) position ==="
    ros2 topic echo /so101/joint_states --once | grep -A 4 "position"
    echo ""
    echo "Both running! Leader=$LEADER Follower=$FOLLOWER"
    echo "To stop: kill $LEADER $FOLLOWER"
    wait
}

case $MODE in
    train)  train ;;
    deploy) deploy ;;
    all)    train && deploy ;;
    *)      echo "Usage: $0 [train|deploy|all]" ;;
esac
