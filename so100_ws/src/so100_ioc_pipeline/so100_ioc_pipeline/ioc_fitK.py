#!/usr/bin/env python3
import numpy as np
import argparse
import csv
import os

# Import python-control for LQR calculation
try:
    import control
    HAS_CONTROL = True
except ImportError:
    HAS_CONTROL = False
    print("WARNING: python-control not installed.")
    print("Install with: pip3 install control --break-system-packages")


def compute_lqr_gain(dt: float, Q_diag: list, R_diag: list) -> tuple:
    """
    Compute optimal LQR gain K using discrete-time Riccati equation.
    """
    if not HAS_CONTROL:
        raise RuntimeError("python-control library required.")
    
    # Discrete-time state-space model
    A = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    
    B = np.array([
        [0, 0],
        [0, 0],
        [dt, 0],
        [0, dt]
    ])
    
    Q = np.diag(Q_diag)
    R = np.diag(R_diag)
    
    print(f"\nSystem matrices:")
    print(f"A =\n{A}")
    print(f"B =\n{B}")
    print(f"Q = diag({Q_diag})")
    print(f"R = diag({R_diag})")
    
    K, P, eigVals = control.dlqr(A, B, Q, R)
    
    return K, P, eigVals


def main():
    ap = argparse.ArgumentParser(description="Compute LQR gains using proper Riccati equation")
    ap.add_argument('--csv', required=True, help="Path to demonstration CSV file")
    ap.add_argument('--out', default='/tmp/ioc_result.npz', help="Output path for K matrix")
    ap.add_argument('--Q', type=float, nargs=4, default=[10.0, 10.0, 1.0, 1.0],
                    help="Q diagonal [q1, q2, dq1, dq2]")
    ap.add_argument('--R', type=float, nargs=2, default=[1.0, 1.0],
                    help="R diagonal [u1, u2]")
    ap.add_argument('--dt', type=float, default=None,
                    help="Sample time in seconds (auto-computed from CSV if not provided)")
    
    args = ap.parse_args()
    
    if not os.path.exists(args.csv):
        raise FileNotFoundError(args.csv)
    
    # Load demonstration data
    times, X, U = [], [], []
    with open(args.csv, 'r') as f:
        r = csv.DictReader(f)
        need = ['t', 'q1', 'q2', 'dq1', 'dq2', 'u1', 'u2']
        for k in need:
            if k not in r.fieldnames:
                raise RuntimeError(f"CSV missing '{k}', got columns: {r.fieldnames}")
        for row in r:
            times.append(float(row['t']))
            x = [float(row['q1']), float(row['q2']), float(row['dq1']), float(row['dq2'])]
            u = [float(row['u1']), float(row['u2'])]
            X.append(x)
            U.append(u)
    
    times = np.asarray(times, dtype=float)
    X = np.asarray(X, dtype=float)
    U = np.asarray(U, dtype=float)
    
    print(f"\n{'='*60}")
    print(f"Loaded N={X.shape[0]} samples from {args.csv}")
    print(f"{'='*60}")
    
    if X.shape[0] < 20:
        raise RuntimeError("Too few samples.")
    
    # Auto-compute dt from timestamps if not provided
    if args.dt is None:
        dt_samples = np.diff(times)
        dt_mean = np.mean(dt_samples)
        dt_std = np.std(dt_samples)
        dt = dt_mean
        print(f"\nAuto-computed dt from timestamps:")
        print(f"  Mean dt: {dt_mean:.6f} s ({1/dt_mean:.2f} Hz)")
        print(f"  Std dt:  {dt_std:.6f} s")
    else:
        dt = args.dt
        print(f"\nUsing specified dt: {dt:.6f} s ({1/dt:.2f} Hz)")
    
    # Compute K using LQR
    print(f"\nMethod: LQR Riccati equation")
    print(f"Q = {args.Q}")
    print(f"R = {args.R}")
    
    K, P, eigVals = compute_lqr_gain(dt, args.Q, args.R)
    
    print(f"\n{'='*60}")
    print(f"Optimal LQR Gain K (2x4):")
    print(K)
    print(f"\nClosed-loop eigenvalues:")
    print(eigVals)
    
    max_eig = np.max(np.abs(eigVals))
    if max_eig >= 1.0:
        print(f"\n⚠️  WARNING: System may be UNSTABLE!")
        print(f"   Max |eigenvalue| = {max_eig:.4f} >= 1.0")
    else:
        print(f"\n✓ System is STABLE")
        print(f"  Max |eigenvalue| = {max_eig:.4f} < 1.0")
    print(f"{'='*60}")
    
    # Save result
    np.savez(args.out, K=K, csv=args.csv, N=X.shape[0], dt=dt)
    print(f"\nSaved K matrix to: {args.out}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
