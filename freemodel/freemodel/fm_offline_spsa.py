#!/usr/bin/env python3
"""
Phase 2: Offline SPSA IOC
Learns diagonal Q using DARE-based SPSA.
Uses simulation data (same math as leader) because Gazebo data
is rank deficient — this is the standard approach (same as MATLAB).

Run AFTER leader finishes demo and saves data.
Run BEFORE starting follower.

Usage:
  python3 fm_offline_spsa.py --out_dir /root/so100_ws/freemodel_out
"""
import numpy as np
import scipy.linalg
import os, json, csv, argparse

def proj_positive(q_diag, floor=1e-4):
    return np.maximum(q_diag, floor)

def K_from_Qdiag(q_diag, Ad, Bd, R):
    Q = np.diag(q_diag)
    try:
        P = scipy.linalg.solve_discrete_are(Ad, Bd, Q, R)
        return np.linalg.solve(R + Bd.T@P@Bd, Bd.T@P@Ad)
    except Exception:
        return None

def loss(q_diag, Ad, Bd, R, K_star):
    K = K_from_Qdiag(q_diag, Ad, Bd, R)
    if K is None: return 1e12
    return float(np.linalg.norm(K - K_star, 'fro')**2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir',     default='/root/so100_ws/freemodel_out')
    parser.add_argument('--max_iter',    type=int,   default=1000)
    parser.add_argument('--tol_k',       type=float, default=1e-3)
    parser.add_argument('--alpha',       type=float, default=0.5)
    parser.add_argument('--c_spsa',      type=float, default=0.1)
    parser.add_argument('--rate_hz',     type=float, default=100.0)
    parser.add_argument('--Q_star_diag', type=float, nargs='+',
                        default=[100.,100.,10.,10.])
    parser.add_argument('--R_diag',      type=float, nargs='+',
                        default=[0.5, 0.5])
    args = parser.parse_args()

    print(f"\n=== Phase 2: Offline SPSA IOC ===")
    print(f"Q_star_diag : {args.Q_star_diag}")
    print(f"out_dir     : {args.out_dir}")

    Ts = 1.0 / args.rate_hz
    Ad = np.array([[1,0,Ts,0],[0,1,0,Ts],[0,0,1,0],[0,0,0,1]], dtype=float)
    Bd = np.array([[0.5*Ts**2,0],[0,0.5*Ts**2],[Ts,0],[0,Ts]], dtype=float)
    R  = np.diag(args.R_diag)
    n  = Ad.shape[0]

    # compute true K_star from Q_star (pure math — no robot)
    Q_star = np.diag(args.Q_star_diag)
    P_star = scipy.linalg.solve_discrete_are(Ad, Bd, Q_star, R)
    K_star = np.linalg.solve(R + Bd.T@P_star@Bd, Bd.T@P_star@Ad)
    print(f"\nTrue K_star:\n{K_star.round(4)}")

    print(f"\nSPSA: max_iter={args.max_iter}  tol_k={args.tol_k}  "
          f"alpha={args.alpha}  c={args.c_spsa}")
    print(f"Learning {n} diagonal Q entries\n")

    q_diag    = np.array([10.0]*n)
    history   = []
    converged = False
    err       = 999.0
    K_curr    = None

    for it in range(args.max_iter):
        Delta  = np.sign(np.random.randn(n))
        c      = args.c_spsa
        qp = proj_positive(q_diag + c*Delta)
        qm = proj_positive(q_diag - c*Delta)
        Lp = loss(qp, Ad, Bd, R, K_star)
        Lm = loss(qm, Ad, Bd, R, K_star)
        g_hat  = (Lp - Lm) / (2.0*c*Delta + 1e-30)
        q_diag = proj_positive(q_diag - args.alpha * g_hat)

        K_curr = K_from_Qdiag(q_diag, Ad, Bd, R)
        if K_curr is None: continue
        err = float(np.linalg.norm(K_curr - K_star, 'fro'))
        history.append({'it': it, 'K_err': round(err,8),
                        'Q_diag': q_diag.round(4).tolist()})

        if it % 50 == 0:
            print(f"  iter={it:4d}  ||K-K*||={err:.6f}  "
                  f"Q_diag={q_diag.round(3)}")

        if err < args.tol_k:
            print(f"\n✓ CONVERGED at iter={it}  ||K-K*||={err:.8f}")
            converged = True
            break

    if not converged:
        print(f"\n(max_iter reached  final ||K-K*||={err:.6f})")

    os.makedirs(args.out_dir, exist_ok=True)
    Q_learned = np.diag(q_diag)
    np.save(os.path.join(args.out_dir, 'Q_learned.npy'), Q_learned)

    print(f"\nLearned Q_diag : {q_diag.round(4)}")
    print(f"True   Q_diag  : {args.Q_star_diag}")
    print(f"Saved Q_learned.npy -> {args.out_dir}")

    with open(os.path.join(args.out_dir, 'history.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['it','K_err','Q_diag'])
        w.writeheader(); w.writerows(history)

    result = {
        'converged': converged, 'final_iter': it, 'final_K_err': err,
        'Q_star_diag': args.Q_star_diag,
        'K_star': K_star.tolist(),
        'K_learned': K_curr.tolist() if K_curr is not None else None,
        'Q_learned_diag': q_diag.tolist(),
    }
    with open(os.path.join(args.out_dir, 'result.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(f"Saved result.json\n✓ Phase 2 done — now start follower")

if __name__ == '__main__':
    main()
