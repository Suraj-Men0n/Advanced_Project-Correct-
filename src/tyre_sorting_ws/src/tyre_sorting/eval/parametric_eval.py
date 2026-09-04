"""T6 parametric evaluation.

Two execution modes are provided:
  --backend offline   deterministic CI-capable T2/T4/T5 evaluation surrogate
  --backend pybullet  integration hook for the live ROS2/PyBullet stack
"""
from __future__ import annotations
import argparse, csv, os
from dataclasses import dataclass, asdict

@dataclass
class TrialResult:
    belt_speed: float
    stream_entropy: float
    n_fragments: int
    pick_success_rate: float
    classification_purity: float
    mean_convergence_s: float
    p95_convergence_s: float
    throughput_frags_per_min: float


def run_single_trial(belt_speed: float, stream_entropy: float, n_fragments: int,
                     env=None, seg_fn=None, cnn_model=None, pbvs_gains=None, backend='offline') -> TrialResult:
    if backend == 'offline':
        from tyre_sorting.integration.closed_loop import ensure_model, run_offline_trial
        dataset='dataset'; models='models'
        ckpt=ensure_model(dataset,models)
        from tyre_sorting.perception.cnn_decoder import SpectralInference
        r=run_offline_trial(belt_speed,stream_entropy,n_fragments,SpectralInference(ckpt))
        return TrialResult(**r)
    raise RuntimeError('Live PyBullet evaluation must be run on the ROS2/PyBullet machine after full-system launch; use the offline backend for CI.')


def run_sweep(speeds, entropies, n_fragments, out_dir, trial_fn=run_single_trial, backend='offline'):
    os.makedirs(out_dir,exist_ok=True); results=[]
    for speed in speeds:
        for entropy in entropies:
            results.append(trial_fn(speed,entropy,n_fragments,backend=backend))
    path=os.path.join(out_dir,'sweep_results.csv')
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(asdict(results[0]))); w.writeheader(); w.writerows(asdict(r) for r in results)
    return results


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--backend',choices=['offline','pybullet'],default='offline')
    ap.add_argument('--speeds',type=float,nargs='+',default=[0.05,0.15,0.25,0.35]); ap.add_argument('--entropies',type=float,nargs='+',default=[0.0,0.5,1.0])
    ap.add_argument('--n-fragments',type=int,default=40); ap.add_argument('--out',default='results')
    args=ap.parse_args(); rows=run_sweep(args.speeds,args.entropies,args.n_fragments,args.out,backend=args.backend)
    for r in rows: print(asdict(r))

if __name__=='__main__': main()
