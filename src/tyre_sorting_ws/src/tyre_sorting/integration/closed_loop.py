"""End-to-end offline closed-loop surrogate.

This is deliberately dependency-light: it exercises the same T2->T4->T5
interfaces using the generated dataset and a conveyor kinematic target model.
It is used for CI and for validating the evaluation logic before the live
PyBullet/ROS 2 launch is run on a robotics machine.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
from tyre_sorting.data.spectroscopy import TYRE_COMPOSITIONS, SpectroscopyConfig, synthesize_signature
from tyre_sorting.control.pbvs_controller import PBVSGains, pbvs_step
from tyre_sorting.decision.routing import route_fragment
from tyre_sorting.perception.cnn_decoder import train_model, SpectralInference, TYRE_LABELS


def ensure_model(dataset='dataset', out_dir='models', epochs=35):
    ckpt=os.path.join(out_dir,'spectral_cnn.pt')
    if not os.path.exists(ckpt):
        train_model(dataset, out_dir, epochs=epochs)
    return ckpt


def run_offline_trial(belt_speed: float, stream_entropy: float, n_fragments: int,
                      inference: SpectralInference, seed: int = 42,
                      max_lin_vel: float = 0.35):
    rng=np.random.default_rng(seed+int(1000*belt_speed)+int(100*stream_entropy))
    tyre_types=list(TYRE_COMPOSITIONS)
    # Entropy controls how close the stream is to the single dominant class.
    dominant=tyre_types[int(rng.integers(0,len(tyre_types)))]
    labels=[]
    for _ in range(n_fragments):
        labels.append(dominant if rng.random() > stream_entropy else tyre_types[int(rng.integers(0,len(tyre_types)))])
    spec_cfg=SpectroscopyConfig(noise_std=0.05, baseline_drift_std=0.035, seed=seed)
    y_true=[]; y_pred=[]; routes=[]; conv=[]; successes=0
    gains=PBVSGains(max_lin_vel=max_lin_vel)
    for i, tyre in enumerate(labels):
        surface = 'tread' if rng.random() < 0.5 else 'sidewall'
        sig=synthesize_signature(TYRE_COMPOSITIONS[tyre], spec_cfg, rng, surface_class=surface)
        pred, conf, surface_pred, surface_conf=inference.predict(sig)
        y_true.append(tyre); y_pred.append(pred)
        routes.append(route_fragment(pred, conf, surface_pred).route)
        # Moving target intercept model. The target starts above and ahead of EE.
        ee=np.array([0.20, 0.0, 0.28]); target=np.array([0.18, rng.uniform(-0.12,0.12), 0.04])
        belt=np.array([belt_speed, 0.0, 0.0]); dt=1/60; t_conv=None
        for step in range(900):
            lin,_,err=pbvs_step(ee, np.array([1.,0,0,0]), target, np.array([1.,0,0,0]), belt, gains)
            ee=ee+lin*dt; target=target+belt*dt
            if np.linalg.norm(err) < 0.012:
                t_conv=(step+1)*dt; break
        if t_conv is not None:
            conv.append(t_conv); successes+=1
    purity=float(np.mean(np.array(y_true)==np.array(y_pred))) if y_true else 0.0
    throughput=successes/(max(1e-9, sum(conv) if conv else n_fragments/60))*60.0
    return {
        'belt_speed':belt_speed,'stream_entropy':stream_entropy,'n_fragments':n_fragments,
        'pick_success_rate':successes/max(1,n_fragments),'classification_purity':purity,
        'mean_convergence_s':float(np.mean(conv)) if conv else float('inf'),
        'p95_convergence_s':float(np.percentile(conv,95)) if conv else float('inf'),
        'throughput_frags_per_min':float(throughput),
    }


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',default='dataset'); ap.add_argument('--models',default='models')
    ap.add_argument('--speeds',type=float,nargs='+',default=[0.05,0.15,0.25,0.35]); ap.add_argument('--entropies',type=float,nargs='+',default=[0.0,0.5,1.0])
    ap.add_argument('--n-fragments',type=int,default=40); ap.add_argument('--epochs',type=int,default=35); ap.add_argument('--out',default='results')
    args=ap.parse_args(); os.makedirs(args.out,exist_ok=True)
    ckpt=ensure_model(args.dataset,args.models,args.epochs); infer=SpectralInference(ckpt)
    results=[]
    for speed in args.speeds:
        for entropy in args.entropies:
            r=run_offline_trial(speed,entropy,args.n_fragments,infer); results.append(r); print(json.dumps(r))
    import csv
    with open(os.path.join(args.out,'offline_sweep_results.csv'),'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(results[0])); w.writeheader(); w.writerows(results)
    with open(os.path.join(args.out,'offline_summary.json'),'w') as f:
        json.dump({'n_trials':len(results),'best_purity':max(r['classification_purity'] for r in results),'best_pick_rate':max(r['pick_success_rate'] for r in results)},f,indent=2)
