from __future__ import annotations
import argparse,csv,os
from collections import defaultdict
import matplotlib.pyplot as plt

def read_csv(path):
    with open(path,newline='') as f:return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--csv',default='results/sweep_results.csv'); ap.add_argument('--out',default='results/plots'); a=ap.parse_args(); os.makedirs(a.out,exist_ok=True)
    rows=read_csv(a.csv)
    for metric,ylabel,name in [('classification_purity','classification purity','purity_vs_speed.png'),('pick_success_rate','pick success rate','pick_success_vs_speed.png'),('throughput_frags_per_min','throughput (fragments/min)','throughput_vs_speed.png')]:
        groups=defaultdict(list)
        for r in rows: groups[float(r['stream_entropy'])].append((float(r['belt_speed']),float(r[metric])))
        plt.figure(figsize=(7,4.5))
        for e,vals in sorted(groups.items()):
            vals.sort(); plt.plot([x for x,_ in vals],[y for _,y in vals],marker='o',label=f'entropy={e:g}')
        plt.xlabel('belt speed (m/s)'); plt.ylabel(ylabel); plt.grid(True,alpha=.25); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(a.out,name),dpi=180); plt.close()
if __name__=='__main__':main()
