"""O5 parametric sweep for the physical pick-and-place system.

Runs pick_and_sort_demo() headless across every (belt_speed,
spawn_interval) combination, collecting pick_success_rate,
mean_cycle_time, throughput, sorting_accuracy (the physical placement
audit) and watchdog_timeouts for each. Writes results/pickplace_sweep.csv.

This is the missing piece of O5: sweep_results.csv already covers the
offline/analytical PBVS model (belt_speed x stream_entropy); this covers
the real PyBullet pick-and-place loop (belt_speed x spawn_interval),
which is what the proposal's "expose where kinematic limits fail to keep
pace" claim is actually about.

    PYTHONPATH=src python3 run_sweep.py

Headless (gui=False), so this should take a few minutes total, not the
~90s-per-GUI-run the interactive demo takes -- each run also skips the
demo's real-time GUI-pacing sleep entirely in headless mode.
"""
import sys
sys.path.insert(0, "src")
import csv
import time
from tyre_sorting.sim.env import pick_and_sort_demo

SPEEDS = [0.05, 0.10, 0.15]
SPAWN_INTERVALS = [3.0, 5.0, 8.0]
MAX_SPAWNS = 12

# n_steps must cover THREE separate things, or a run gets cut off before
# it can produce a meaningful number:
#   1. time for all MAX_SPAWNS fragments to actually spawn
#      (spawn_interval * MAX_SPAWNS -- belt speed has NO effect on this)
#   2. worst-case full-belt transit time for the LAST fragment spawned, if
#      it's never picked and has to travel the belt's full length to the
#      reject tray (belt_length / speed)
#   3. a buffer for the arm to finish whatever cycle it's mid-way through
#      when the last fragment resolves
#
# An earlier version scaled n_steps down for higher speed (reasoning only
# about term 2) while spawn_interval scaling stayed based on the default
# config -- for 6 of the 9 (speed, spawn_interval) combinations this cut
# the run off before all 12 fragments had even finished spawning. Caught
# by computing the actual numbers before running anything, not by running
# it and discovering truncated results after the fact.
BELT_LENGTH = 2.4
BUFFER_S = 15.0


def n_steps_for(speed: float, spawn_interval: float) -> int:
    spawn_phase_s = spawn_interval * MAX_SPAWNS
    transit_s = BELT_LENGTH / speed
    return int((spawn_phase_s + transit_s + BUFFER_S) * 240)


def main():
    rows = []
    total = len(SPEEDS) * len(SPAWN_INTERVALS)
    run_i = 0
    for speed in SPEEDS:
        for spawn_interval in SPAWN_INTERVALS:
            run_i += 1
            steps = n_steps_for(speed, spawn_interval)
            print(f"[{run_i}/{total}] speed={speed} spawn_interval={spawn_interval} "
                  f"n_steps={steps} ...", flush=True)
            t_wall = time.time()
            metrics = pick_and_sort_demo(
                speed=speed, spawn_interval=spawn_interval,
                gui=False, max_spawns=MAX_SPAWNS, n_steps=steps,
                verbose=False)
            elapsed = time.time() - t_wall
            metrics["wall_clock_s"] = round(elapsed, 1)
            rows.append(metrics)
            print(f"    -> pick_success_rate={metrics['pick_success_rate']} "
                  f"throughput={metrics['throughput_per_min']:.2f}/min "
                  f"sorting_accuracy={metrics['sorting_accuracy']} "
                  f"watchdog_timeouts={metrics['watchdog_timeouts']} "
                  f"({elapsed:.1f}s wall clock)", flush=True)

    out_path = "results/pickplace_sweep.csv"
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
