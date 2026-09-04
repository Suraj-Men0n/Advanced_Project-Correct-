#!/usr/bin/env bash
# Runs the PBVS live demo and automatically saves the full terminal output
# to a timestamped log file in ~/tyre_sorting_ros2_ws/logs/, so there's
# never a need to remember the `tee` redirection by hand -- just run this
# script, wait for it to finish (or Ctrl+C if something hangs), and upload
# the log file it prints the path to at the end.
set -o pipefail

cd "$(dirname "$0")"
mkdir -p logs
LOGFILE="logs/pbvs_run_$(date +%Y%m%d_%H%M%S).log"

echo "Logging to: $LOGFILE"
echo "(Ctrl+C to stop early if it hangs -- the log up to that point is still saved and still useful.)"
echo ""

source /opt/ros/jazzy/setup.bash
source install/setup.bash

ros2 launch tyre_sorting full_system.launch.py control_mode:=pbvs 2>&1 | tee "$LOGFILE"

echo ""
echo "Done. Log saved to: $LOGFILE"
echo "Upload that file directly."
