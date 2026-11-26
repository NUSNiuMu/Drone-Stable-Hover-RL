#!/usr/bin/env bash
# Run a python script inside the Isaac Sim 4.5 / Isaac Lab 2.1 venv with a clean environment.
#   VENV=~/workspace/isaaclab-env hover_rl/run.sh hover_rl/train.py --algo ppo --seed 0
set -e
VENV=${VENV:-$HOME/workspace/isaaclab-env}
unset PYTHONPATH AMENT_PREFIX_PATH   # avoid ROS python packages shadowing the venv
export OMNI_KIT_ACCEPT_EULA=YES
source "$VENV/bin/activate"
exec python "$@"
