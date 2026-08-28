#!/usr/bin/env bash
set -euo pipefail

export CARB_APP_PATH="${ISAAC_SIM}/kit"
export ISAAC_PATH="${ISAAC_SIM}"
export EXP_PATH="${ISAAC_SIM}/apps"
export PYTHONPATH="/workspace/subliminal/src:${PYTHONPATH:-}"

# Isaac Lab's launcher expects this environment to be sourced in containerized runs.
source "${ISAAC_SIM}/setup_python_env.sh"

exec "$@"
