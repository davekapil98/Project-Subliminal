#!/usr/bin/env bash
set -Eeuo pipefail

# Keep the public entry point tiny and initialize the container-only interpreter
# path before the strict implementation expands it under `set -u`.
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
export ISAAC_PYTHON=/workspace/isaaclab/_isaac_sim/python.sh
exec "${PROJECT_DIR}/.launch_stage1_impl.sh" "$@"
