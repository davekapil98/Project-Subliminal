#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
CONFIG="${PROJECT_DIR}/configs/simulation/stage1_gb100.toml"
OUTPUT=""
RUN_ID=""
PREFLIGHT_ONLY=0
AUTO_START=0
ALLOW_NON_GB100=0
NO_BUILD=0
FORCE_BUILD=0
NEW_RUN=0
LIVESTREAM=0

usage() {
    command cat <<'USAGE'
Usage: ./launch_stage1.sh [options]

The default flow performs host checks, builds the pinned Isaac image, boots every
world with two SO-101 instances, verifies camera/IMU/labels, prints previews, and
then waits for Enter before any dataset recording begins.

Options:
  --config PATH          Stage 1 TOML (default: production GB100 plan)
  --output PATH          Host raw-data root (default from TOML)
  --run-id ID            Stable run id; existing committed episodes resume safely
  --new-run              Ignore an unfinished active-run marker
  --preflight-only       Verify everything and exit before the start prompt
  --yes                  Start recording after a passing preflight without prompting
  --no-build             Require an already-built pinned container image
  --rebuild              Rebuild the container even when the image exists
  --allow-non-gb100      Permit a deliberate smoke run on another NVIDIA GPU
  --livestream           Enable Isaac WebRTC livestream during verification/recording
  -h, --help             Show this help
USAGE
}

while (($#)); do
    case "$1" in
        --config) CONFIG="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --run-id) RUN_ID="$2"; shift 2 ;;
        --new-run) NEW_RUN=1; shift ;;
        --preflight-only) PREFLIGHT_ONLY=1; shift ;;
        --yes) AUTO_START=1; shift ;;
        --no-build) NO_BUILD=1; shift ;;
        --rebuild) FORCE_BUILD=1; shift ;;
        --allow-non-gb100) ALLOW_NON_GB100=1; shift ;;
        --livestream) LIVESTREAM=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

CONFIG="$(realpath -- "$CONFIG")"
case "$CONFIG" in
    "${PROJECT_DIR}"/*) ;;
    *) printf 'Config must be inside the copied project folder: %s\n' "$CONFIG" >&2; exit 2 ;;
esac
CONFIG_RELATIVE="${CONFIG#"${PROJECT_DIR}/"}"

if [[ -z "$OUTPUT" ]]; then
    OUTPUT_RELATIVE="$(PYTHONPATH="${PROJECT_DIR}/src" python3 "${PROJECT_DIR}/scripts/stage1_host.py" config-value --config "$CONFIG" output_root)"
    OUTPUT="${PROJECT_DIR}/${OUTPUT_RELATIVE}"
fi
mkdir -p -- "$OUTPUT"
OUTPUT="$(realpath -- "$OUTPUT")"

HOST_PREFLIGHT=(python3 "${PROJECT_DIR}/scripts/stage1_host.py" preflight --config "$CONFIG" --output "$OUTPUT")
if ((ALLOW_NON_GB100)); then HOST_PREFLIGHT+=(--allow-non-gb100); fi
printf '\n[1/4] Host and GB100 preflight\n'
PYTHONPATH="${PROJECT_DIR}/src" "${HOST_PREFLIGHT[@]}"

IMAGE="$(PYTHONPATH="${PROJECT_DIR}/src" python3 "${PROJECT_DIR}/scripts/stage1_host.py" config-value --config "$CONFIG" image)"
WORKSHOP_REPOSITORY="$(PYTHONPATH="${PROJECT_DIR}/src" python3 "${PROJECT_DIR}/scripts/stage1_host.py" config-value --config "$CONFIG" workshop_repository)"
WORKSHOP_COMMIT="$(PYTHONPATH="${PROJECT_DIR}/src" python3 "${PROJECT_DIR}/scripts/stage1_host.py" config-value --config "$CONFIG" workshop_commit)"
LEROBOT_COMMIT="$(PYTHONPATH="${PROJECT_DIR}/src" python3 "${PROJECT_DIR}/scripts/stage1_host.py" config-value --config "$CONFIG" lerobot_commit)"
CONFIG_DIGEST="$(PYTHONPATH="${PROJECT_DIR}/src" python3 "${PROJECT_DIR}/scripts/stage1_host.py" config-value --config "$CONFIG" digest)"

IMAGE_EXISTS=0
if docker image inspect "$IMAGE" >/dev/null 2>&1; then IMAGE_EXISTS=1; fi
if ((FORCE_BUILD)) || ((!IMAGE_EXISTS)); then
    if ((NO_BUILD)); then
        printf 'Required image is missing and --no-build was supplied: %s\n' "$IMAGE" >&2
        exit 2
    fi
    printf '\n[2/4] Building pinned Isaac Lab image (first build can take a while)\n'
    docker build \
        --file "${PROJECT_DIR}/docker/stage1/Dockerfile" \
        --tag "$IMAGE" \
        --build-arg "WORKSHOP_REPOSITORY=${WORKSHOP_REPOSITORY}" \
        --build-arg "WORKSHOP_COMMIT=${WORKSHOP_COMMIT}" \
        --build-arg "LEROBOT_COMMIT=${LEROBOT_COMMIT}" \
        "$PROJECT_DIR"
else
    printf '\n[2/4] Pinned Isaac Lab image is already available: %s\n' "$IMAGE"
fi

IMAGE_ID="$(docker image inspect --format '{{.Id}}' "$IMAGE")"
SOURCE_COMMIT="$(git -C "$PROJECT_DIR" rev-parse HEAD 2>/dev/null || printf unknown)"
SOURCE_DIRTY=0
if [[ -n "$(git -C "$PROJECT_DIR" status --porcelain 2>/dev/null || true)" ]]; then SOURCE_DIRTY=1; fi
PREFLIGHT_DIR="${OUTPUT}/.stage1_preflight/${CONFIG_DIGEST}"
mkdir -p -- "$PREFLIGHT_DIR"
REPORT="${PREFLIGHT_DIR}/report.json"

DOCKER_COMMON=(
    docker run --rm --gpus all --network host --ipc host
    --ulimit memlock=-1 --ulimit stack=67108864
    --env ACCEPT_EULA=Y --env PRIVACY_CONSENT=Y
    --env "SUBLIMINAL_IMAGE_REF=${IMAGE}"
    --env "SUBLIMINAL_IMAGE_ID=${IMAGE_ID}"
    --env "SUBLIMINAL_SOURCE_COMMIT=${SOURCE_COMMIT}"
    --env "SUBLIMINAL_SOURCE_DIRTY=${SOURCE_DIRTY}"
    --volume "${PROJECT_DIR}:/workspace/subliminal:ro"
    --volume "${OUTPUT}:/workspace/output:rw"
    --volume subliminal-isaac-cache:/root/.cache/ov
    --volume subliminal-pip-cache:/root/.cache/pip
)
APP_ARGS=(--headless)
if ((LIVESTREAM)); then APP_ARGS+=(--livestream 2); fi

printf '\n[3/4] Booting and verifying every Isaac world (recording remains OFF)\n'
"${DOCKER_COMMON[@]}" "$IMAGE" "$ISAAC_PYTHON" -m sim.stage1.isaac_cli \
    --mode verify \
    --config "/workspace/subliminal/${CONFIG_RELATIVE}" \
    --output /workspace/output \
    --report "/workspace/output/.stage1_preflight/${CONFIG_DIGEST}/report.json" \
    "${APP_ARGS[@]}"
PYTHONPATH="${PROJECT_DIR}/src" python3 "${PROJECT_DIR}/scripts/stage1_host.py" show-isaac-report --report "$REPORT"

if ((PREFLIGHT_ONLY)); then
    printf '\nPreflight-only requested. No dataset recording was started.\n'
    exit 0
fi

printf '\n[4/4] READY TO RECORD\n'
printf '  Config: %s\n' "$CONFIG"
printf '  Output: %s\n' "$OUTPUT"
printf '  This is the first point at which recording may start.\n\n'
if ((!AUTO_START)); then
    read -r -p "Press ENTER to start simulation data recording, or type q then ENTER to abort: " RESPONSE
    if [[ "${RESPONSE,,}" == q ]]; then
        printf 'Aborted. No recording was started.\n'
        exit 0
    fi
fi

ACTIVE_MARKER="${OUTPUT}/.stage1_active_run"
if [[ -z "$RUN_ID" ]] && ((!NEW_RUN)) && [[ -f "$ACTIVE_MARKER" ]]; then
    RUN_ID="$(<"$ACTIVE_MARKER")"
    printf 'Resuming active run: %s\n' "$RUN_ID"
fi
if [[ -z "$RUN_ID" ]]; then
    RUN_ID="stage1_$(date -u +%Y%m%dT%H%M%SZ)"
fi
printf '%s\n' "$RUN_ID" > "$ACTIVE_MARKER"
mkdir -p -- "${OUTPUT}/${RUN_ID}"

set +e
"${DOCKER_COMMON[@]}" "$IMAGE" "$ISAAC_PYTHON" -m sim.stage1.isaac_cli \
    --mode record \
    --config "/workspace/subliminal/${CONFIG_RELATIVE}" \
    --output /workspace/output \
    --run-id "$RUN_ID" \
    "${APP_ARGS[@]}" \
    2>&1 | tee -a "${OUTPUT}/${RUN_ID}/launcher.log"
RECORD_STATUS=${PIPESTATUS[0]}
set -e

if [[ -f "${OUTPUT}/${RUN_ID}/RUN_COMPLETE.json" ]]; then
    "${DOCKER_COMMON[@]}" "$IMAGE" "$ISAAC_PYTHON" \
        /workspace/subliminal/scripts/stage1_host.py validate \
        --run "/workspace/output/${RUN_ID}"
    rm -f -- "$ACTIVE_MARKER"
    printf '\nStage 1 recording completed and passed the full checksum audit: %s\n' "${OUTPUT}/${RUN_ID}"
elif ((RECORD_STATUS == 3)); then
    printf '\nRecording stopped cleanly at the configured disk quota. Resume with the same command after adding storage.\n'
else
    printf '\nRecording stopped before completion (exit %d). The active marker preserves run %s for safe resume.\n' "$RECORD_STATUS" "$RUN_ID" >&2
fi
exit "$RECORD_STATUS"
