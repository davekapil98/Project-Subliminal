#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'MESSAGE'
This launcher belongs to the superseded v1.2 paid-GB100/custom-Isaac plan.
Master spec v1.3 defines Stage 1 as public dataset acquisition, validation and
canonicalization. The legacy launcher is intentionally disabled. See:
  docs/legacy_v12_isaac.md
MESSAGE
exit 2
