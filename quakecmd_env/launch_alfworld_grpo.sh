#!/usr/bin/env bash
set -euo pipefail

case "${ALFWORLD_ACCELERATOR:-}" in
    ""|nvidia|ppu) ;;
    *)
        echo "ERROR: set ALFWORLD_ACCELERATOR to nvidia or ppu in quakecmd --envs." >&2
        exit 2
        ;;
esac

case "${ALGORITHM:-grpo}" in
    grpo)
        export ALFWORLD_RUNNER="run_alfworld_grpo.sh"
        ;;
    gigpo)
        export ALFWORLD_RUNNER="run_alfworld_gigpo.sh"
        ;;
    *)
        echo "ERROR: ALGORITHM must be grpo or gigpo (got ${ALGORITHM})." >&2
        exit 2
        ;;
esac

exec python3 quakecmd_env/alfworld_grpo.py "$@"
