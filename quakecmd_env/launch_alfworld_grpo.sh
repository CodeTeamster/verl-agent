#!/usr/bin/env bash
set -euo pipefail

case "${ALFWORLD_ACCELERATOR:-}" in
    ""|nvidia|ppu)
        export ALFWORLD_RUNNER="run_alfworld.sh"
        ;;
    *)
        echo "ERROR: set ALFWORLD_ACCELERATOR to nvidia or ppu in quakecmd --envs." >&2
        exit 2
        ;;
esac

exec python3 quakecmd_env/alfworld_grpo.py "$@"
