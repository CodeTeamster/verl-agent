#!/usr/bin/env bash
set -euo pipefail

accelerator=""
args=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --accelerator)
            if [ "$#" -lt 2 ]; then
                echo "ERROR: --accelerator requires nvidia or ppu." >&2
                exit 2
            fi
            accelerator="$2"
            shift 2
            ;;
        --accelerator=*)
            accelerator="${1#*=}"
            shift
            ;;
        *)
            args+=("$1")
            shift
            ;;
    esac
done

case "$accelerator" in
    ""|nvidia)
        export ALFWORLD_RUNNER="run_alfworld.sh"
        ;;
    ppu)
        export ALFWORLD_RUNNER="run_alfworld_ppu.sh"
        ;;
    *)
        echo "ERROR: set --accelerator to nvidia or ppu in quakecmd --training_start_params." >&2
        exit 2
        ;;
esac

exec python3 quakecmd_env/alfworld_grpo.py "${args[@]}"
