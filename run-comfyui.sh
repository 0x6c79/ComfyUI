#!/usr/bin/env bash
#
# ComfyUI launcher for AMD ROCm on NixOS.
#
# Exports the MIOpen environment for the detected GPU architecture and picks
# ComfyUI flags. Extra arguments are passed through to main.py.
#
# The MIOpen profile comes from the CFZ-Caching custom node, which ships an
# architecture preset for every RDNA generation. Without that node the
# launcher still works, it just leaves MIOpen on its own defaults.
#
# Four messages show up in the log that are expected here and can be
# ignored:
#
#   [CFZ Load] No cache files found
#       CFZ-Caching prints this while its cache directory is still empty.
#       Run a workflow with CFZ Save once, or drop the CFZ Load nodes.
#
#   '_POSIX_C_SOURCE' redefined
#       The nixpkgs python3.13 headers define it as 200809L and the nix
#       glibc 2.44 headers as 202405L, and triton's driver.c pulls in both.
#       Cosmetic, printed once per kernel compile and cached after that.
#       GCC has no flag for it, -Wcpp only covers #warning directives.
#
#   MIOpen(HIP): Warning [ParseAndLoadDb] File is unreadable: gfx1030_30.HIP.fdb.txt
#       The ROCm SDK ships the gfx1030_36 database, which is the wavefront
#       this card runs, but not the wavefront 30 one MIOpen also asks for.
#       A packaging gap in the wheel, it shows up with or without the
#       MIOpen profile.
#
#   xnack 'Off' was requested for a processor that does not support it
#       XNACK is a CDNA feature and this is RDNA2. The HSA runtime always
#       says this.
#
# Environment overrides:
#   COMFYUI_ROOT                 project directory (default: this script's dir)
#   COMFYUI_PYTHON               interpreter (default: $COMFYUI_ROOT/.venv/bin/python)
#   COMFYUI_CACHE_LRU            node result cache size, 0 disables (default: 20)
#   COMFYUI_DYNAMIC_VRAM         1 enables aimdo dynamic VRAM (default: 1)
#   COMFYUI_DISABLE_PINNED_MEMORY  1 disables pinned memory (default: 0)
#   COMFYUI_MIOPEN_PROFILE       RDNA2/RDNA3/RDNA4, or "off" (default: from the GPU arch)

set -euo pipefail

REPO_ROOT="${COMFYUI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON="${COMFYUI_PYTHON:-$REPO_ROOT/.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
    echo "run-comfyui: no interpreter at $PYTHON" >&2
    echo "  set COMFYUI_PYTHON, or create the venv" >&2
    exit 1
fi

arch="$("$PYTHON" -c 'import torch; print(torch.cuda.get_device_properties(0).gcnArchName.split(":")[0])' 2>/dev/null || true)"
if [[ ! "$arch" =~ ^gfx[0-9]+ ]]; then
    echo "run-comfyui: could not detect the GPU architecture" >&2
    exit 1
fi

case "$arch" in
    gfx103*)            default_profile=RDNA2 ;;
    gfx110*|gfx115*)    default_profile=RDNA3 ;;
    gfx120*|gfx125*)    default_profile=RDNA4 ;;
    *)                  default_profile="" ;;
esac
profile="${COMFYUI_MIOPEN_PROFILE:-$default_profile}"
if [[ "$profile" == "off" ]]; then
    profile=""
fi

if [[ -n "$profile" ]]; then
    node_dir="$REPO_ROOT/custom_nodes/CFZ-Caching"
    if [[ -f "$node_dir/cfz_miopen.py" ]]; then
        profile_env="$("$PYTHON" - "$profile" "$node_dir" <<'PY'
import contextlib
import io
import sys

profile, node_dir = sys.argv[1], sys.argv[2]
sys.path.insert(0, node_dir)

# the module prints a startup banner on import, keep it out of the env dump
with contextlib.redirect_stdout(io.StringIO()):
    from cfz_miopen import _PROFILES

for key, value in _PROFILES[profile].items():
    print(f"{key}={value}")
PY
)"
        while IFS= read -r kv; do
            if [[ -n "$kv" ]]; then export "$kv"; fi
        done <<< "$profile_env"
        # FIND_MODE/FIND_ENFORCE are node inputs rather than profile entries,
        # these are the values the CFZ MIOpen Settings node defaults to.
        export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-2}"
        export MIOPEN_FIND_ENFORCE="${MIOPEN_FIND_ENFORCE:-1}"
    else
        echo "run-comfyui: CFZ-Caching not installed, skipping the $profile MIOpen profile" >&2
    fi
fi

cache_root="${XDG_CACHE_HOME:-$HOME/.cache}/comfyui"
mkdir -p "$cache_root/triton" "$cache_root/tunableop"
export TRITON_CACHE_DIR="$cache_root/triton"
export PYTORCH_TUNABLEOP_CACHE_DIR="$cache_root/tunableop"

args=()
if [[ "${COMFYUI_DYNAMIC_VRAM:-1}" == "1" ]]; then
    args+=(--enable-dynamic-vram)
fi
cache_lru="${COMFYUI_CACHE_LRU:-20}"
if [[ "$cache_lru" != "0" ]]; then
    args+=(--cache-lru "$cache_lru")
fi
if [[ "${COMFYUI_DISABLE_PINNED_MEMORY:-0}" == "1" ]]; then
    args+=(--disable-pinned-memory)
fi

cd "$REPO_ROOT"
echo "run-comfyui: $arch${profile:+ / $profile MIOpen profile} ${args[*]} $*"
exec "$PYTHON" main.py "${args[@]}" "$@"
