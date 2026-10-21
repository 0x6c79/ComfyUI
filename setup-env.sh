#!/usr/bin/env bash
#
# Build the ComfyUI python environment for an AMD GPU.
#
#   ./setup-env.sh [venv-dir]
#
# The ROCm wheel index is chosen with COMFYUI_ROCM_INDEX. The default is the
# stable ROCm 7.14 channel, which ships a release torch build and a triton
# that still targets gfx1030.
#
# Environment overrides:
#   COMFYUI_ROCM_INDEX   pytorch wheel index (default: stable rocm7.14)
#   COMFYUI_ROCM_ARCH    gfx arch to target, skips detection
#   COMFYUI_TRITON       triton version to pin (default: 3.8.0)
#   COMFYUI_UV           uv binary to use
#
# Two things this gets right that a plain "pip install torch" does not:
#
#   - triton 3.9 dropped gfx1030 (RDNA2) as a compilation target, so every
#     triton kernel fails to build on those cards. triton is pinned instead.
#   - the torch wheel depends on the rocm meta package, whose device-all
#     extra pulls a device package for every architecture AMD supports. Only
#     the one for the detected GPU is kept, ~0.8GB instead of ~11GB.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${1:-$REPO_ROOT/.venv}"
INDEX="${COMFYUI_ROCM_INDEX:-https://download.pytorch.org/whl/rocm7.14/}"
TRITON="${COMFYUI_TRITON:-3.8.0}"
PYTHON="${COMFYUI_PYTHON:-python3.13}"

if ! command -v "$PYTHON" >/dev/null; then
    echo "setup-env: $PYTHON not found, set COMFYUI_PYTHON" >&2
    exit 1
fi

UV="${COMFYUI_UV:-}"
if [[ -z "$UV" ]]; then
    if command -v uv >/dev/null; then
        UV="uv"
    elif [[ -x "$REPO_ROOT/.venv/bin/uv" ]]; then
        UV="$REPO_ROOT/.venv/bin/uv"
    else
        echo "setup-env: uv not found, set COMFYUI_UV" >&2
        exit 1
    fi
fi

echo "=== 创建 venv: $VENV ==="
"$PYTHON" -m venv "$VENV"
PY="$VENV/bin/python"

# rocm-sdk-core is small and ships rocminfo, which is how the architecture
# gets detected without a torch install.
echo "=== 安装 rocm-sdk-core (用于探测架构) ==="
"$UV" pip install --python "$PY" --index-url "$INDEX" rocm-sdk-core

ARCH="${COMFYUI_ROCM_ARCH:-}"
if [[ -z "$ARCH" ]]; then
    ARCH="$("$VENV/bin/rocminfo" 2>/dev/null | grep -oE 'gfx[0-9a-z]+' | head -1 || true)"
fi
if [[ ! "$ARCH" =~ ^gfx[0-9]+ ]]; then
    echo "setup-env: could not detect the GPU architecture, set COMFYUI_ROCM_ARCH" >&2
    exit 1
fi
echo "=== GPU 架构: $ARCH ==="

echo "=== 安装 torch / ROCm / triton ==="
# torch and torchvision are installed without dependencies. Their metadata
# asks for rocm[device-all], which would pull a device package for every
# architecture AMD supports.
"$UV" pip install --python "$PY" --index-url "$INDEX" torch torchvision --no-deps
"$UV" pip install --python "$PY" --index-url "$INDEX" \
    "rocm-sdk-libraries" "rocm-sdk-device-$ARCH" "triton-rocm==$TRITON" \
    filelock fsspec jinja2 markupsafe mpmath networkx sympy typing-extensions setuptools

echo "=== 安装 ComfyUI 依赖 ==="
grep -vE '^\s*(torch|torchvision|torchaudio|triton)\s*([=<>!~]|$)' -i \
    "$REPO_ROOT/requirements.txt" > "$VENV/requirements-nontorch.txt"

# Resolve the whole graph first, then drop the device packages for the
# architectures this GPU will never run on.
"$UV" pip compile --python "$PY" \
    --index-url "$INDEX" --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match \
    --output-file "$VENV/requirements-resolved.txt" \
    "$VENV/requirements-nontorch.txt" >/dev/null
grep -v '^rocm-sdk-device-gfx' "$VENV/requirements-resolved.txt" > "$VENV/requirements-slim.txt"
echo "rocm-sdk-device-$ARCH==$(grep -oE '[0-9][0-9.]*$' <<< "$(sed -n "s/^rocm-sdk-device-$ARCH==//p" "$VENV/requirements-resolved.txt")" | head -1)" >> "$VENV/requirements-slim.txt"

"$UV" pip install --python "$PY" --no-deps \
    --index-url "$INDEX" --extra-index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match -r "$VENV/requirements-slim.txt"
"$UV" pip install --python "$PY" --index-url https://pypi.org/simple \
    -r "$REPO_ROOT/manager_requirements.txt"

echo "=== 验证 ==="
"$PY" - <<'PY'
import torch
import triton
import triton.language as tl

props = torch.cuda.get_device_properties(0)
print(f"  torch      {torch.__version__}")
print(f"  hip        {torch.version.hip}")
print(f"  triton     {triton.__version__}")
print(f"  gpu        {props.name} ({props.gcnArchName})")


@triton.jit
def _probe(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=mask) * 2.0, mask=mask)


n = 4096
x = torch.randn(n, device="cuda")
y = torch.empty_like(x)
_probe[(n // 256,)](x, y, n, BLOCK=256)
torch.cuda.synchronize()
assert torch.allclose(y, x * 2), "triton kernel produced wrong results"
print("  triton kernel on this GPU: OK")
PY

echo "=== 完成: $VENV ==="
echo "  启动: COMFYUI_ROOT=$REPO_ROOT $REPO_ROOT/run-comfyui.sh"
