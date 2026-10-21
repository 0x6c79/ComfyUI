#!/usr/bin/env bash
#
# Build the ComfyUI python environment for an AMD GPU.
#
#   ./setup-env.sh [venv-dir]
#
# Packages come from AMD's multi-arch ROCm wheel index, the one the ROCm
# install docs point at. The device extra keeps the install to the GPU that
# is actually present, so only that architecture's kernels are pulled in.
#
# Environment overrides:
#   COMFYUI_ROCM_INDEX   ROCm wheel index (default: AMD multi-arch)
#   COMFYUI_ROCM_ARCH    gfx arch to target, skips detection
#   COMFYUI_UV           uv binary to use
#
# Note on triton: ROCm builds that ship triton 3.9 dropped gfx1030 (RDNA2) as
# a compilation target, which breaks every triton kernel on those cards. The
# AMD index still pairs torch with triton 3.8, and the verification step at
# the end fails loudly if a build ever pairs them with a triton that no
# longer targets the card.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="${1:-$REPO_ROOT/.venv}"
INDEX="${COMFYUI_ROCM_INDEX:-https://repo.amd.com/rocm/whl-multi-arch/}"
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

echo "=== 安装 torch / torchvision (device=$ARCH) ==="
"$UV" pip install --python "$PY" --index-url "$INDEX" \
    "torch[device-$ARCH]" "torchvision[device-$ARCH]"

printf 'torch==%s\ntorchvision==%s\n' \
    "$("$PY" -c 'import importlib.metadata as m; print(m.version("torch"))')" \
    "$("$PY" -c 'import importlib.metadata as m; print(m.version("torchvision"))')" \
    > "$VENV/constraints.txt"

echo "=== 安装 ComfyUI 依赖 ==="
grep -vE '^\s*(torch|torchvision|torchaudio|triton)\s*([=<>!~]|$)' -i \
    "$REPO_ROOT/requirements.txt" > "$VENV/requirements-nontorch.txt"
"$UV" pip install --python "$PY" --index-url https://pypi.org/simple \
    --index-strategy unsafe-best-match \
    --constraint "$VENV/constraints.txt" -r "$VENV/requirements-nontorch.txt"
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
