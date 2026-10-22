"""
Keep the int8 formats that have no GEMM kernel on this GPU out of the native
path.

On RDNA2 (gfx10xx) neither rocBLAS nor hipBLASLt ships int8 GEMM kernels, so
a native int8 matmul there loads no kernel at all. The failure is not a clean
error: the missing kernel gets dispatched anyway and the process dies with a
GPU memory access fault, or with a bare HIPBLAS_STATUS_INVALID_VALUE.

ComfyUI-INT8-Fast-ROCM works around that for the formats it covers, by
swapping kitchen's int8 GEMM for its own Triton kernel:

  int8_tensorwise  -> int8_linear          patched
  asym_w4a8_int8   -> w4a8_int8_linear     patched
  convrot_w4a4     -> w4a4_convrot_linear  not patched
  w6a8_int8        -> w6a8                 not patched

The two unpatched ones are forced into the emulated path here, which
dequantizes the weights and does a regular matmul. Slower than a native int8
GEMM, but it cannot fault the GPU.

If a future ComfyUI-INT8-Fast-ROCM release patches those ops too, drop them
from _EXTRA_DISABLED and the native path becomes usable again.
"""
import logging

import torch

import comfy.model_management
import comfy.ops

_log = logging.getLogger("rocm_rdn2_int8_guard")

# int8 formats whose native matmul has no kernel on RDNA2
_EXTRA_DISABLED = {"convrot_w4a4", "w6a8_int8"}

# architectures without int8 GEMM kernels
_RDNA2_PREFIXES = ("gfx101", "gfx102", "gfx103")


def _is_rdn2(device=None):
    try:
        if not comfy.model_management.is_amd():
            return False
        props = torch.cuda.get_device_properties(comfy.model_management.get_torch_device())
        return props.gcnArchName.split(":")[0].startswith(_RDNA2_PREFIXES)
    except Exception:
        return False


def _patched(device=None):
    disabled = set(_ORIGINAL(device))
    if _is_rdn2(device):
        disabled |= _EXTRA_DISABLED
    return disabled


_ORIGINAL = comfy.ops.get_disabled_quant_formats
comfy.ops.get_disabled_quant_formats = _patched

# this module registers no nodes, it only adjusts the capability detection
NODE_CLASS_MAPPINGS = {}

_log.error("[rocm-rdn2-int8-guard] int8 formats forced to the emulated path: %s",
           sorted(_EXTRA_DISABLED))
