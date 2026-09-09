# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import logging
import os

import torch
import triton
import triton.language as tl
from _kunlunxin.utils.codegen_config_utils import CodeGenConfig
from flag_gems.runtime import torch_device_fn

from ..utils.pointwise_dynamic import pointwise_dynamic

# ---------------------------------------------------------------------------
# tle.raw fast path for the tensor-vs-scalar less compare (P800 xpu3,
# cluster C payload in lt_raw.xpu). The compiler cannot vectorize a
# tensor-vs-scalar float compare (CoreTiling hands the scalar kernel 8
# elements/core/iteration, below the f32 vector width of 16, so `arith.cmpf`
# stays scalarized -- ~370us on 16M elements; TRITONXPU_VEC_REPORT shows the
# closure vetoed with keyState=Conflict). The hand-written payload streams
# the input once per core with pipelined GM2LM/LM2GM DMA and compares with
# the hardware vector-lt intrinsics (x < s with the vector on the LEFT:
# vvlt(x, splat(s)), the vendor xdnn SWAP=false recipe), with the same memory
# footprint as the ATen reference. It also matches ATen's scalar-dtype
# promotion exactly. Mirrors the not_equal/greater scalar payloads
# (ne_raw.xpu / gt_raw.xpu); see lt_raw.xpu for the full analysis.
try:
    import triton.experimental.tle as tle

    _TLE_OK = True
except ImportError:
    tle = None
    _TLE_OK = False

_HERE = os.path.dirname(os.path.abspath(__file__))
_NCLUSTER = 12  # P800 (xpu3): one Triton program == one cluster of 64 cores
# Payload scalars are i32 (do_not_specialize); guard the byte range.
_RAW_MAX_ELEMS = 2**31 - 1
# Must match CHUNK_BYTES in lt_raw.xpu (the chunk-grid partition contract).
_RAW_CHUNK_BYTES = 2048
# Below this element count the op is launch-bound and the bare pointwise
# scalar kernel (single launch, no extra host work) is fine; the payload wins
# from ~64K elements up (same crossover as the not_equal/greater payloads).
_RAW_SMALL_SCALAR_LIMIT = 65536

_RAW_TYPE_CODE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
}

if _TLE_OK:

    @tle.raw.dialect("xpu3", file=os.path.join(_HERE, "lt_raw.xpu"))
    def lt_scalar_raw(in_, out, numel, esz, type_code, scalar_bits,
                      chunk_start, chunk_count):
        ...

    @triton.jit(do_not_specialize=["numel", "esz", "type_code", "scalar_bits",
                                   "chunk_count"])
    def lt_scalar_raw_kernel(In, Out, numel, esz, type_code, scalar_bits,
                             chunk_count):
        pid = tl.program_id(0)
        tle.raw.call(lt_scalar_raw, (In, Out, numel, esz, type_code,
                                     scalar_bits, pid * chunk_count,
                                     chunk_count))


def _view_u8(t):
    """Byte view of a tensor; works for 0-dim tensors too."""
    if t.dim() == 0:
        return t.view(1).view(torch.uint8)
    return t.view(torch.uint8)


@functools.lru_cache(maxsize=1024)
def _scalar_bits(B, dtype):
    """The scalar promoted to `dtype`, as a sign-extended int32 bit pattern.

    Matches torch.less's type promotion: the python float scalar is converted
    to the tensor's dtype and compared in that dtype. Cached: the dtype
    conversion costs a couple of microseconds on the host, which is directly
    visible on launch-bound small shapes.
    """
    if dtype == torch.float32:
        return int(torch.tensor(B, dtype=torch.float32).view(torch.int32).item())
    # fp16 / bf16: the payload only reads the low 16 bits.
    return int(torch.tensor(B, dtype=dtype).view(torch.int16).item())


def _raw_lt_scalar(A, B):
    """less(A, scalar) via the raw payload, or None when it does not apply.

    Only the contiguous case in the supported float dtypes is handled;
    anything else falls back to the pointwise scalar kernel below.
    """
    if not _TLE_OK or not A.is_contiguous():
        return None
    type_code = _RAW_TYPE_CODE.get(A.dtype)
    if type_code is None:
        return None
    M = A.numel()
    if M == 0 or M > _RAW_MAX_ELEMS:
        return None
    esz = A.element_size()
    s_bits = _scalar_bits(B, A.dtype)
    out = torch.empty(A.shape, dtype=torch.bool, device=A.device)
    # partition by payload chunks (CHUNK_BYTES/esz elements each): every
    # program and core gets whole chunks so all GM2LM/LM2GM transfers are
    # CHUNK_BYTES-aligned in global memory.
    chunk_elems = _RAW_CHUNK_BYTES // esz
    total_chunks = (M + chunk_elems - 1) // chunk_elems
    per = (total_chunks + _NCLUSTER - 1) // _NCLUSTER
    with torch_device_fn.device(A.device):
        lt_scalar_raw_kernel[(_NCLUSTER,)](
            _view_u8(A), _view_u8(out), M, esz, type_code, s_bits, per)
    return out

logger = logging.getLogger(__name__)

config_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    isCloseMemoryAsync=False,
    unroll_num=8,
    kunlunAutoGrid=True,
)


@pointwise_dynamic(
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def lt_func(x, y):
    return x.to(tl.float32) < y


def lt(A, B):
    logger.debug("GEMS_KUNLUNXIN LT")
    os.environ["TRITONXPU_COMPARE_FUSION"] = "1"
    os.environ["TRITONXPU_FP16_FAST"] = "1"
    res = lt_func(A, B)
    del os.environ["TRITONXPU_COMPARE_FUSION"]
    del os.environ["TRITONXPU_FP16_FAST"]
    return res


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_,
)
@triton.jit
def lt_func_scalar(x, y):
    return x.to(tl.float32) < y


def lt_scalar(A, B):
    logger.debug("GEMS_KUNLUNXIN LT_SCALAR")
    # Fast path: hand-written cluster-C payload (lt_raw.xpu). The compiler
    # scalarizes the tensor-vs-scalar compare (~370us on 16M elements, see the
    # header comment); the payload streams A once with per-core pipelined DMA
    # and the hardware vector-lt intrinsics at the same memory footprint as
    # ATen, and matches ATen's scalar-dtype promotion exactly.
    if A.numel() >= _RAW_SMALL_SCALAR_LIMIT:
        raw_out = _raw_lt_scalar(A, B)
        if raw_out is not None:
            return raw_out
    res = lt_func_scalar(A, B)
    return res


# lt_ / lt_scalar_ are the in-place aliases of lt (out = (x < y) written back
# into x). They were NOT overridden by kunlunxin, so they fell to the generic
# ops/lt_.py -- a bare `@pointwise_dynamic` with NO CodeGenConfig -> discrete /
# launch-bound slow path. Baseline IR (ir-lt_-dev0 / ir-lt_scalar_-dev1) shows
# 512-wide discrete masked stores (`tt.ptr<f16, 0>` + per-element i32 column
# offsets) -> catastrophic latency on large shapes.
#
# Fix: a dedicated tuned pointwise_dynamic that writes in place (out0=A).
# CRITICAL: this in-place variant must NOT reuse lt's config_ -- config_ has
# `isCloseMemoryAsync=False` (async memory copy ON), and with in-place aliasing
# (input tensor == output tensor) the async double-buffered copy path deadlocks
# the device ("noc idle timeout" hang). The out-of-place lt is fine because its
# output is a fresh bool tensor (no aliasing). So use a config with the DEFAULT
# isCloseMemoryAsync (True = async closed), mirroring the proven in-place op
# greater_equal_. Body returns tl.where(...,1,0) (int 0/1) which stores cleanly
# into A's original fp16/bf16/fp32 dtype. The scalar path additionally must NOT
# set the TRITONXPU_COMPARE_FUSION / FP16_FAST fusion env vars (tensor-vs-scalar
# fp16 compare trips `arith.cmpf same-type` -> uni_sram overflow compile fail).
config_inplace_ = CodeGenConfig(
    512,
    (65536, 65536, 65536),
    32,
    True,
    prefer_1d_tile=True,
    kunlunAutoGrid=True,
    unroll_num=8,
)


@pointwise_dynamic(promotion_methods=[(0, 1, "ALWAYS_BOOL")], config=config_inplace_)
@triton.jit
def lt_func_(x, y):
    return tl.where(x.to(tl.float32) < y.to(tl.float32), 1, 0)


def lt_(A, B):
    logger.debug("GEMS_KUNLUNXIN LT_")
    if A.device != B.device:
        B = B.to(A.device)
    lt_func_(A, B, out0=A)
    return A


@pointwise_dynamic(
    is_tensor=[True, False],
    promotion_methods=[(0, 1, "ALWAYS_BOOL")],
    config=config_inplace_,
)
@triton.jit
def lt_func_scalar_(x, y):
    return tl.where(x.to(tl.float32) < y, 1, 0)


def lt_scalar_(A, B):
    logger.debug("GEMS_KUNLUNXIN LT_SCALAR_")
    lt_func_scalar_(A, B, out0=A)
    return A
