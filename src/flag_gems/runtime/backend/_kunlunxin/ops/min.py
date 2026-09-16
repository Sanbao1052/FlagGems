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

import builtins
import logging
import math
import os
from collections import namedtuple

import torch
import triton
import triton.language as tl

# from flag_gems import runtime
from flag_gems.runtime import torch_device_fn
from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as ext
from flag_gems.utils.limits import get_dtype_max

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# tle.raw fast path (P800 xpu3, cluster C payload in min_raw.xpu). min_dim
# routes every supported contiguous inner-dim (K == 1) reduction here first:
# the compiler row-reduce is structurally capped on this XPU (wide-row
# CoreTiling serialization + uni_sram OOR), the payload reaches 0.77-1.0 on
# the core shapes. MIN_USE_TLE=0 disables it (tle.gpu / pure-Triton fallback
# only; used to evaluate the skill kernels).
# ---------------------------------------------------------------------------
_MIN_USE_TLE = os.environ.get("MIN_USE_TLE", "1") != "0"  # default ON (raw payload approved for the row-reduce path); MIN_USE_TLE=0 forces the tle.gpu / pure-Triton fallback
try:
    import triton.experimental.tle as tle

    _TLE_OK = _MIN_USE_TLE
except ImportError:
    tle = None
    _TLE_OK = False

_HERE = os.path.dirname(os.path.abspath(__file__))
_NCLUSTER = 12  # P800 (xpu3): one Triton program == one cluster of 64 cores
# Payload scalars are i32 (do_not_specialize); guard the byte range.
_RAW_MIN_ELEMS = 2**31 - 1

_RAW_TYPE_CODE = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.int32: 3,
    torch.int64: 4,
    torch.int16: 5,
    torch.int8: 6,
    torch.uint8: 7,
    torch.bool: 7,
    torch.float64: 8,
}

if _TLE_OK:

    @tle.raw.dialect("xpu3", file=os.path.join(_HERE, "min_raw.xpu"),
                     flags=[f"-I{_HERE}"])
    def min_row_raw(in_, out_val, out_idx, M, N, esz, type_code, rows_start,
                    rows_count, rpc):
        ...

    @triton.jit(do_not_specialize=["M", "N", "esz", "type_code", "per", "rpc"])
    def min_dim_raw_kernel(In, OutV, OutI, M, N, esz, type_code, per, rpc):
        pid = tl.program_id(0)
        tle.raw.call(
            min_row_raw, (In, OutV, OutI, M, N, esz, type_code, pid * per, per, rpc)
        )

    @tle.raw.dialect("xpu3", file=os.path.join(_HERE, "min_full_sm.xpu"),
                     flags=[f"-I{_HERE}"])
    def min_full_raw(in_, mid, M, esz, type_code, per, slot):
        ...

    @triton.jit(do_not_specialize=["M", "esz", "type_code", "per"])
    def min_full_raw_kernel(In, Mid, M, esz, type_code, per):
        pid = tl.program_id(0)
        tle.raw.call(min_full_raw, (In, Mid, M, esz, type_code, per, pid))

    @tle.raw.dialect("xpu3", file=os.path.join(_HERE, "min_full_sm.xpu"),
                     flags=[f"-I{_HERE}"])
    def min_combine_raw(mid, out, n, esz, type_code):
        ...

    @triton.jit(do_not_specialize=["n", "esz", "type_code"])
    def min_full_combine_kernel(Mid, Out, n, esz, type_code):
        tle.raw.call(min_combine_raw, (Mid, Out, n, esz, type_code))


def _view_u8(t):
    """Byte view of a tensor; works for 0-dim tensors too."""
    if t.dim() == 0:
        return t.view(1).view(torch.uint8)
    return t.view(torch.uint8)


def _raw_min_dim(inp, dim, keepdim):
    """min.dim along the innermost (contiguous) dim via the raw payload.

    Returns (values, indices) or None when the raw path does not apply.
    """
    if not _TLE_OK:
        return None
    type_code = _RAW_TYPE_CODE.get(inp.dtype)
    if type_code is None:
        return None
    shape = inp.shape
    N = shape[dim]
    M = shape[:dim] and math.prod(shape[:dim]) or 1
    K = inp.numel() // M // N
    if K != 1:  # payload assumes contiguous rows of length N
        return None
    M = inp.numel() // N  # total rows (M * K with K == 1)
    if M * N > _RAW_MIN_ELEMS:
        return None
    esz = inp.element_size()

    shape_list = list(shape)
    shape_list[dim] = 1
    out_value = torch.empty(shape_list, dtype=inp.dtype, device=inp.device)
    out_index = torch.empty(shape_list, dtype=torch.int64, device=inp.device)
    if not keepdim:
        out_value = torch.squeeze(out_value, dim)
        out_index = torch.squeeze(out_index, dim)

    # Rows per core: each core's output block must be >= 8 bytes (narrow <8B
    # global stores are pathologically slow on this device). For large M the
    # default 12-cluster split already yields >=8B blocks; for small M we
    # consolidate into fewer clusters so every active core writes a full rpc
    # block (e.g. [64,64] fp16: rpc=4 -> 16 active cores, 8B stores).
    per_core_default = (math.ceil(M / _NCLUSTER) + 63) // 64
    rpc = builtins.max(per_core_default, (8 + esz - 1) // esz)
    if per_core_default * esz < 8:
        grid_n = builtins.max(1, builtins.min(_NCLUSTER, (M + rpc * 64 - 1) // (rpc * 64)))
    else:
        grid_n = _NCLUSTER
    per = (M + grid_n - 1) // grid_n
    with torch_device_fn.device(inp.device):
        min_dim_raw_kernel[(grid_n,)](
            _view_u8(inp),
            _view_u8(out_value),
            out_index,
            M,
            N,
            esz,
            type_code,
            per,
            rpc,
        )
    return out_value, out_index


def _raw_min_full(inp):
    """Full-tensor min via per-cluster partials + a tiny combine kernel.

    Returns the scalar result tensor, or None when not applicable. The payload
    does a per-cluster SM reduce (12 GM partials) instead of the 768 narrow
    per-core stores that dominated the 1D latency.
    """
    if not _TLE_OK:
        return None
    type_code = _RAW_TYPE_CODE.get(inp.dtype)
    if type_code is None:
        return None
    M = inp.numel()
    if M > _RAW_MIN_ELEMS:
        return None
    esz = inp.element_size()

    ncores = _NCLUSTER * 64
    # mid: 8-byte slots. float dtypes + i32 use the per-cluster SM reduce
    # inside the payload (12 slots); other dtypes write per-core 8B partials
    # (768 slots). The combine reads the first nused slots either way.
    float_i32 = inp.dtype in (torch.float32, torch.float16, torch.bfloat16,
                              torch.int32)
    nused = builtins.min(_NCLUSTER, M) if float_i32 else builtins.min(ncores, M)
    per = (M + ncores - 1) // ncores
    mid = torch.empty(ncores * 8, dtype=torch.uint8, device=inp.device)
    out = torch.empty([], dtype=inp.dtype, device=inp.device)
    with torch_device_fn.device(inp.device):
        min_full_raw_kernel[(_NCLUSTER,)](
            _view_u8(inp),
            mid,
            M,
            esz,
            type_code,
            per,
        )
        min_full_combine_kernel[(1,)](
            mid,
            _view_u8(out),
            nused,
            esz,
            type_code,
        )
    return out

# ---------------------------------------------------------------------------
# tle.gpu fast path for min.dim (fp16, contiguous innermost dim, K == 1),
# following PR #6311's sum_dim row-reduce pattern: one TensorDescriptor per
# [XBLOCK, YBLOCK] tile, tle.gpu.copy GM -> LM -> registers, and the running
# per-row best kept as a packed int32 key. The value and the first-min column
# are packed into one word -- key = signed_order(value16) << 16 | col -- so a
# single tl.min(key, axis=1) per tile resolves both, ties going to the FIRST
# column (ATen semantics). The order transform is the signed-order encoding
# (negative floats -> (~u)^0x8000, positive -> u), which needs SHIFT == 16 so
# the value's sign bit lands on key bit 31 (the int32 min on this backend is
# SIGNED; SHIFT < 16 inverts the ordering -- measured). The index ALU is the
# dominant cost here, so YBLOCK is pushed to the LM limit (2048 fp16 at XBLOCK
# 64) to minimise the number of per-tile reduces.
#
# Short tiles are NOT masked: the copy clamps to the descriptor, the stale LM
# bytes are cleared with the min identity (dtype-max) on the one step that can
# be short, and rows whose winning key came from the padding (col >= N, i.e.
# every real element is +inf) are repaired to +inf / index 0.
try:
    import triton.experimental.tle.language as tle
    from triton.tools.tensor_descriptor import TensorDescriptor

    _TLE_GPU_OK = True
except ImportError:
    tle = None
    TensorDescriptor = None
    _TLE_GPU_OK = False


if _TLE_GPU_OK:

    @triton.jit(do_not_specialize=["N"],
                do_not_specialize_on_alignment=["a_desc", "ov_desc", "oi_desc"])
    def _tle_min_row_kernel(
        a_desc, ov_desc, oi_desc, N,
        XBLOCK: tl.constexpr, YBLOCK: tl.constexpr,
        NEED_PAD: tl.constexpr,
    ):
        pid = tl.program_id(0)
        row_off = pid * XBLOCK
        a_lmem = tle.gpu.alloc([XBLOCK, YBLOCK], dtype=tl.float16,
                               layout=None, scope=tle.gpu.lmem)
        ov_lmem = tle.gpu.alloc([XBLOCK], dtype=tl.float16,
                                layout=None, scope=tle.gpu.lmem)
        oi_lmem = tle.gpu.alloc([XBLOCK], dtype=tl.int64,
                                layout=None, scope=tle.gpu.lmem)
        row_ids = tl.broadcast_to(tl.arange(0, XBLOCK)[:, None], (XBLOCK, YBLOCK))
        col_ids = tl.broadcast_to(tl.arange(0, YBLOCK)[None, :], (XBLOCK, YBLOCK))
        a_ptrs = tle.gpu.local_ptr(a_lmem, (row_ids, col_ids))
        ov_ptrs = tle.gpu.local_ptr(ov_lmem, (tl.arange(0, XBLOCK),))
        oi_ptrs = tle.gpu.local_ptr(oi_lmem, (tl.arange(0, XBLOCK),))
        best = tl.full([XBLOCK], 2147483647, tl.int32)
        for coff in tl.range(0, N, YBLOCK):
            if NEED_PAD:
                if coff + YBLOCK > N:
                    tl.store(a_ptrs, tl.full([XBLOCK, YBLOCK], 65504.0, tl.float16))
            tle.gpu.copy(a_desc, a_lmem, [XBLOCK, YBLOCK], [row_off, coff])
            u16 = tl.load(a_ptrs).to(tl.int16, bitcast=True)
            key16 = tl.where(u16 < 0, (~u16) ^ -32768, u16)
            key = (((key16.to(tl.int32) & 0xFFFF) << 16) | (coff + col_ids))
            best = tl.minimum(best, tl.min(key, axis=1))
        k = best
        col = k & 0xFFFF
        vb = (k >> 16) & 0xFFFF
        v16 = tl.where(vb >= 0x8000, (~vb) ^ -32768, vb)
        vf = v16.to(tl.int16).to(tl.float16, bitcast=True)
        broken = col >= N
        inf_bits = tl.full([XBLOCK], 0x7C00, tl.int16)
        vf = tl.where(broken, inf_bits.to(tl.float16, bitcast=True), vf)
        col = tl.where(broken, 0, col)
        tl.store(ov_ptrs, vf)
        tl.store(oi_ptrs, col.to(tl.int64))
        tle.gpu.copy(ov_lmem, ov_desc, [XBLOCK], [row_off])
        tle.gpu.copy(oi_lmem, oi_desc, [XBLOCK], [row_off])


def _tle_min_dim(inp, dim, keepdim):
    """fp16 min.dim via the tle.gpu row-reduce; None when not applicable."""
    if not _TLE_GPU_OK:
        return None
    if inp.dtype != torch.float16:
        return None
    shape = inp.shape
    N = shape[dim]
    M = shape[:dim] and math.prod(shape[:dim]) or 1
    K = inp.numel() // M // N
    if K != 1:  # payload assumes contiguous rows of length N
        return None
    M = inp.numel() // N
    # The packed int32 key holds 16 value bits + 16 column bits, so N must fit
    # in 16 bits. Below M=256 the per-element ALU + launch overhead loses to
    # the chunked three-pass path; the benchmark-relevant win is M >= 4096.
    if not (256 <= M <= 2**31 - 1 and 64 <= N <= 65535):
        return None
    if M * N > 2**31 - 1:
        return None

    shape_list = list(shape)
    shape_list[dim] = 1
    out_value = torch.empty(shape_list, dtype=inp.dtype, device=inp.device)
    out_index = torch.empty(shape_list, dtype=torch.int64, device=inp.device)

    # fp16 min geometry: XBLOCK 64 (one row per core), YBLOCK to the LM limit
    # (2048 fp16 = 4 KB/core at XBLOCK 64). Measured [4096,4096]: XB=64/YB=2048
    # 368us vs XB=512/YB=256 798us (the per-element key ALU dominates, so few
    # big tiles beat the sum-dim preference for wide XBLOCK).
    xblock = 64
    yblock = triton.next_power_of_2(N) if N < 2048 else 2048
    with torch_device_fn.device(inp.device):
        grid = (triton.cdiv(M, xblock),)
        _tle_min_row_kernel[grid](
            TensorDescriptor.from_tensor(inp, block_shape=[xblock, yblock]),
            TensorDescriptor.from_tensor(out_value.view(-1), block_shape=[xblock]),
            TensorDescriptor.from_tensor(out_index.view(-1), block_shape=[xblock]),
            N,
            xblock,
            yblock,
            N % yblock != 0,
        )
    if not keepdim:
        out_value = torch.squeeze(out_value, dim)
        out_index = torch.squeeze(out_index, dim)
    return out_value, out_index

# NOTE (kunlunxin/XPU): performance recipe (2026-08-17) follows the
# amax/amin 2026-08-16 closure for the value-only paths and the argmin
# 2026-08-11 "packed index" closure for the min.dim (values, indices) path:
#   * masked loading uses `other=+inf` everywhere (NOT get_dtype_max), so
#     all-+inf blocks stay +inf (tests/test_min.py::test_min_all_inf);
#   * NEED_MASK constexpr fast 2D row-reduction: mask-free tiles when M and N
#     both divide the picked [BLOCK_M, BLOCK_N] (skips the XPU masked-memory
#     slow path);
#   * the reduced dim is brought innermost with the native strided copy
#     (`torch.ops.aten._copy_from`; flag_gems does not override _copy_from)
#     instead of the slow gems `.contiguous()` override;
#   * flat (dim=None) reduction = rows-of-8192 with the 2D tile kernel plus a
#     staged 8192-wide mid reduce -- the HEAD staged `min_kernel_1` path
#     (tl.reduce + combine_fn) stalls in the XPU compiler on the very first
#     launch (probe: >10min on a 2-element tensor; matches the 2026-08-08 /
#     2026-08-10 BLOCKED records), so the flat path is replaced entirely;
#   * N == 1 identity: `_copy_from` for values + a zeros index tensor.
# min.dim needs values AND indices. A per-element index-carrying reduce
# (`tl.min(... return_indices=True)` or a packed (value, index) int64 word)
# is ~15-20x slower than a value-only reduce on this XPU, so like argmin we
# split it into three passes:
#   1. min_split_kernel: value-only fp32 min per (row, BLOCK_CHUNK);
#   2. min_chunk_kernel: per-row min over the nc chunk minima (leftmost chunk
#      wins ties on this XPU backend, matching torch "first minimal").
#   3. min_scan_kernel:   re-read only the winning chunk, pack
#      (order-preserving fp32 bit-map << 32) | column into an int64 and take
#      the plain int64 min so the first minimal lane wins; NaN and +inf map
#      above every finite value, exactly the XPU device-native fmin family
#      semantics (amax/amin 2026-08-16 evidence; device native min ignores
#      NaN, an all-NaN row yields NaN/value and index 0).

_FULL_REDUCTION_BLOCK_SIZE = 8192
_FAST_MIN_N = 64
_FAST_BN_FP16 = (1024, 256, 512, 128, 64)
_FAST_BN_FP32 = (512, 256, 1024, 128, 64)
_FAST_BM_FP16 = (128, 64, 32, 256, 512, 16, 8, 4, 2)
_FAST_BM_FP32 = (64, 128, 32, 256, 512, 16, 8, 4, 2)
# XPU compile guard: tiny [BLOCK_M, BLOCK_N] tiles (e.g. [64, 64]) fail the
# TritonXPU `uni_sram` pass ("PassManager::run failed"). Only tiles with at
# least this many lanes are routed to the mask-free row kernels; smaller
# shapes fall back to the legacy masked kernel.
_MIN_FAST_TILE_LANES = 8192


def _is_fast_dtype(dtype):
    return dtype in (torch.float16, torch.float32, torch.bfloat16)


def _pick_fast_tile(M, N, is_fp32):
    """Return (BLOCK_M, BLOCK_N) with M % BLOCK_M == 0, N % BLOCK_N == 0 and
    BLOCK_M * BLOCK_N >= _MIN_FAST_TILE_LANES, or None when no such mask-free
    tile covers this shape (small shapes keep the legacy masked kernel)."""
    if N < _FAST_MIN_N:
        return None
    bns = _FAST_BN_FP32 if is_fp32 else _FAST_BN_FP16
    bms = _FAST_BM_FP32 if is_fp32 else _FAST_BM_FP16
    for bn in bns:
        if N % bn != 0:
            continue
        for bm in bms:
            if M % bm != 0:
                continue
            if bm * bn >= _MIN_FAST_TILE_LANES:
                return bm, bn


@libentry()
@triton.jit
def min_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    pid = ext.program_id(0)

    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    if NEED_MASK:
        mask = offset < M
        inp_val = tl.load(inp_ptrs, mask=mask, other=float("inf"))
    else:
        inp_val = tl.load(inp_ptrs)
    min_val = tl.min(inp_val)
    mid_ptr = mid + pid
    tl.store(mid_ptr, min_val)


def heur_m_block_size(args):
    return triton.next_power_of_2(triton.cdiv(args["M"], 12))  # cluster_num


def heur_n_block_size(args):
    import builtins

    return builtins.min(triton.next_power_of_2(args["N"]), 8192)


@libentry()
@triton.heuristics(
    values={
        "BLOCK_M": heur_m_block_size,
        "BLOCK_N": heur_n_block_size,
    },
)
@triton.jit
def min_kernel(
    inp,
    out_value,
    out_index,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Legacy masked (values, indices) kernel for the non-fast paths (small N,
    # int dtypes). Preserved byte-for-byte from HEAD.
    # set offset
    pid_m = ext.program_id(0)
    pid_k = ext.program_id(1)
    m_offset = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    dtype = inp.type.element_ty
    # you just cannot create a function that return a tl.dtype in triton lang
    acc_type = tl.float32 if dtype is tl.bfloat16 else dtype
    max_value = get_dtype_max(dtype)
    min_values = tl.full([BLOCK_M], dtype=acc_type, value=max_value)
    argmin_values = tl.full([BLOCK_M], dtype=tl.int64, value=0)
    for start_n in range(0, N, BLOCK_N):
        n_offset = start_n + tl.arange(0, BLOCK_N)
        offset = m_offset[:, None] * N * K + n_offset[None, :] * K + pid_k
        mask = m_offset[:, None] < M and n_offset[None, :] < N
        inp_ptrs = inp + offset
        inp_vals = tl.load(inp_ptrs, mask=mask, other=max_value)
        local_min, local_argmin = tl.min(inp_vals, 1, return_indices=True)
        update = local_min < min_values
        min_values = tl.where(update, local_min, min_values)
        argmin_values = tl.where(update, start_n + local_argmin, argmin_values)

    # f32 identity-clamp (see min_kernel_1 note): an all-+inf f32 row must
    # give the device-native FLT_MAX, matching torch.min on this backend.
    if dtype is tl.float32:
        min_values = tl.minimum(min_values, 3.4028234663852886e+38)

    offset_index = m_offset * K + pid_k
    out_value_ptrs = out_value + offset_index
    out_index_ptrs = out_index + offset_index
    mask1 = m_offset < M
    tl.store(out_value_ptrs, min_values, mask=mask1)
    tl.store(out_index_ptrs, argmin_values, mask=mask1)


@libentry()
@triton.jit
def min_kernel_2d(
    inp,
    out,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # Value-only 2D row reduction (flat rows-of-8192 / plain dim reduction).
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows * N
    out = out + rows
    row_mask = rows < M

    # Keep only a [BLOCK_M, 1] running accumulator and reduce each [BLOCK_M,
    # BLOCK_N] block along N *inside* the loop (reduce-INSIDE; the only form
    # that is numerically correct on this XPU -- see amax_kernel_2d notes).
    # NEED_MASK=False compiles to a mask-free kernel (M and N both divide by
    # the block sizes), avoiding the XPU masked-memory slow path entirely.
    acc = tl.full([BLOCK_M, 1], value=float("inf"), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)[None, :]
        if NEED_MASK:
            col_mask = cols < N
            mask = row_mask and col_mask
            a = tl.load(inp + cols, mask, other=float("inf")).to(tl.float32)
            a = tl.where(mask, a, float("inf"))
            blk = tl.min(a, axis=1)[:, None]
        else:
            a = tl.load(inp + cols).to(tl.float32)
            blk = tl.min(a, axis=1)[:, None]
        acc = tl.minimum(acc, blk)
    if NEED_MASK:
        tl.store(out, acc, row_mask)
    else:
        tl.store(out, acc)


def _pad_value(dtype):
    """Identity value for the min reduction: +inf for floats, dtype max otherwise."""
    if dtype.is_floating_point:
        return float("inf")
    return torch.iinfo(dtype).max


def _pad_buffer(src, pad_to, device):
    """Return a buffer of `pad_to` elements: [0, n) = src, [n, pad_to) filled
    with the reduction identity (+inf for floats, dtype-max for ints)."""
    n = src.numel()
    buf = torch.full((pad_to,), _pad_value(src.dtype), dtype=src.dtype, device=device)
    if n:
        torch.ops.aten._copy_from(src, buf[:n], False)
    return buf


def _reduce_to_scalar(src, out):
    """Repeatedly reduce `src` with 8192-wide blocks until a scalar remains.
    Every masked lane is eliminated by padding each stage to a multiple of
    8192 with the reduction identity (XPU masked tails read OOB -- backend
    limitation; see the module note)."""
    block = _FULL_REDUCTION_BLOCK_SIZE
    n = src.numel()
    data = src
    while n > block:
        n_pad = triton.cdiv(n, block) * block
        if n_pad > n:
            data = _pad_buffer(data, n_pad, data.device)
        mid_size = triton.cdiv(n, block)
        mid = torch.empty((mid_size,), dtype=data.dtype, device=data.device)
        min_kernel_1[(mid_size, 1)](
            data,
            mid,
            n,
            block,
            False,
            buffer_size_limit=2048,
        )
        data = mid
        n = mid_size
    if n < block:
        data = _pad_buffer(data, block, data.device)
    min_kernel_1[(1, 1)](
        data,
        out,
        n,
        block,
        False,
        buffer_size_limit=2048,
    )


def _min_flat(inp, out, device):
    """Full (dim=None) reduction over `inp` (any numel). Mask-free 2-D row
    tiles over rows-of-8192 plus a staged scalar reduce; every tail is reads
    are padded so no masked/OOB load can happen on the XPU backend."""
    block = _FULL_REDUCTION_BLOCK_SIZE
    numel = inp.numel()
    pad_val = _pad_value(inp.dtype)
    if numel <= block:
        buf = torch.full((block,), pad_val, dtype=inp.dtype, device=device)
        torch.ops.aten._copy_from(inp, buf[:numel], False)
        min_kernel_1[(1, 1)](
            buf,
            out,
            numel,
            block,
            False,
            buffer_size_limit=2048,
        )
        return
    rows = numel // block
    res = numel - rows * block
    is_fp32 = inp.dtype == torch.float32
    bn = 1024 if not is_fp32 else 256
    if block % bn != 0:
        bn = next((b for b in _FAST_BN_FP32 if block % b == 0), block)
    bm = 128 if not is_fp32 else 64
    if rows % bm != 0:
        bm = next(
            (
                m
                for m in _FAST_BM_FP32
                if rows % m == 0 and m * bn >= _MIN_FAST_TILE_LANES
            ),
            128,
        )
    # exact (rows // bm) * bm rows run mask-free on the input directly; the
    # leftover rows are copied into a padded [bm, block] buffer first so the
    # same mask-free kernel covers them.
    rows_exact = (rows // bm) * bm
    mid = torch.empty((rows + (1 if res else 0),), dtype=inp.dtype, device=device)
    with torch_device_fn.device(device):
        if rows_exact:
            min_kernel_2d[(rows_exact // bm, 1)](
                inp,
                mid,
                rows_exact,
                block,
                bm,
                bn,
                False,
                buffer_size_limit=2048,
            )
        if rows > rows_exact:
            # The leftover rows are copied into a fully-allocated padded
            # [bm, block] buffer and reduced with a fully mask-free launch
            # (M = bm covers the whole padded buffer; extra rows are the
            # reduction identity and cannot win). The bm row-minima go to a
            # scratch buffer first so the real mid output is never written
            # out of bounds. (XPU masked tails read OOB and can also return
            # wrong values -- backend limitation, see the module note.)
            tail_rows = rows - rows_exact
            tail_buf = torch.full(
                (bm * block,), pad_val, dtype=inp.dtype, device=device
            )
            torch.ops.aten._copy_from(
                inp[rows_exact * block : rows * block],
                tail_buf[: tail_rows * block],
                False,
            )
            tail_mid = torch.empty((bm,), dtype=inp.dtype, device=device)
            min_kernel_2d[(1, 1)](
                tail_buf,
                tail_mid,
                bm,
                block,
                bm,
                bn,
                False,
                buffer_size_limit=2048,
            )
            torch.ops.aten._copy_from(
                tail_mid[:tail_rows], mid[rows_exact : rows_exact + tail_rows], False
            )
        if res:
            res_buf = torch.full((block,), pad_val, dtype=inp.dtype, device=device)
            torch.ops.aten._copy_from(inp[rows * block :], res_buf[:res], False)
            min_kernel_1[(1, 1)](
                res_buf,
                mid[rows:],
                res,
                block,
                False,
                buffer_size_limit=2048,
            )
        _reduce_to_scalar(mid, out)


def min(inp):
    logger.debug("GEMS_KUNLUNXIN MIN")
    # tle.raw fast path: the per-cluster SM reduce + tiny combine reaches
    # ~0.9x on the 1D flat shapes; the pure-Triton _min_flat is structurally
    # capped (~0.26x on 1M: a single-cluster 2D row reduce -- more programs
    # measured SLOWER on this backend, so no Triton tiling fixes it).
    inp = inp.contiguous()
    raw = _raw_min_full(inp) if _TLE_OK else None
    if raw is not None:
        return raw
    dtype = inp.dtype
    out = torch.empty([], dtype=dtype, device=inp.device)
    with torch_device_fn.device(inp.device):
        _min_flat(inp, out, inp.device)
    return out


@libentry()
@triton.jit
def min_split_kernel(
    inp,
    part_val,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
    NEED_MASK: tl.constexpr,
):
    # Pass 1 (value-only f32 min, NaN/inf never win: same fmin family
    # semantics as the XPU device-native min; see module note).
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
    inp = inp + rows.to(tl.int64) * N
    row_mask = rows < M
    ic = 0
    for off in range(0, N, BLOCK_CHUNK):
        cols = off + tl.arange(0, BLOCK_CHUNK)[None, :]
        if NEED_MASK:
            mask = row_mask and cols < N
            a = tl.load(inp + cols, mask=mask, other=float("inf")).to(tl.float32)
        else:
            a = tl.load(inp + cols).to(tl.float32)
        blk = tl.min(a, axis=1)[:, None]
        # f32 identity-clamp (see min_kernel_1 note): an all-+inf f32 row must
        # give the device-native FLT_MAX, matching torch.min on this backend.
        if inp.type.element_ty is tl.float32:
            blk = tl.minimum(blk, 3.4028234663852886e+38)
        if NEED_MASK:
            tl.store(part_val + rows.to(tl.int64) + ic * M, blk, mask=row_mask)
        else:
            tl.store(part_val + rows.to(tl.int64) + ic * M, blk)
        ic += 1


@libentry()
@triton.jit
def min_chunk_kernel(
    part_val,
    best_c,
    M,
    NC,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Pass 2: per row, argmin over the NC chunk minima (leftmost chunk wins
    # ties on this XPU backend, matching the torch "first minimal" rule; NC is
    # small, so return_indices is cheap).
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    c_off = tl.arange(0, BLOCK_C)
    cmask = c_off < NC
    vals = tl.load(
        part_val + rows[:, None] + c_off[None, :].to(tl.int64) * M,
        mask=row_mask[:, None] and cmask[None, :],
        other=float("inf"),
    )
    _, bc = tl.min(vals, axis=1, return_indices=True)
    tl.store(best_c + rows.to(tl.int64), bc, mask=row_mask)


@libentry()
@triton.jit
def min_scan_kernel(
    inp,
    part_val,
    best_c,
    out_val,
    out_idx,
    M,
    N,
    BLOCK_M: tl.constexpr,
    BLOCK_CHUNK: tl.constexpr,
):
    # Pass 3: re-read only the winning chunk per row and take the earliest
    # lane equal to the chunk min. The 1/NC data slice is packed into
    # (ordered fp32 value << 32) | column words and reduced with a plain
    # int64 min: the first minimal lane wins (torch tie rule), -0.0 sorts
    # below +0.0, and NaN/+inf bits sort above every finite value so they
    # never win (XPU fmin / device-native semantics; an all-NaN row falls
    # back to index 0 and the NaN value).
    pid = ext.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    row_mask = rows < M
    c = tl.load(best_c + rows)  # [BM] int32
    base = rows.to(tl.int64) * N + c.to(tl.int64) * BLOCK_CHUNK
    cols = tl.arange(0, BLOCK_CHUNK)
    col_ok = cols[None, :] < (N - c * BLOCK_CHUNK)[:, None]
    a = tl.load(
        inp + base[:, None] + cols[None, :],
        mask=row_mask[:, None] and col_ok,
        other=float("inf"),
    ).to(tl.float32)
    u = a.to(tl.int32, bitcast=True)
    neg = u < 0
    ordered = tl.where(neg, ~u, u ^ -2147483648)
    # NOTE: the value/index word must use a 30-bit column shift (like the
    # argmin scan kernel). A full 64-bit `<< 32` is miscompiled by the XPU
    # backend (probed: the int64 min then picks wrong lanes); << 30 with
    # BLOCK_CHUNK <= 8192 < 2^30 is correct on every probe.
    pack = ((ordered.to(tl.int64) & 0xFFFFFFFF) << 30) | cols.to(tl.int64)
    blk = tl.min(pack, axis=1)
    pos = (blk & 0x3FFFFFFF) + c.to(tl.int64) * BLOCK_CHUNK
    # value of the winning lane: the chunk min computed by pass 1 (equal to
    # the winning lane's value, incl. the XPU fmin NaN semantics).
    m = tl.load(part_val + rows.to(tl.int64) + c.to(tl.int64) * M, mask=row_mask)
    tl.store(out_val + rows.to(tl.int64), m, mask=row_mask)
    tl.store(out_idx + rows.to(tl.int64), pos, mask=row_mask)


def min_dim(inp, dim=None, keepdim=False):
    logger.debug("GEMS_KUNLUNXIN MIN_DIM")
    assert dim >= -inp.ndim and dim < inp.ndim, "Invalid dim"
    shape = inp.shape
    dim = dim % inp.ndim
    N = shape[dim]
    M = math.prod(shape[:dim])
    K = inp.numel() // M // N

    # ---- tle.raw fast path (big contiguous inner-dim reductions) ----------
    # Mirrors max_dim: the payload reaches 0.77-1.0 on the core shapes while
    # the compiler row-reduce is structurally capped on this XPU (wide-row
    # CoreTiling serialization + uni_sram OOR on the tiled kernels). The
    # tle.gpu / three-pass / legacy paths below are the fallbacks for shapes
    # the payload cannot handle (K > 1, out-of-i32-range, unsupported dtypes).
    inp = inp.contiguous()
    if len(shape) > 1:
        raw = _raw_min_dim(inp, dim, keepdim) if _TLE_OK else None
        if raw is not None:
            Min_out = namedtuple("min", ["values", "indices"])
            return Min_out(values=raw[0], indices=raw[1])

    shape_list = list(shape)
    shape_list[dim] = 1
    out_value = torch.empty(shape_list, dtype=inp.dtype, device=inp.device)
    out_index = torch.empty(shape_list, dtype=torch.int64, device=inp.device)

    if N == 1:
        # min along a size-1 dim is the identity (value = input, index = 0) --
        # the native strided copy engine instead of launching a kernel.
        with torch_device_fn.device(inp.device):
            torch.ops.aten._copy_from(inp, out_value, False)
        out_index.zero_()
        if not keepdim:
            out_value = torch.squeeze(out_value, dim)
            out_index = torch.squeeze(out_index, dim)
        Min_out = namedtuple("min", ["values", "indices"])
        return Min_out(values=out_value, indices=out_index)

    # ---- tle.gpu fast path (fp16, contiguous innermost dim) ---------------
    tle_res = _tle_min_dim(inp, dim, keepdim) if _TLE_GPU_OK else None
    if tle_res is not None:
        Min_out = namedtuple("min", ["values", "indices"])
        return Min_out(values=tle_res[0], indices=tle_res[1])

    # ---- fast chunked three-pass path (floats only, N >= _FAST_MIN_N) ----
    if N >= _FAST_MIN_N and _is_fast_dtype(inp.dtype):
        M2 = M * K
        is_fp32 = inp.dtype == torch.float32
        tile = _pick_fast_tile(M2, N, is_fp32)
        if tile is not None:
            block_m, block_n = tile
            grid_m = M2 // block_m
            need_mask = False
            # Bring the reduced dim innermost (same order as dim_compress) and
            # materialize with the native strided copy (not gems contiguous).
            perm = [d for d in range(inp.dim()) if d != dim] + [dim]
            view = inp.permute(perm)
            if view.is_contiguous():
                src = view
            else:
                src = torch.empty(list(view.shape), dtype=inp.dtype, device=inp.device)
                with torch_device_fn.device(inp.device):
                    torch.ops.aten._copy_from(view, src, False)
            nc = triton.cdiv(N, block_n)
            part_val = torch.empty((M2, nc), dtype=torch.float32, device=inp.device)
            best_c = torch.empty((M2,), dtype=torch.int32, device=inp.device)
            out_flat = out_value.reshape(-1)
            out_idx_flat = out_index.reshape(-1)
            with torch_device_fn.device(inp.device):
                min_split_kernel[(grid_m,)](
                    src,
                    part_val,
                    M2,
                    N,
                    block_m,
                    block_n,
                    need_mask,
                    buffer_size_limit=2048,
                )
                min_chunk_kernel[(grid_m,)](
                    part_val,
                    best_c,
                    M2,
                    nc,
                    block_m,
                    triton.next_power_of_2(nc),
                    buffer_size_limit=2048,
                )
                min_scan_kernel[(grid_m,)](
                    src,
                    part_val,
                    best_c,
                    out_flat,
                    out_idx_flat,
                    M2,
                    N,
                    block_m,
                    block_n,
                    buffer_size_limit=2048,
                )
            if not keepdim:
                out_value = torch.squeeze(out_value, dim)
                out_index = torch.squeeze(out_index, dim)
            Min_out = namedtuple("min", ["values", "indices"])
            return Min_out(values=out_value, indices=out_index)

    # ---- legacy path (unchanged HEAD behavior, incl. int dtypes) ----------
    inp = inp.contiguous()

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]),
        K,
    )
    # NOTE (kunlunxin/XPU): the `tl.min(..., return_indices=True)` argmin
    # combine makes the `TritonXPUCoreTiling` pass emit incompatible `tt.reduce`
    # slice orders (order=[0,1] value vs order=[1,0] index) for most 2D (K==1)
    # tiles, which fails compilation ("out of resource: uni_sram /
    # PassManager::run failed"). Closing core-tiling side-steps the buggy
    # layout for the argmin reduce and compiles every shape/dtype.
    isCloseCoreTiling = True
    with torch_device_fn.device(inp.device):
        min_kernel[grid](
            inp, out_value, out_index, M, N, K, isCloseCoreTiling=isCloseCoreTiling
        )
    if not keepdim:
        out_value = torch.squeeze(out_value, dim)
        out_index = torch.squeeze(out_index, dim)
    Min_out = namedtuple("min", ["values", "indices"])
    out = Min_out(values=out_value, indices=out_index)
    return out
