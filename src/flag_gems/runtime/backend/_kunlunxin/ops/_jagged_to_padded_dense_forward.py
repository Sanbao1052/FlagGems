# Copyright 2026, The FlagOS Contributors.
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
#
# Kunlunxin (XPU) override of aten::_jagged_to_padded_dense_forward.
#
# The generic kernel (src/flag_gems/ops/_jagged_to_padded_dense_forward.py)
# writes the output TWICE: first it fills the entire row with the padding
# value, then it overwrites [0, seq_length) with the gathered values.  On XPU
# that doubles global store traffic, which is the dominant cost of this
# gather/scatter-shaped op.
#
# This override writes every output element exactly once with two disjoint
# store streams per row:
#   * a "copy" stream: masked load of values[seq_start : seq_end] and a store
#     whose mask is the *same* predicate (the XPU backend fuses a masked load
#     into the following masked store and only honors one mask; identical
#     predicates make the fusion correct), and
#   * a "tail pad" stream: pure constant store of the padding value over
#     [seq_length : max_length) (no load feeds it, so it cannot be merged with
#     the copy stream).
# BLOCK_SIZE=256 makes the common shapes (max_length <= 256) use a single
# iteration per stream, which measured fastest on XPU.
import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils import libentry
from flag_gems.utils import triton_lang_extension as tle

logger = logging.getLogger(__name__)


@libentry()
@triton.jit
def _jagged_to_padded_dense_forward_kernel(
    values,
    offsets,
    output,
    padding_value: tl.constexpr,
    batch_size: tl.constexpr,
    max_length: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel for converting jagged tensor to padded dense tensor.

    Args:
        values: 1D tensor containing concatenated variable-length sequences
        offsets: 1D tensor of start positions for each sequence
        output: 2D output tensor of shape (batch_size, max_length)
        padding_value: scalar value for padding
        batch_size: number of sequences
        max_length: maximum length of each sequence
    """
    pid = tle.program_id(axis=0)
    if pid >= batch_size:
        return

    # Get the start and end offset for this sequence
    seq_start = tl.load(offsets + pid)
    seq_end = tl.load(offsets + pid + 1)
    seq_length = seq_end - seq_start

    row_offset = pid * max_length

    # Copy actual values (vectorized per block).  Since seq_length <=
    # max_length, the store mask below is exactly the load mask, which keeps
    # the XPU backend's load/store fusion correct.
    for j in tl.range(0, seq_length, BLOCK_SIZE):
        offsets_vec = seq_start + j + tl.arange(0, BLOCK_SIZE)
        mask = offsets_vec < seq_end

        values_vec = tl.load(values + offsets_vec, mask=mask, other=padding_value)
        tl.store(output + row_offset + j + tl.arange(0, BLOCK_SIZE), values_vec, mask=mask)

    # Fill the tail [seq_length, max_length) with the padding value.  This is
    # a constant store with no feeding load, so it stays a separate store
    # stream and the output is written exactly once in total.
    tail = max_length - seq_length
    for j in tl.range(0, tail, BLOCK_SIZE):
        tail_offsets = seq_length + j + tl.arange(0, BLOCK_SIZE)
        tail_mask = tail_offsets < max_length
        tl.store(
            output + row_offset + tail_offsets, padding_value, mask=tail_mask
        )


def _jagged_to_padded_dense_forward(values, offsets, max_lengths, padding_value=0.0):
    """Convert a jagged (variable-length) tensor to a padded dense tensor.

    Args:
        values: 1D tensor containing concatenated variable-length sequences
        offsets: List of 1D tensors containing start positions for each sequence
        max_lengths: List of integers specifying maximum length for each batch dimension
        padding_value: Value to use for padding (default: 0.0)

    Returns:
        Padded dense tensor
    """
    logger.debug("GEMS_KUNLUNXIN JAGGED TO PADDED DENSE FORWARD")

    # Currently only supports single batch dimension
    if not isinstance(offsets, (list, tuple)):
        offsets = [offsets]
    if not isinstance(max_lengths, (list, tuple)):
        max_lengths = [max_lengths]

    num_batch_dims = len(offsets)
    assert (
        num_batch_dims == 1
    ), f"Only single batch dimension is supported, got {num_batch_dims}"

    # Single batch dimension: 1D values, 1D offsets
    offsets_0 = offsets[0]
    batch_size = offsets_0.numel() - 1
    max_length = max_lengths[0]

    # Compute output shape
    # For single batch dim: (batch_size, max_length)
    output_shape = (batch_size, max_length)
    output = torch.empty(output_shape, dtype=values.dtype, device=values.device)

    grid = lambda meta: (batch_size,)
    _jagged_to_padded_dense_forward_kernel[grid](
        values,
        offsets_0,
        output,
        padding_value,
        batch_size,
        max_length,
        # BLOCK_SIZE=256: single iteration for the common max_length <= 256
        # shapes and the fastest measured configuration on XPU.
        BLOCK_SIZE=256,
    )

    return output