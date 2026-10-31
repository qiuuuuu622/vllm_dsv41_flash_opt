# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU collectives. Shared packing functions are exercised by portable tests."""

import torch
import torch.distributed as dist

from .layout import (
    RowPartition,
    pack_heads_to_rows,
    pack_rows_to_heads,
    unpack_heads_to_rows,
    unpack_rows_to_heads,
)


def project_rows(x, weight, group):
    plan = RowPartition(x.shape[0], group.world_size)
    local = plan.shard(x, group.rank_in_group)
    if local.device.type == "cuda":
        projected = torch.mm(local, weight.T, out_dtype=torch.float32)
    else:
        # Portable Gloo validation; the serving hook only selects CUDA H20.
        projected = torch.mm(local.float(), weight.float().T)
    return group.all_gather(projected, dim=0)[: plan.rows]


def _exchange(send, process_group):
    receive = torch.empty_like(send)
    dist.all_to_all_single(receive, send, group=process_group)
    return receive


def heads_to_rows(q, group):
    sent = pack_heads_to_rows(q, group.world_size)
    return unpack_heads_to_rows(_exchange(sent, group.device_group))


def rows_to_heads(out, rows, group):
    sent = pack_rows_to_heads(out, group.world_size)
    return unpack_rows_to_heads(_exchange(sent, group.device_group), rows)
