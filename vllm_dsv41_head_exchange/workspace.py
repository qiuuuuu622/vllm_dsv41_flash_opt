# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent NCCL reference and fused peer-push workspaces.

Construct collectively outside CUDA Graph capture. Inputs and KV/index metadata
must describe the same rows on every TP rank. A workspace is serialized on one
stream (capture may use a different stream after explicit synchronization).
Returned buffers are borrowed until the next invocation. No serving hook is
installed by importing this module.
"""

import ctypes
import math
import socket

import torch
import torch.distributed as dist

from .build import library_path
from .layout import HeadRowLayout


class _Workspace:
    def __init__(self, q, sink, group=None, *, blocks=None, threads=256):
        self.group = group if group is not None else dist.group.WORLD
        self.world = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        self.device = q.device
        # Agree on eligibility/configuration before allocating peer buffers.
        error = None
        try:
            self.plan = HeadRowLayout(q.shape[0], self.world)
            if q.shape[1:] != (1, 64, 512) or q.dtype != torch.bfloat16:
                raise ValueError("Q must be BF16 [M, 1, 64, 512], eight useful heads")
            if not q.is_cuda or not q.is_contiguous():
                raise ValueError("Q must be a contiguous CUDA tensor")
            if torch.cuda.get_device_capability(q.device) != (9, 0):
                raise ValueError("This prototype requires SM90")
            if dist.get_backend(self.group) != "nccl":
                raise ValueError("A single-node NCCL process group is required")
            if (
                sink.shape != (64,)
                or sink.dtype != torch.float32
                or sink.device != q.device
            ):
                raise ValueError("Sink must be FP32 [64] on the Q device")
            if torch.cuda.is_current_stream_capturing():
                raise ValueError("Construct the workspace before graph capture")
            total_bytes = self.world * self.plan.chunk * self.plan.segment_bytes
            self.blocks = (
                blocks
                if blocks is not None
                else min(16, max(1, math.ceil(total_bytes / 32768)))
            )
            self.threads = threads
            if not 1 <= self.blocks <= 32 or not 32 <= threads <= 1024 or threads % 32:
                raise ValueError(
                    "Expected 1..32 blocks and 32..1024 warp-aligned threads"
                )
            self._prepare_transport()
        except (ValueError, OSError, RuntimeError) as exc:
            error = str(exc)
        spec = (
            socket.gethostname(),
            type(self).__name__,
            tuple(q.shape),
            str(q.dtype),
            getattr(self, "blocks", None),
            threads,
        )
        gathered = [None] * self.world
        dist.all_gather_object(gathered, (error, spec), group=self.group)
        if any(item[0] for item in gathered) or any(
            item[1] != spec for item in gathered
        ):
            raise ValueError(f"Incompatible head-exchange ranks: {gathered}")
        self.rows, self.chunk = self.plan.rows, self.plan.chunk
        self.active = self.plan.active_heads
        self.input = q  # Keep the graph-bound pointer alive.
        self.sink = torch.full_like(sink, -torch.inf)
        gathered_sink = torch.empty(self.active, device=q.device, dtype=torch.float32)
        dist.all_gather_into_tensor(
            gathered_sink, sink[:8].contiguous(), group=self.group
        )
        self.sink[: self.active].copy_(gathered_sink)

    def _prepare_transport(self):
        pass

    def metadata(self, tensor, invalid=0):
        return self.plan.shard(tensor, self.rank, invalid)

    def _check_input(self, q):
        if q.data_ptr() != self.input.data_ptr() or q.shape != self.input.shape:
            raise ValueError("Workspace is bound to its original Q buffer")

    def _ready(self):
        torch.accelerator.synchronize(self.device)
        dist.barrier(group=self.group)


class NcclWorkspace(_Workspace):
    """Reference implementation with packing and two NCCL all-to-all calls."""

    def __init__(self, q, sink, group=None, **kwargs):
        super().__init__(q, sink, group, **kwargs)
        shape = (self.world, self.chunk, 8, 512)
        self.q_send = torch.zeros(shape, device=q.device, dtype=q.dtype)
        self.q_recv = torch.empty_like(self.q_send)
        self.o_send = torch.empty_like(self.q_send)
        self.o_recv = torch.empty_like(self.q_send)
        self.row_q = torch.zeros(self.chunk, 1, 64, 512, device=q.device, dtype=q.dtype)
        self.row_o = torch.empty_like(self.row_q)
        self.output = torch.zeros_like(q)
        self._ready()

    def forward_q(self, q):
        self._check_input(q)
        self.q_send.view(-1, 8, 512)[: self.rows].copy_(q[:, 0, :8])
        dist.all_to_all_single(self.q_recv, self.q_send, group=self.group)
        self.row_q[:, 0, : self.active].view(self.chunk, self.world, 8, 512).copy_(
            self.q_recv.permute(1, 0, 2, 3)
        )
        return self.row_q

    def backward_o(self):
        self.o_send.copy_(
            self.row_o[:, 0, : self.active]
            .view(self.chunk, self.world, 8, 512)
            .permute(1, 0, 2, 3)
        )
        dist.all_to_all_single(self.o_recv, self.o_send, group=self.group)
        self.output[:, 0, :8].copy_(self.o_recv.view(-1, 8, 512)[: self.rows])
        return self.output


class _Plane:
    def __init__(self, data_bytes, owner):
        import torch.distributed._symmetric_memory as symm

        self.flag_offset = (data_bytes + 255) // 256 * 256
        self.buffer = symm.empty(
            self.flag_offset + 128 * owner.blocks,
            dtype=torch.uint8,
            device=owner.device,
        )
        self.buffer.zero_()
        self.handle = symm.rendezvous(self.buffer, owner.group)
        pointers = self.handle.buffer_ptrs
        if pointers[owner.rank] != self.buffer.data_ptr():
            raise RuntimeError("Symmetric-memory peer ordering does not match TP group")
        self.peer_array = torch.tensor(pointers, dtype=torch.int64, device=owner.device)
        self.data = self.buffer[:data_bytes]


class FusedWorkspace(_Workspace):
    """Push eight-head segments directly into final Q/O layouts over NVLink."""

    def _prepare_transport(self):
        self.library = ctypes.CDLL(str(library_path()))
        self.launch = self.library.launch_segment_exchange
        self.launch.restype = ctypes.c_int
        self.launch.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_uint64,
            *([ctypes.c_int64] * 4),
            *([ctypes.c_int] * 7),
            ctypes.c_void_p,
        ]

    def __init__(self, q, sink, group=None, **kwargs):
        super().__init__(q, sink, group, **kwargs)
        row_bytes = self.plan.row_bytes
        self.q_plane = _Plane(self.chunk * row_bytes, self)
        self.o_plane = _Plane(self.world * self.chunk * row_bytes, self)
        self.row_q = self.q_plane.data.view(torch.bfloat16).view(self.chunk, 1, 64, 512)
        self.row_o = torch.empty_like(self.row_q)
        self.output = self.o_plane.data.view(torch.bfloat16).view(
            self.world * self.chunk, 1, 64, 512
        )[: self.rows]
        self._ready()

    def _call(self, source, plane, direction, valid_rows):
        rc = self.launch(
            source.data_ptr(),
            plane.peer_array.data_ptr(),
            plane.flag_offset,
            *self.plan.strides(direction),
            self.chunk,
            valid_rows,
            self.plan.segment_bytes,
            self.world,
            self.rank,
            self.blocks,
            self.threads,
            torch.cuda.current_stream(self.device).cuda_stream,
        )
        if rc:
            raise RuntimeError(f"Fused head exchange launch failed: CUDA error {rc}")

    def forward_q(self, q):
        self._check_input(q)
        self._call(q, self.q_plane, "q", self.rows)
        return self.row_q

    def backward_o(self):
        self._call(self.row_o, self.o_plane, "o", self.world * self.chunk)
        return self.output
