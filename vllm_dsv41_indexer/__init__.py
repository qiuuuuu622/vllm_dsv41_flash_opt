# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Explicit FP8 candidate-indexer API; importing installs no serving hooks."""

from .parallel import IndexerParallel

__all__ = ["IndexerParallel"]
