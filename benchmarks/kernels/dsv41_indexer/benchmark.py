# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Whole-function A/B, plus four-consumer amortization with distinct queries.

All ordinary compact timings INCLUDE publication. The explicit four-consumer
variant publishes once and then runs four different queries/weights, matching
the same-ratio consumer layers' state lifetime. Source candidate selection is
common to both paths and not measured here. This is not a model benchmark.
"""

import argparse
import itertools
import json

from .common import check_sources, check_vllm, device_info, measure, save


def run_case(rows, length, verify, args, native):
    import torch

    from vllm.model_executor.kernels.attention.dsa.candidate_blocks import (
        apply_candidate_mask,
    )
    from vllm.utils.deep_gemm import (
        fp8_fp4_paged_mqa_logits,
        get_paged_mqa_logits_metadata,
    )
    from vllm_dsv41_indexer import IndexerParallel
    from vllm_dsv41_indexer.score import candidate_logits as score
    from vllm_dsv41_indexer.workspace import CandidateState, CompactIndexer

    from .inputs import VllmSelector, make_inputs
    from .legacy import CandidateWorkspace

    case = dict(
        rows=rows,
        compressed_length=length,
        rows_per_request=verify,
        page_size=64,
        candidate_blocks=2048,
    )
    q, _, weights, cache, table, req, lens, blocks = make_inputs(
        case, stress=args.stress
    )
    if not args.stress:
        # Native source top-k returns unique block IDs and always includes the
        # newest block. The old generator's overwrite could duplicate that ID.
        # Replace only that accidental duplicate with a missing valid block.
        host = blocks.cpu().tolist()
        lengths = lens.cpu().tolist()
        for row, candidates in enumerate(host):
            last = (lengths[row] - 1) // 8
            if last in candidates[:-1]:
                used = set(candidates)
                missing = next(
                    b for b in range((lengths[row] + 7) // 8) if b not in used
                )
                candidates[candidates.index(last)] = missing
        blocks = torch.tensor(host, dtype=torch.int32, device=q.device)
        assert all(len(set(b)) == len(b) for b in host)
    # Four actual consumer layers share the candidate set/KV but own Q/weights.
    qs = [q] + [q.float().roll(i * 7, dims=-1).to(q.dtype) for i in range(1, 4)]
    weights_list = [weights] + [
        weights.roll(i * 3, dims=-1).contiguous() for i in range(1, 4)
    ]
    old = CandidateWorkspace(rows, 2048, q.device, q_dtype=q.dtype)
    new = CandidateState(rows, q.device, parallel=IndexerParallel(args.tp_size, 0))
    consumers = [
        CompactIndexer(new, qs[i], weights_list[i], cache, table, req) for i in range(4)
    ]
    selector = VllmSelector(q.device, rows)
    output = torch.empty_like(consumers[0].output)
    native_n = verify if verify in (1, 2, 4) else 1
    baseline_table = table if native_n == verify else table[req.long()].contiguous()
    baseline_lens = lens.view(-1, native_n)
    metadata = get_paged_mqa_logits_metadata(
        baseline_lens,
        64,
        torch.cuda.get_device_properties(q.device).multi_processor_count,
    )

    def base_scores(layer=0):
        scores = fp8_fp4_paged_mqa_logits(
            (qs[layer].view(-1, native_n, 32, 128), None),
            cache,
            weights_list[layer],
            baseline_lens,
            baseline_table,
            metadata,
            length,
            clean_logits=False,
        )
        apply_candidate_mask(scores, None, lens, blocks, 8, 1)
        return scores

    def baseline(layer=0):
        selector(base_scores(layer), baseline_lens, output, length)
        return output

    def scoring(ws, layer=0):
        score(
            qs[layer],
            weights_list[layer],
            table,
            req,
            ws.blocks,
            ws.counts,
            cache,
            64,
            ws.scores,
            cache_page_size=64,
            candidate_block_size=8,
            fp32_scores=True,
        )

    def previous():
        old.prepare_candidates(blocks, lens)
        scoring(old)
        selector(old.scores, old.counts[:, None], old.selected, 16384)
        return old.map_output()

    def publish():
        new.publish(blocks, lens, source_unique=not args.stress)

    def consume(layer=0):
        return consumers[layer].forward()

    def improved():
        publish()
        return consume()

    def four_baseline():
        for layer in range(4):
            baseline(layer)

    def four_consumers():
        publish()
        for layer in range(4):
            consume(layer)

    checks = []
    old.prepare_candidates(blocks, lens)
    publish()
    assert torch.equal(old.blocks, new.blocks) and torch.equal(old.counts, new.counts)
    positions = (
        new.blocks.long()[..., None] * 8 + torch.arange(8, device=q.device)
    ).flatten(1)
    visible = torch.arange(16384, device=q.device)[None, :] < new.counts[:, None]
    allowed = [set(positions[r][visible[r]].cpu().tolist()) for r in range(rows)]
    counts = new.counts.cpu().tolist()
    for layer in range(4):
        full = base_scores(layer)
        selector(full, baseline_lens, output, length)
        base = output.clone()
        expected = full.gather(1, positions.clamp(0, length - 1))
        finite = (
            torch.isfinite(full.gather(1, base.long().clamp(0, length - 1)))
            .cpu()
            .tolist()
        )
        got = consume(layer).clone()
        diff = (consumers[layer].scores[visible] - expected[visible]).float()
        nrmse = (diff.norm() / expected[visible].float().norm().clamp_min(1e-20)).item()
        recall = []
        boundary_differences = []
        for r, (a, b) in enumerate(zip(base.cpu().tolist(), got.cpu().tolist())):
            target = {x for x, f in zip(a, finite[r]) if x >= 0 and f}
            selected = [x for x in b if x >= 0]
            assert len(selected) == len(set(selected)) == min(512, counts[r])
            assert set(selected) <= allowed[r]
            recall.append(
                len(target & set(selected)) / len(target)
                if target
                else float(not selected)
            )
            if set(selected) != target:
                missing, extra = (
                    sorted(target - set(selected)),
                    sorted(set(selected) - target),
                )
                boundary_differences.append(
                    dict(
                        row=r,
                        missing=missing,
                        extra=extra,
                        missing_baseline_scores=full[r, missing].cpu().tolist(),
                        extra_baseline_scores=full[r, extra].cpu().tolist(),
                    )
                )
        assert nrmse <= 1e-4 and min(recall) >= 0.99, (nrmse, min(recall))
        checks.append(
            dict(
                layer=layer,
                score_nrmse=nrmse,
                min_topk_set_recall=min(recall),
                score_max_abs=diff.abs().max().item() if diff.numel() else 0.0,
                boundary_differences=boundary_differences,
            )
        )
    result = dict(case=case, correctness=checks, timings={})
    variants = [
        ("vllm_original", baseline),
        ("previous_port", previous),
        ("one_sort_ant_topk_fused_map", improved),
        ("four_vllm_consumers", four_baseline),
        ("four_compact_consumers_publication_included", four_consumers),
    ]
    if args.reverse_order:
        variants.reverse()
    for mode in args.modes:
        result["timings"][mode] = {
            name: measure(
                fn, repeats=args.repeats, inner=args.inner, graph=mode == "graph"
            )
            for name, fn in variants
        }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=int, nargs="+", default=[1, 8, 32, 128])
    p.add_argument("--context", type=int, nargs="+", default=[65536, 131072, 262144])
    p.add_argument("--verify", type=int, nargs="+", default=[1])
    p.add_argument(
        "--modes", nargs="+", choices=["eager", "graph"], default=["eager", "graph"]
    )
    p.add_argument("--stress", action="store_true")
    p.add_argument("--reverse-order", action="store_true")
    p.add_argument(
        "--tp-size",
        type=int,
        choices=[1, 2, 4, 8],
        default=1,
        help="Logical TP contract; this benchmark runs one physical GPU",
    )
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--inner", type=int, default=10)
    p.add_argument("--output", default="build/indexer/benchmark.json")
    args = p.parse_args()
    import torch

    torch.accelerator.set_device_index(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    from vllm_dsv41_indexer.native import load_kernels

    native = load_kernels()
    result = dict(
        status="running",
        args=vars(args),
        device=device_info(),
        baseline=check_vllm(),
        provenance=check_sources(),
        cases=[],
    )
    try:
        for rows, length, verify in itertools.product(
            args.rows, args.context, args.verify
        ):
            if rows % verify:
                continue
            assert length >= 16384
            entry = run_case(rows, length, verify, args, native)
            result["cases"].append(entry)
            save(args.output, result)
            print(
                json.dumps(
                    dict(
                        case=entry["case"],
                        checks=entry["correctness"],
                        timings={
                            mode: {k: v["p50_us"] for k, v in t.items()}
                            for mode, t in entry["timings"].items()
                        },
                    )
                ),
                flush=True,
            )
        result["status"] = "completed"
    except BaseException as e:
        result.update(status="failed", error=repr(e))
        raise
    finally:
        save(args.output, result)


if __name__ == "__main__":
    main()
