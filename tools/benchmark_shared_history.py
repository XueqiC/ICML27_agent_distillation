"""Offline CPU micro-benchmark; synthetic conversations and random Gemma4 only.

Run: CUDA_VISIBLE_DEVICES='' python -m tools.benchmark_shared_history --output REPORT.json
Timings include layout/mask construction and scorer overhead; no GPU extrapolation.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import socket
import statistics
import sys
import time


def synthetic_conversation(seed, turns=20):
    rng = random.Random(seed)
    history, rows = [2] + [rng.randrange(10, 100) for _ in range(128)], []
    for _ in range(turns):
        native = history + [7] + [rng.randrange(10, 100) for _ in range(rng.randrange(8, 20))] + [8]
        reply = [rng.randrange(10, 100) for _ in range(rng.randrange(3, 12))] + [106]
        prompt = native + [3, 4, 5]
        rows.append({"packing": "turn", "input_ids": prompt + reply,
            "labels": [-100] * len(prompt) + reply, "generation_start": len(native),
            "prompt_tokens": len(prompt)})
        history = native + reply + [9]
    return {"packing": "turn", "turns": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or args.repeats < 1:
        raise ValueError("Set CUDA_VISIBLE_DEVICES='' and use positive repeats")
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    os.environ.pop("OPENAI_API_KEY", None)

    def forbidden(*args, **kwargs):
        raise RuntimeError("Benchmark is offline")

    for name in ("connect", "connect_ex", "sendto"):
        setattr(socket.socket, name, forbidden)
    socket.create_connection = socket.getaddrinfo = forbidden
    import torch
    import transformers
    from transformers import Gemma4UnifiedConfig, Gemma4UnifiedTextConfig, Gemma4UnifiedForConditionalGeneration
    from baselines.common.fsdp_train import batch_logps, dpo_objective
    from baselines.common.shared_history import pack_shared_history, attention_masks
    torch.set_num_threads(1)
    torch.manual_seed(42)
    text = Gemma4UnifiedTextConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        global_head_dim=8, max_position_embeddings=32768,
        layer_types=["sliding_attention", "full_attention"], sliding_window=7)
    config = Gemma4UnifiedConfig(text_config=text, vision_config=None, audio_config=None,
                                image_token_id=120, audio_token_id=121, video_token_id=122)
    config._attn_implementation = "sdpa"
    model = Gemma4UnifiedForConditionalGeneration(config).eval()
    from copy import deepcopy
    reference = deepcopy(model).requires_grad_(False)
    pair = {"chosen": synthetic_conversation(1), "rejected": synthetic_conversation(2)}
    records = []
    for operation in ("score_one_side", "dpo_pair_two_pass", "ipr_pair_two_pass"):
        for scoring in ("per_turn", "shared_history"):
            tokens, calls = [], []

            def count_forward(module, args, kwargs):
                tokens.append(kwargs["input_ids"].numel())
                calls.append(1)

            handles = [candidate.register_forward_pre_hook(count_forward, with_kwargs=True)
                       for candidate in (model, reference)]
            times = []
            for repeat in range(args.repeats + 1):
                model.zero_grad(set_to_none=True)
                tokens.clear()
                calls.clear()
                start = time.perf_counter()
                if operation == "score_one_side":
                    with torch.no_grad():
                        value = batch_logps(model, [pair["chosen"]], "cpu", scoring=scoring)[0]
                else:
                    value, _ = dpo_objective(model, reference, pair, "cpu", beta=.1,
                        nll_weight=float(operation.startswith("ipr")), backward_scale=1., scoring=scoring)
                elapsed = time.perf_counter() - start
                if repeat:
                    times.append(elapsed)
            for handle in handles:
                handle.remove()
            records.append({"operation": operation, "scoring": scoring, "forward_tokens": sum(tokens),
                "forward_calls": len(calls), "median_wall_seconds": statistics.median(times),
                "wall_seconds": times, "value": float(value.detach())})
    long_row = synthetic_conversation(3)
    # Native prefix extension for the mask-only allocation check.
    for turn in long_row["turns"]:
        turn["input_ids"] = [2] * 30000 + turn["input_ids"]
        turn["labels"] = [-100] * 30000 + turn["labels"]
        turn["generation_start"] += 30000
        turn["prompt_tokens"] += 30000
    packed = pack_shared_history(long_row)
    positions, masks = attention_masks([packed], text, "cpu", backend="flex_attention")
    metadata_bytes = sum(v.numel() * v.element_size() for mask in masks.values()
                         for v in vars(mask).values() if isinstance(v, torch.Tensor))
    report = {"torch": torch.__version__, "transformers": transformers.__version__, "device": "cpu",
        "python": sys.version.split()[0], "model_seed": 42, "conversation_seeds": [1, 2],
        "data_source": "synthetic token conversations; no task or model checkpoint data",
        "text_config": text.to_dict(),
        "code_sha256": {name: hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in (
            "baselines/common/shared_history.py", "baselines/common/fsdp_train.py",
            "train/dataset.py", "tools/benchmark_shared_history.py")},
        "dtype": "float32", "threads": 1, "turns_per_side": 20, "repeats": args.repeats,
        "attention_for_timing": "sdpa_cpu_dense_test_oracle", "records": records,
        "mask_check": {"packed_tokens": len(packed["input_ids"]), "block_metadata_bytes": metadata_bytes,
            "position_and_branch_bytes": positions.numel() * positions.element_size() * 2,
            "dense_gpu_mask": False},
        "scope": "Synthetic CPU engineering benchmark; no GPU/FSDP throughput or scientific claim"}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
