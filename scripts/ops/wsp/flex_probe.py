#!/usr/bin/env python3
"""flex_probe.py CELL_DIR — one GPU, a few minutes. For each candidate set of Flex attention tile options, run ONE
forward + backward of the SFT objective on the longest sealed example of the cell (LoRA attached as in training) and
report whether the Triton kernels compile and run on this GPU. Read-only; prints one PROBE line per candidate."""
import json, sys, time
import torch
from tools.wsp_onestep import load_cell
from baselines.common import fsdp_train as trainer

cell = load_cell(sys.argv[1])
row = max(cell["examples"], key=trainer.supervised_count)
print("gpu:", torch.cuda.get_device_name(0), "| example supervised tokens:", trainer.supervised_count(row), flush=True)
CANDS = [{"BLOCK_M": 32, "BLOCK_N": 32}, {"BLOCK_M": 16, "BLOCK_N": 16}, {"BLOCK_M": 16, "BLOCK_N": 32},
         {"BLOCK_M": 32, "BLOCK_N": 16}, {"BLOCK_M": 32, "BLOCK_N": 32, "num_stages": 1},
         {"BLOCK_M": 16, "BLOCK_N": 16, "num_stages": 1}]
for opt in CANDS:
    t0 = time.time(); model = None
    try:
        model = trainer.prepare_lora_model(cell["identity"]["snapshot"], device="cuda", adaptation="kang_lora",
                                           scoring="shared_history", flex_kernel_options=opt)
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, _ = trainer.sft_objective(model, row, "cuda", scoring="shared_history")
        loss.backward()
        torch.cuda.synchronize()
        print("PROBE OK  ", json.dumps(opt, sort_keys=True), "loss=%.6f" % float(loss), "peak_GB=%.1f" % (torch.cuda.max_memory_allocated() / 2**30),
              "sec=%.0f" % (time.time() - t0), flush=True)
    except Exception as e:
        print("PROBE FAIL", json.dumps(opt, sort_keys=True), type(e).__name__, str(e).replace("\n", " ")[-260:], flush=True)
    finally:
        del model
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
print("PROBE_DONE")
