"""Publish every SAD budget crossing as an independently evaluable snapshot."""
import os
from pathlib import Path

from baselines.sad.data import DENOMINATORS
from sed.storage import digest, fsync_dir, write_json


def publisher(output, base_snapshot, provenance, plan, *, rank=0, export=None, barrier=None):
    from baselines.common.fsdp_train import export_checkpoint
    if barrier is None:
        import torch.distributed as dist
        barrier = lambda: dist.barrier() if dist.is_initialized() else None
    export = export or export_checkpoint
    output = Path(output)
    pending = list(zip(DENOMINATORS, plan["checkpoint_tokens"]))
    models = []

    def publish(model, state):
        while pending and state["processed_token_positions"] >= pending[0][1]:
            denominator, threshold = pending.pop(0)
            name = f"sad-1of{denominator}"
            staging, final = output / ("." + name + ".partial"), output / name
            saved = export(model, staging, base_snapshot, rank=rank)
            barrier()
            if rank == 0:
                write_json(staging / "training_state.json", {**state, **saved,
                    "provenance_sha256": digest(provenance), "final": True,
                    "budget_complete": denominator == 1, "checkpoint_fraction": f"1/{denominator}",
                    "checkpoint_threshold": threshold,
                    "threshold_overshoot": state["processed_token_positions"] - threshold})
                os.rename(staging, final)
                fsync_dir(output)
                # Retried training must not hide an already published earlier attempt.
                model_name = name + "-" + output.parent.name if output.parent.name.startswith("attempt-") else name
                models.append({"name": model_name, "checkpoint": str(final.resolve()),
                               "fraction": f"1/{denominator}", "processed_token_positions": state["processed_token_positions"]})
                write_json(output / "models.json", models)
                # Also publish immediately at pipeline root, even if a later update fails.
                if output.parent.parent.parent.name == "stages":
                    from baselines.eto.pipeline import publish_models
                    publish_models(output.parent.parent.parent.parent, models)
            barrier()
            # export_checkpoint temporarily enables caching for the saved config.
            if hasattr(model, "config"):
                model.config.get_text_config().use_cache = False
    return publish
