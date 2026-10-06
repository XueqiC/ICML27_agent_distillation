"""AgentEvol control: fresh Seed-SFT LoRA on checked D_m, matched passes."""
import argparse
from fractions import Fraction
import math
from pathlib import Path

from baselines.agentgym.data import training_examples
from baselines.common.dose_control import read, validate_init
from sed.config import load_config
from sed.manifest import load_frozen
from sed.provenance import write_provenance
from sed.seed_evidence import artifact_hashes, load_verified_collection
from sed.storage import digest, write_json
from train.dataset import load_tokenizer, mask_report
from train.modeling import attach_lora, load_student, snapshot_metadata
from train.sft import deterministic_seed, exposure_plan, train_loop


def matched_plan(examples, init):
    """Match realized episode passes, rounding only the final complete episode.

    Use Seed-SFT's seeded permutation, 512-token closure and loss unchanged.
    A fractional init endpoint may not be representable on the larger D_m;
    rounding is at most one episode and both requested/actual passes are logged.
    """
    recipe = init["recipe"]
    costs = [e["supervised_tokens"] for e in examples]
    options = dict(token_budget=recipe["tokens_per_update"], seed=recipe["seed"], packing="episode",
                   turn_counts=[e["turn_count"] for e in examples])
    # Recover the exact rational from the source state's integer exposure.
    state = init["training_state"]
    source_n = round(state["examples_seen"] / init["passes"])
    passes = Fraction(state["examples_seen"], source_n)
    count = math.ceil(passes * len(examples))
    full = exposure_plan(costs, passes=math.ceil(passes), **options)
    order = [i for batch in full for i in batch["indices"]][:count]
    return exposure_plan(costs, supervised_tokens=sum(costs[i] for i in order), **options)


def main(argv=None, *, device="cuda"):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "snapshot", "verified", "sft-data", "identity", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args(argv)
    config, identity = load_config(args.config), read(args.identity)
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    collection, _, _ = load_verified_collection(args.verified, manifest, freeze)
    init = validate_init(identity["init"]["path"], args.snapshot, args.verified, config,
                         collection_sha256=digest(collection))
    init["merge_policy"] = "seed_sft_fp32"
    if (identity.get("method") != "agentgym" or init != identity["init"]
            or artifact_hashes(args.snapshot) != identity["snapshot_artifact_sha256"]):
        raise ValueError("AgentEvol init/base identity changed")
    if Path(read(Path(args.sft_data).parent / "provenance.json")["details"]["verified"]).resolve() != Path(args.verified).resolve():
        raise ValueError("AgentEvol dataset belongs to a different verified collection")
    recipe = init["recipe"]
    deterministic_seed(recipe["seed"])
    tokenizer = load_tokenizer(args.snapshot)
    examples = training_examples(args.sft_data, config, tokenizer, packing="episode", max_length=recipe["max_length"])
    plan = matched_plan(examples, init)
    config["sft"] = {**recipe, "snapshot": args.snapshot, "base_snapshot": snapshot_metadata(args.snapshot),
        "verified": args.verified, "output": args.output, "passes": init["passes"], "supervised_tokens": None,
        "checkpoint_tokens": [], "plan_sha256": digest(plan), "examples_sha256": digest(examples),
        "template_sha256": digest(tokenizer.chat_template)}
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    provenance = write_provenance(output, command="agentgym_dose_sft", config=config,
        inputs=[args.config, args.identity, args.sft_data, Path(args.sft_data).parent / "provenance.json"],
        details={"control": "dose_matched_init", "init": init, "fresh_from_base": True,
            "requested_passes": init["passes"], "realized_passes": plan[-1]["passes"],
            "endpoint_rule": "ceil(requested passes * D_m episodes); no episode splitting"})
    write_json(output / "exposure_plan.json", plan)
    write_json(output / "mask_report.json", mask_report(examples, tokenizer))
    write_json(output / "example_sources.json", [e["label_provenance"] for e in examples])
    model = attach_lora(load_student(args.snapshot, device=device, dtype="bfloat16"),
        rank=recipe["rank"], alpha=recipe["alpha"], dropout=recipe["dropout"])
    checkpoints = train_loop(model, examples, plan, output, learning_rate=recipe["learning_rate"],
        tokenizer=tokenizer, identity_sha256=digest(provenance), gradient_clip=recipe["gradient_clip"])
    final = checkpoints[-1]
    write_json(output / "result.json", {**final, "checkpoint": final["path"],
        "requested_passes": init["passes"], "supervised_tokens_seen": final["supervised_tokens"],
        "adaptation": "seed_sft_lora", "fresh_from_base": True, "unmerged_serving": True})


if __name__ == "__main__":
    main()
