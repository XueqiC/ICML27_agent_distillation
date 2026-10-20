"""Deterministic process shards sharing one full-TRAIN evidence collection."""

from copy import deepcopy
import json
from pathlib import Path

from sed.manifest import frozen_write, resolve_task, verify_task_files
from sed.provenance import code_hashes, safe_config, versions, write_provenance
from sed.runner import _environment
from sed.seed_selection import is_full_train, ordered_candidates, validate_selection
from sed.storage import digest, locked, write_json
from teachers.capped import SharedReservations, CollectionBudgetExceeded
from teachers.openai_teacher import Teacher, TeacherCallError, rate_limit_options
from teachers.stage import teacher_stage, transport_history, execution_attempt

MAX_CALLS = 160_000
MAX_INPUT_TOKENS = 700_000_000


def collection_identity(selection, manifest, freeze, config, *, attempts, shard_count,
                        max_calls, max_input_tokens):
    validate_selection(selection, manifest, freeze)
    if not is_full_train(selection) or attempts != 3:
        raise ValueError("Full TRAIN requires all_train_v1 and exactly 3 attempts per game")
    if type(shard_count) is not int or shard_count < 1:
        raise ValueError("Positive shard count required")
    if (type(max_calls) is not int or not 0 <= max_calls <= MAX_CALLS
            or type(max_input_tokens) is not int or not 0 <= max_input_tokens <= MAX_INPUT_TOKENS):
        raise ValueError("Full TRAIN caps cannot exceed 160000 calls / 700M reserved input tokens")
    limits = config["models"]["teacher"].get("rate_limit", {})
    # Ceiling = 90% of the provider headers read 2026-10-19 (5000 requests / 2,000,000 tokens per minute).
    if (not 0 < limits.get("requests_per_minute", 450) <= 4500
            or not 0 < limits.get("tokens_per_minute", 180_000) <= 1_800_000):
        raise ValueError("Full TRAIN rate limits cannot exceed 4500 rpm / 1800000 tpm")
    return {"selection": selection, "config": safe_config(config),
            "attempts_per_candidate": attempts, "shard_count": shard_count,
            "max_calls": max_calls, "max_input_tokens": max_input_tokens,
            "code_sha256": code_hashes(Path(__file__).resolve().parents[1]),
            "dependencies": versions()}


def check_output(output, identity, resume):
    output = Path(output)
    path = Path(output) / "identity.json"
    if path.exists():
        if not resume:
            raise ValueError("Collection exists; use --resume with the same frozen identity")
        if json.loads(path.read_text()) != identity:
            raise ValueError("Resume identity changed (selection/config/code/dependencies/shards/caps)")
        if (not (output / "teacher_budget/reservations.sqlite3").is_file()
                and any((output / "attempts").glob("*/started.json"))):
            raise ValueError("Missing reservation ledger; refusing to reset spent collection budget")
    elif Path(output).exists() and any(Path(output).iterdir()):
        # A new shard may have just created the lock file before initialization.
        if any(p.name != ".collection.lock" for p in Path(output).iterdir()):
            raise ValueError("Existing output has no collection identity")


def preflight(selection, manifest, freeze, config, output, *, attempts=3, shard_count=1,
              max_calls=MAX_CALLS, max_input_tokens=MAX_INPUT_TOKENS, resume=False):
    """Read-only validation of every selected game, before launching any worker."""
    identity = collection_identity(selection, manifest, freeze, config, attempts=attempts,
        shard_count=shard_count, max_calls=max_calls, max_input_tokens=max_input_tokens)
    check_output(output, identity, resume)
    for candidate in ordered_candidates(selection):
        verify_task_files(config["environment"]["data_root"],
                          resolve_task(manifest, freeze, candidate["task_id"]))
    return identity


@teacher_stage("output", name="teachers.full_train")
def collect_full_train(selection, manifest, freeze, config, output, *, attempts=3,
                       shard_index=0, shard_count=1, resume=False,
                       max_calls=MAX_CALLS, max_input_tokens=MAX_INPUT_TOKENS,
                       env_factory=None, teacher_factory=None, input_paths=()):
    from teachers.collect_seeds import collect_attempt

    config, output = deepcopy(config), Path(output).resolve()
    identity = collection_identity(selection, manifest, freeze, config, attempts=attempts,
        shard_count=shard_count, max_calls=max_calls, max_input_tokens=max_input_tokens)
    if type(shard_index) is not int or not 0 <= shard_index < shard_count:
        raise ValueError("Shard index must be in [0, shard_count)")
    with locked(output / ".collection.lock"):
        check_output(output, identity, resume)
        if not (output / "identity.json").exists():
            write_json(output / "identity.json", identity, exclusive=True)
        frozen_write(output / "selection.json", selection)
        if not (output / "provenance.json").exists():
            write_provenance(output, command="collect_full_train", config=config, inputs=input_paths,
                             details={"selection_sha256": digest(selection), "attempts": attempts,
                                      "shard_count": shard_count, "max_calls": max_calls,
                                      "max_input_tokens": max_input_tokens})

    budget = SharedReservations(output / "teacher_budget/reservations.sqlite3",
                                max_calls=max_calls, max_input_tokens=max_input_tokens)
    env_factory = env_factory or (lambda task: _environment(config, task))
    factory = teacher_factory

    def teacher(task, cache, ledger):
        spec = config["models"]["teacher"]
        policy = (factory(task, cache, ledger) if factory else Teacher(
            cache_dir=cache, usage_ledger=ledger, run_id="pending",
            provider_model_id=spec["provider_model_id"], sampling=spec["sampling"],
            max_attempts=spec["max_attempts"], timeout_seconds=spec["timeout_seconds"],
            secrets_file=config["paths"]["secrets_file"]))
        if not isinstance(policy, Teacher):
            raise ValueError("Full TRAIN requires Teacher (inject a stub transport for tests)")
        from teachers.rate_limit import RateLimiter
        options = rate_limit_options(spec, output / "data/teacher_rate_limit")
        policy.limiter = RateLimiter(options.pop("rate_limit_dir"), policy.provider_model_id,
                                     clock=policy.clock, sleep=policy.sleep, **options)
        policy.reservation_budget = budget
        return policy

    candidates = ordered_candidates(selection)
    order = {r["task_id"]: i for i, r in enumerate(candidates)}
    receipts = []

    def publish(status):
        with locked(output / ".collection.lock"):
            write_json(output / "shards" / f"{shard_index}.json",
                       {"shard_index": shard_index, "status": status, "attempts": receipts})
            shards = [json.loads(p.read_text()) for p in sorted((output / "shards").glob("*.json"))]
            all_receipts = sorted((r for shard in shards for r in shard["attempts"]),
                                  key=lambda r: (order[r["task_id"]], r["attempt"]))
            verified = [r for r in all_receipts if r["won"]]
            last = {r["task_id"]: r for r in all_receipts}
            failed = [tid for tid, r in last.items() if not r["won"] and r["attempt"] == attempts]
            done = len(verified) + len(failed) == selection["candidates"]
            accounting = budget.snapshot()
            status = ("transport_failed" if any(s["status"] == "transport_failed" for s in shards)
                      else "budget_exhausted" if accounting["stopped"] else "complete" if done
                      else "interrupted" if any(s["status"] == "interrupted" for s in shards)
                      else "collecting")
            value = {"schema_version": 1, "status": status, "K": len(verified),
                     "K_counts": "trusted_executed_trajectories", "candidates": selection["candidates"],
                     "selection_rule": selection["selection_rule"], "identity_sha256": digest(identity),
                     "selection_sha256": digest(selection), "trusted_trajectories": len(verified),
                     "unique_instances": len({r["task_id"] for r in verified}),
                     "task_groups": len({r["group_id"] for r in verified}),
                     "attempted_unique_instances": len(last),
                     "attempted_task_groups": len({r["group_id"] for r in all_receipts}),
                     "failed_candidates": failed, "attempts": all_receipts, "verified": verified,
                     "parent_seed_ids": [], "source": "external_frozen_train_pool",
                     "training_exposure": 0, "budget": accounting,
                     "transport_failure_evidence": transport_history(output / "attempts"),
                     "transport_failures": len(list((output / "attempts").rglob("transport_failure.json"))),
                     "rejections": sum(r["usage"].get("rejections", 0) for r in all_receipts)}
            write_json(output / "manifest.json", value)
            return value

    # Whole-shard ownership prevents concurrent resumptions from duplicating work;
    # the collection lock is held only for short initialization/publication steps.
    with locked(output / "shards" / f"{shard_index}.lock"):
        try:
            for index, candidate in enumerate(candidates[shard_index::shard_count]):
                task = resolve_task(manifest, freeze, candidate["task_id"])
                for number in range(1, attempts + 1):
                    attempt_dir = output / "attempts" / (digest(task["task_id"])[:20] + f"-{number}")
                    # Recover completed/uncertain attempts even after the budget stopped.
                    if budget.snapshot()["stopped"]:
                        retained = execution_attempt(attempt_dir, start="started.json")
                        episode_path = retained / "episode/episode.json"
                        episode = json.loads(episode_path.read_text()) if episode_path.exists() else {}
                        if not (retained / "receipt.json").exists() and not episode.get("complete"):
                            raise CollectionBudgetExceeded("Collection budget exhausted; stage NOT complete")
                    receipt = collect_attempt(task, number, task["task_id"], config, output,
                        env_factory=env_factory, teacher_factory=teacher, input_paths=input_paths)
                    receipts.append(receipt)
                    if receipt["won"]:
                        break
                # Receipts are durable immediately. Batch the large aggregate JSON;
                # --resume reconstructs it from receipts after an ungraceful exit.
                if index % 25 == 0:
                    publish("collecting")
            return publish("complete")
        except TeacherCallError:
            publish("transport_failed")
            raise
        except BaseException:
            publish("interrupted")
            raise
