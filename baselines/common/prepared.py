"""CPU-sealed LoRA examples; GPU startup checks only bound top-level inputs.

Collections and returned examples are immutable by protocol. The seal is an
integrity record on trusted storage, not a signature against a malicious writer.
"""
import json
import os
from copy import deepcopy
from pathlib import Path

from sed.config import load_config
from sed.manifest import load_frozen
from sed.provenance import code_hashes
from sed.seed_evidence import artifact_hashes, load_verified_collection
from sed.startup_timing import timed
from sed.storage import atomic_write, canonical_bytes, digest, file_hash, write_json
from sed.validation_cache import anchors

ROOT = Path(__file__).resolve().parents[2]
METHODS = {"kang", "adftp", "smartad"}


class PreparedDatasetError(ValueError):
    pass


def refuse(reason):
    raise PreparedDatasetError(f"Prepared dataset refused: {reason}. Run the CPU --mode prepare stage "
                               "with the same inputs, recipe and output before training "
                               "(use a new output if the frozen pipeline identity changed).")


def read(path):
    return json.loads(Path(path).read_bytes())


def pipeline_identity(output, config, *, method=None, recipe=None, verified=None,
                      snapshot=None, gpus=None, current_code=None):
    try:
        identity = read(Path(output) / "identity.json")
        expected = {"config_sha256": digest(config),
                    "code_sha256": code_hashes(ROOT) if current_code is None else current_code}
        for key, value in (("method", method), ("recipe", recipe), ("gpus", gpus)):
            if value is not None:
                expected[key] = value
        for key, value in (("verified", verified), ("snapshot", snapshot)):
            if value is not None:
                expected[key] = str(Path(value).resolve())
        if identity["method"] not in METHODS or any(identity.get(k) != v for k, v in expected.items()):
            refuse("pipeline identity/config/recipe/code mismatch (different frozen artifact)")
        return identity
    except (OSError, KeyError, TypeError, ValueError) as error:
        if isinstance(error, PreparedDatasetError):
            raise
        refuse("missing or invalid pipeline identity")


def trainer_recipe(args):
    # Output/transport locations differ between preparation and train attempts.
    # Every other effective trainer option, including masks and seeds, is bound.
    return {k: v for k, v in vars(args).items() if k not in {"output", "pipeline_output"}}


def check_trainer_identity(identity, args):
    if hasattr(args, "objective"):
        method = ("smartad" if args.objective == "smartad" else
                  "adftp" if args.allow_empty_visible_replies else "kang")
        if identity["method"] != method:
            refuse("trainer method differs from pipeline identity")
    for key, value in identity["recipe"].items():
        option = "seed" if key == "training_seed" else key
        if hasattr(args, option) and getattr(args, option) != value:
            refuse("trainer recipe differs from pipeline identity")


def seal_path(output, identity):
    return Path(output).resolve() / "prepared" / digest(identity) / "seal.json"


def tokenizer_binding(hashes, chat_template):
    assets = {name: sha for name, sha in hashes.items()
              if any(word in name.lower() for word in ("token", "vocab", "merge", "template", "special"))}
    return {"artifacts": assets, "chat_template_sha256": digest(chat_template)}


def complete_bindings(bindings, config):
    """Include every view/parent/support, including parents of copied FTP views."""
    manifest, freeze = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    visited = set()
    while set(bindings) - visited:
        root = next(iter(set(bindings) - visited))
        visited.add(root)
        identity = read(Path(root) / "identity.json")
        for name in ("source_collection", "parent_collection", "support_view", "extension_source"):
            reference = identity.get(name)
            if reference:
                load_verified_collection((Path(root) / reference["path"]).resolve(), manifest, freeze)
    # Also bind selection-only top-level metadata, without encoding audit fields.
    for root, hashes in list(bindings.items()):
        if anchors(root) != hashes:
            refuse("collection changed during CPU preparation")
        bindings[root] = hashes = dict(hashes)
        path = Path(root) / "smartad_selection.json"
        if path.exists():
            hashes["smartad_selection.json"] = file_hash(path)


def write_prepared(args, data, bindings):
    config, collection, examples, chat_template = data
    raw_config = load_config(args.config)
    identity = pipeline_identity(args.pipeline_output, raw_config, verified=args.verified, snapshot=args.snapshot)
    check_trainer_identity(identity, args)
    if digest(collection) != identity["collection_sha256"]:
        refuse("validated collection differs from pipeline identity")
    hashes = artifact_hashes(args.snapshot)
    if hashes != identity["snapshot_artifact_sha256"]:
        refuse("snapshot differs from pipeline identity")
    target = seal_path(args.pipeline_output, identity)
    # Reload config at training time after checking its digest; do not duplicate
    # arbitrary config values (which could contain secrets) in the seal.
    dataset = {"adaptation": config["training"]["adaptation"], "collection": {k: collection[k] for k in
               ("K", "unique_instances", "task_groups")}, "chat_template": chat_template}
    examples_path = target.with_name("examples.json")
    atomic_write(examples_path, canonical_bytes(examples) + b"\n")
    inputs = [args.config, raw_config["paths"]["manifest"], raw_config["paths"]["split_freeze"],
              Path(args.pipeline_output) / "tasks.json"]
    seal = {"version": 1, "pipeline_identity_sha256": digest(identity),
            "recipe_config_sha256": digest([identity["recipe"], raw_config]),
            "trainer_recipe_sha256": digest(trainer_recipe(args)),
            "input_collections": bindings,
            "input_files": {str(Path(p).resolve()): file_hash(p) for p in inputs},
            "snapshot": identity["snapshot"], "snapshot_artifact_sha256": hashes,
            "tokenizer": tokenizer_binding(hashes, chat_template),
            "tokenizer_sha256": digest(tokenizer_binding(hashes, chat_template)),
            "code_sha256": identity["code_sha256"], "dataset": dataset,
            "encoded_examples_sha256": file_hash(examples_path), "examples_sha256": digest(examples),
            "examples_count": len(examples)}
    # Seal is published last. An interruption can never authorize partial bytes.
    write_json(target, {"seal": seal, "sha256": digest(seal)})
    return {"path": str(target), "sha256": file_hash(target),
            "encoded_examples_sha256": seal["encoded_examples_sha256"]}


@timed("seal_verification")
def verify_prepared(output, identity, config, *, args=None):
    try:
        target = seal_path(output, identity)
        envelope = read(target)
        seal = envelope["seal"]
        if (envelope["sha256"] != digest(seal) or seal["version"] != 1
                or seal["pipeline_identity_sha256"] != digest(identity)
                or seal["recipe_config_sha256"] != digest([identity["recipe"], config])
                or seal["code_sha256"] != identity["code_sha256"]
                or seal["code_sha256"] != code_hashes(ROOT)):
            refuse("seal identity/recipe/code mismatch")
        if args is not None and seal["trainer_recipe_sha256"] != digest(trainer_recipe(args)):
            refuse("trainer recipe mismatch")
        if args is not None:
            check_trainer_identity(identity, args)
        if identity["verified"] not in seal["input_collections"]:
            refuse("missing verified collection binding")
        for root, hashes in seal["input_collections"].items():
            if not {"manifest.json", "identity.json", "selection.json"} <= hashes.keys():
                refuse("incomplete collection binding")
            if any(file_hash(Path(root) / name) != sha for name, sha in hashes.items()):
                refuse("input collection manifest/identity/selection changed")
        if any(file_hash(path) != sha for path, sha in seal["input_files"].items()):
            refuse("config/split/task input changed")
        if (seal["snapshot"] != identity["snapshot"]
                or seal["snapshot_artifact_sha256"] != identity["snapshot_artifact_sha256"]
                or artifact_hashes(identity["snapshot"]) != seal["snapshot_artifact_sha256"]):
            refuse("snapshot artifact receipts changed")
        dataset = seal["dataset"]
        tokenizer = tokenizer_binding(seal["snapshot_artifact_sha256"], dataset["chat_template"])
        if seal["tokenizer"] != tokenizer or seal["tokenizer_sha256"] != digest(tokenizer):
            refuse("tokenizer/chat-template mismatch")
        examples_path = target.with_name("examples.json")
        if file_hash(examples_path) != seal["encoded_examples_sha256"]:
            refuse("encoded examples changed")
        examples = read(examples_path)
        if digest(examples) != seal["examples_sha256"] or len(examples) != seal["examples_count"]:
            refuse("encoded examples digest/count mismatch")
        reference = {"path": str(target), "sha256": file_hash(target),
                     "encoded_examples_sha256": seal["encoded_examples_sha256"]}
        effective_config = deepcopy(config)
        effective_config["training"]["adaptation"] = dataset["adaptation"]
        return (effective_config, dataset["collection"], examples, dataset["chat_template"]), reference
    except (OSError, KeyError, TypeError, ValueError) as error:
        if isinstance(error, PreparedDatasetError):
            raise
        refuse("missing, unreadable or invalid seal/input")


def load_prepared(args):
    if not getattr(args, "pipeline_output", None):
        refuse("missing --pipeline-output/seal")
    config = load_config(args.config)
    identity = pipeline_identity(args.pipeline_output, config, verified=args.verified, snapshot=args.snapshot,
                                 gpus=int(os.environ.get("WORLD_SIZE", "1")))
    if Path(args.base_snapshot).resolve() != Path(args.snapshot).resolve():
        refuse("base snapshot mismatch")
    return verify_prepared(args.pipeline_output, identity, config, args=args)
