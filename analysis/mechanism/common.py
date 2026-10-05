"""Analysis IO, frozen-task authorization, immutable attempts and provenance."""
from copy import deepcopy
import json
from pathlib import Path

from analysis.mechanism.goals import refuse_sealed
from sed.manifest import authorize_task, load_frozen, resolve_task, verify_task_files
from sed.provenance import code_hashes, safe_config, versions, write_provenance
from sed.storage import BlobStore, digest, file_hash, write_json

ROOT = Path(__file__).resolve().parents[2]
PLAN = ROOT / "docs/MECHANISM_PLAN.md"


def read_json(path):
    return json.loads(Path(path).read_text())


def authorize_episode(episode):
    refuse_sealed(episode["task_id"])
    authorize_task(episode, "diagnosis")


def episode_path(path):
    path = Path(path).resolve()
    return path / "episode.json" if path.is_dir() else path


def load_episode(path):
    path = episode_path(path)
    episode = read_json(path)
    authorize_episode(episode)
    return episode


def audit_states(path, episode):
    authorize_episode(episode)
    store = BlobStore(episode_path(path).parent / "blobs")
    result = []
    for step in [episode["initial"], *episode["steps"]]:
        audit = store.get(step["verification_ref"])["audit"]
        if digest(audit) != step["audit_sha256"]:
            raise ValueError("Audit differs from episode evidence")
        result.append(audit)
    return result


def outcome(episode, audits):
    if not episode.get("complete") or episode.get("status") != "ok":
        return "unknown"
    won = audits[-1].get("won", "unknown")
    if type(won) is not bool or episode.get("success") is not won:
        raise ValueError("Episode success disagrees with real environment audit")
    return won


def frozen_episode(path, config, *, check_files=False, frozen=None):
    episode = load_episode(path)
    manifest, freeze = frozen or load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    task = resolve_task(manifest, freeze, episode["task_id"])
    authorize_task(task, "diagnosis")
    if episode["split"] != task["split"] or episode["game_sha256"] != task["game_sha256"]:
        raise ValueError("Episode differs from frozen task")
    if check_files:
        verify_task_files(config["environment"]["data_root"], task)
    return episode, task


def output_guard(output, *, inputs=(), config=None):
    """Refuse data, source runs, symlinks into them, and nested evaluation writes."""
    output = Path(output).resolve()
    protected = [ROOT / "data", ROOT / "third_party"]
    if config:
        protected.extend(Path(config["paths"][k]) for k in ("raw", "verified"))
        if config["environment"].get("data_root"):
            protected.append(Path(config["environment"]["data_root"]))
    for directory in protected:
        if output.is_relative_to(directory.resolve()):
            raise ValueError("Analysis output cannot be in data/ or third_party/")
    for source in inputs:
        source = Path(source).resolve()
        if source.is_file() and source.name == "episode.json":
            source = source.parent
        if output == source or output.is_relative_to(source):
            raise ValueError("Analysis output cannot be inside an input run")
    for ancestor in [output, *output.parents]:
        if (ancestor / "episode.json").exists():
            raise ValueError("Never write into an existing episode directory")
        provenance = ancestor / "provenance.json"
        if provenance.exists() and not read_json(provenance).get("command", "").startswith("mechanism_"):
            raise ValueError("Never write into an existing non-mechanism run")
    return output


def analysis_inputs(config, extra=()):
    return [PLAN, config["paths"]["manifest"], config["paths"]["split_freeze"], *extra]


def new_output(output, *, config, command, inputs=(), details=None):
    output = output_guard(output, inputs=inputs, config=config)
    output.mkdir(parents=True, exist_ok=False)
    write_provenance(output, command="mechanism_" + command, config=config,
                     inputs=analysis_inputs(config, inputs), details=details)
    return output


def workspace(output, *, config, command, identity, inputs=(), details=None):
    """Only an explicitly owned workspace resumes; individual attempts never do."""
    output = output_guard(output, inputs=inputs, config=config)
    identity = {"schema_version": 1, "command": command, "settings": identity,
                "config": safe_config(config), "code_sha256": code_hashes(ROOT), "dependencies": versions(),
                "inputs": {str(Path(p).resolve()): file_hash(p) for p in analysis_inputs(config, inputs)}}
    # Server IDs/ports change on a restart. Their full provenance is per attempt.
    identity = deepcopy(identity)
    identity["config"].pop("server_provenance", None)
    identity["config"].pop("lora_equivalence", None)
    if output.exists():
        if not (output / "identity.json").is_file():
            raise ValueError("Resume identity changed; use a new analysis workspace")
        saved = read_json(output / "identity.json")
        if saved != identity:
            if not compatible_identity(saved, identity):
                raise ValueError("Resume identity changed; use a new analysis workspace")
            # Preserve the original identity and record the exact engineering
            # transition. Individual resumed attempts retain current provenance.
            proof = output/"resume_compatibility"/(digest(identity) + ".json")
            try:
                write_json(proof, {"original_identity": saved, "execution_identity": identity}, exclusive=True)
            except FileExistsError:
                if read_json(proof) != {"original_identity": saved, "execution_identity": identity}:
                    raise ValueError("Resume compatibility evidence changed") from None
    else:
        new_output(output, config=config, command=command, inputs=inputs, details=details)
        write_json(output / "identity.json", identity, exclusive=True)
    return output


def compatible_identity(saved, current):
    """Only the explicitly pinned Stage 4c engineering transition may resume."""
    if saved.get("command") not in {"launcher", "fce", "interventions"}:
        return False
    registry = read_json(ROOT/"analysis/mechanism/resume_compatibility.json")
    if {"from": digest(saved.get("code_sha256")), "to": digest(current.get("code_sha256"))} not in registry["transitions"]:
        return False
    def semantic(identity):
        identity = deepcopy(identity)
        identity.pop("code_sha256")
        # Only checkout-local input locations move. Their exact content hashes,
        # all config/settings, external data paths and dependencies still match.
        plans = [p for p in identity["inputs"] if p.endswith("/docs/MECHANISM_PLAN.md")]
        if len(plans) == 1:
            checkout = str(Path(plans[0]).parents[1]) + "/"
            identity["inputs"] = {("checkout:"+p[len(checkout):] if p.startswith(checkout) else p): sha
                                  for p, sha in identity["inputs"].items()}
        return identity
    return semantic(saved) == semantic(current)


def budget_directory(requested, own):
    path = output_guard(requested or own)
    proof = path / "provenance.json"
    if not proof.is_file() or not read_json(proof).get("command", "").startswith("mechanism_"):
        raise ValueError("Shared budget directory must be an owned mechanism workspace")
    return path


def attempt_path(root, name):
    root = Path(root) / name
    index = 0
    while (root / f"attempt-{index:03d}").exists():
        index += 1
    return root / f"attempt-{index:03d}"


def completed_branch(root):
    """Validate and reuse only sealed receipts; interrupted attempts remain intact."""
    from sed.seed_evidence import artifact_hashes
    for receipt in sorted(Path(root).glob("attempt-*/receipt.json")):
        row = read_json(receipt)
        episode_dir = receipt.parent / "episode"
        if artifact_hashes(episode_dir) != row["episode_hashes"]:
            raise ValueError("Completed branch artifacts changed")
        episode = load_episode(episode_dir)
        if episode.get("complete") and episode.get("status") == "ok":
            return row
    return None


def seal_branch(attempt, episode):
    from sed.seed_evidence import artifact_hashes
    directory = Path(attempt) / "episode"
    success = outcome(episode, audit_states(directory, episode))
    row = {"task_id": episode["task_id"], "success": success,
           "episode_path": str((directory / "episode.json").resolve()),
           "branch": episode["branch"], "episode_hashes": artifact_hashes(directory)}
    write_json(Path(attempt) / "receipt.json", row, exclusive=True)
    return row


def gap_episodes(path, config):
    """Consume the frozen taxonomy snapshot, never a changing in-progress run."""
    report = read_json(path)
    if report["G"].get("size") is None:
        raise ValueError("A teacher comparison is required to define G")
    ids = report["G"]["task_ids"]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate gap tasks")
    rows = {r["task_id"]: r for r in report["runs"]["student"]["episodes"]}
    teachers = {r["task_id"] for r in report["runs"].get("teacher", {}).get("episodes", []) if r["success"] is True}
    if set(ids) != {key for key, row in rows.items() if row["success"] is False and key in teachers}:
        raise ValueError("Gap set differs from the recorded teacher/student outcomes")
    results = []
    frozen = load_frozen(config["paths"]["manifest"], config["paths"]["split_freeze"])
    for task_id in ids:
        refuse_sealed(task_id)
        row = rows[task_id]
        parent = Path(row["episode_path"])
        if file_hash(parent) != row["episode_sha256"]:
            raise ValueError("Gap parent changed since taxonomy snapshot")
        episode, task = frozen_episode(parent, config, frozen=frozen)
        if row["success"] is not False or outcome(episode, audit_states(parent, episode)) is not False or task["split"] != "dev":
            raise ValueError("Mechanism gap population requires failed dev episodes")
        results.append((row, episode, task))
    return report, results
