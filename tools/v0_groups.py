"""CPU deterministic V0 k-medoids, episode weights, and tensor SNR ranking."""
import argparse
import json
import math
from pathlib import Path
import sys

if __package__ in (None, ""):
    root = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(root), str(root / "src")]

import numpy as np
from repair.artifacts import read_json
from repair.v2_inputs import ValidationError, require
from sed.storage import digest, file_hash, write_json


def k_medoids(sketches, episode_ids, raw_norms):
    """All tie breaks use episode_id, including initialization and merging."""
    require(len(set(episode_ids)) == len(episode_ids), "Duplicate episode ids")
    order = sorted(range(len(episode_ids)), key=lambda i: episode_ids[i])
    ids = [episode_ids[i] for i in order]
    x = np.asarray(sketches, dtype=np.float64)[order]
    norms = np.asarray(raw_norms, dtype=np.float64)[order]
    require(x.ndim == 2 and len(norms) == len(ids) and np.isfinite(x).all()
            and np.isfinite(norms).all() and (norms >= 0).all(), "Invalid clustering features")
    if not ids:
        return {"assignment": {}, "groups": [], "iterations": 0, "initial_medoids": []}
    lengths = np.linalg.norm(x, axis=1)
    require((lengths > 0).all() and (norms >= 1e-12).all(), "Zero gradients/sketches must be excluded")
    x = x / lengths[:, None]
    cosine = np.clip(x @ x.T, -1, 1)
    distance = 1 - cosine
    np.fill_diagonal(distance, 0)
    n = len(ids)
    k = 1 if n < 4 else min(6, n // 2)
    medoids = [min(range(n), key=lambda i: (-norms[i], ids[i]))]
    while len(medoids) < k:
        medoids.append(min((i for i in range(n) if i not in medoids),
                           key=lambda i: (-min(distance[i, j] for j in medoids), ids[i])))
    initial = [ids[i] for i in medoids]

    def assign(centres):
        groups = {m: [] for m in centres}
        for i in range(n):
            groups[min(centres, key=lambda m: (distance[i, m], ids[m]))].append(i)
        return {m: members for m, members in groups.items() if members}

    def centre(members):
        return min(members, key=lambda i: (float(distance[i, members].sum()), ids[i]))

    for iteration in range(1, 21):
        groups = assign(medoids)
        updated = sorted(centre(members) for members in groups.values())
        if updated == sorted(medoids):
            break
        medoids = updated
    groups = assign(medoids)
    while len(groups) > 1 and any(len(members) < 2 for members in groups.values()):
        small = min((m for m in groups if len(groups[m]) < 2), key=lambda m: ids[m])
        target = min((m for m in groups if m != small), key=lambda m: (distance[small, m], ids[m]))
        members = groups.pop(small) + groups.pop(target)
        groups[centre(members)] = sorted(members)
    result, assignment = [], {}
    for medoid, members in sorted(groups.items(), key=lambda pair: ids[pair[0]]):
        gid = f"group-{len(result):02d}"
        size = len(members)
        mean_cosine = float(cosine[np.ix_(members, members)][np.triu_indices(size, 1)].mean()) if size > 1 else 1.0
        farthest = min((i for i in members if i != medoid),
                       key=lambda i: (-distance[i, medoid], ids[i]), default=None)
        result.append({"group_id": gid, "medoid": ids[medoid],
            "farthest_member": ids[farthest] if farthest is not None else None,
            "members": [ids[i] for i in sorted(members)], "size": size,
            "mean_pairwise_cosine": mean_cosine, "consistency": mean_cosine * size})
        assignment.update({ids[i]: gid for i in members})
    return {"assignment": assignment, "groups": result, "iterations": iteration, "initial_medoids": initial}


def episode_weights(groups, entries):
    consistency = {g["group_id"]: g["consistency"] for g in groups["groups"]}
    weights = {}
    rg = [e["episode_id"] for e in entries if e["bucket"] in ("R", "G")]
    for entry in entries:
        eid = entry["episode_id"]
        group = groups["assignment"].get(eid)
        weights[eid] = consistency[group] if eid in rg and group is not None else 1.0
    # Signed consistency is not necessarily a probability. Do not quietly
    # clip/shift it and claim that the preregistered weighting was implemented.
    require(all(math.isfinite(weights[eid]) and weights[eid] >= 0 for eid in rg),
            "Negative group consistency cannot define sampling probabilities; weighting protocol needs a decision")
    if rg:
        mean = sum(weights[eid] for eid in rg) / len(rg)
        require(mean > 0, "All R/G consistency weights are zero; weighting protocol needs a decision")
        for eid in rg:
            weights[eid] /= mean
    for bucket in ("R", "G"):
        selected = [weights[e["episode_id"]] for e in entries if e["bucket"] == bucket]
        require(not selected or sum(selected) > 0, "A nonempty bucket has no positive episode weight")
    return weights


def rank_modules(modules):
    require(isinstance(modules, dict) and modules, "Empty module statistics")
    require(all(isinstance(n, str) and any(p in ("lora_A", "lora_B") for p in n.split("."))
                and type(v.get("snr")) in (int, float) and math.isfinite(v["snr"]) and v["snr"] >= 0
                for n, v in modules.items()), "Invalid module SNR")
    names = sorted(modules, key=lambda n: (-modules[n]["snr"], n))
    count = math.ceil(len(names) / 4)
    return {"selected": names[:count], "fraction": 0.25,
            "ranking": [{"name": name, "rank": i + 1, **modules[name]} for i, name in enumerate(names)]}


def run(features, output):
    features, output = Path(features).resolve(), Path(output).resolve()
    require(not output.exists() and not output.is_relative_to(features), "Output must be new and outside features")
    index = read_json(features / "index.json")
    require(index.get("schema") == "v0-gradient-features/1", "Invalid feature schema")
    for name in ("sketch.npy", "norms.json", "modules.json"):
        require(index["artifact_sha256"].get(name) == file_hash(features / name), "Feature artifact hash mismatch")
    ids, entries = index["episode_ids"], index["episodes"]
    require(ids == sorted(set(ids)) and [e["episode_id"] for e in entries] == ids, "Feature episode order mismatch")
    require(all(e["bucket"] in ("R", "G", "S", "D") for e in entries), "Invalid feature bucket")
    sketch = np.load(features / "sketch.npy", allow_pickle=False)
    require(sketch.shape == (len(ids), 2048) and sketch.dtype == np.float32 and np.isfinite(sketch).all(),
            "Expected finite N x 2048 float32 sketch")
    norms, modules = read_json(features / "norms.json"), read_json(features / "modules.json")
    require(set(norms) == set(ids), "Norm episode mismatch")
    valid, excluded = [], {}
    for i, eid in enumerate(ids):
        norm = norms[eid]["raw_l2_norm"]
        require(type(norm) in (int, float) and math.isfinite(norm) and norm >= 0
                and norms[eid]["ZERO_GRADIENT"] == (norm < 1e-12), "Invalid raw gradient norm/flag")
        if norm < 1e-12:
            excluded[eid] = "ZERO_GRADIENT"
        elif np.linalg.norm(sketch[i]) == 0:
            excluded[eid] = "ZERO_SKETCH"
        else:
            valid.append(i)
    groups = k_medoids(sketch[valid], [ids[i] for i in valid], [norms[ids[i]]["raw_l2_norm"] for i in valid])
    identity = {k: index[k] for k in ("dataset_sha256", "adapter_sha256", "parameter_names_sha256")}
    require(all(modules.get(k) == index[k] for k in ("dataset_sha256", "adapter_sha256")), "Module identity mismatch")
    require(sorted(modules["modules"]) == index["parameter_names"] and
            digest(index["parameter_names"]) == index["parameter_names_sha256"], "Module name hash mismatch")
    ranking = rank_modules(modules["modules"])
    common = {**identity, "features_index_sha256": file_hash(features / "index.json"),
              "code_sha256": file_hash(Path(__file__)), "numpy_version": np.__version__,
              "smoke_subset": len(ids) != index["available_episodes"]}
    output.mkdir(parents=True)
    write_json(output / "groups.json", {"schema": "v0-groups/1", **common, **groups, "excluded": excluded}, exclusive=True)
    write_json(output / "trainable_modules.json", {"schema": "v0-trainable-modules/1", **common, **ranking}, exclusive=True)
    try:
        weights = episode_weights(groups, entries)
    except ValidationError as exc:
        # Preserve negative results and the independently valid D-param output.
        write_json(output / "weights.json", {"schema": "v0-episode-weights/1", **common,
            "status": "invalid", "reason": str(exc)}, exclusive=True)
        raise
    write_json(output / "weights.json", {"schema": "v0-episode-weights/1", **common, "status": "valid", "weights": weights,
        "normalization": "R/G episode mean 1; S/D 1; excluded gradients neutral before normalization"}, exclusive=True)
    return groups


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    try:
        groups = run(args.features, args.output)
        print(json.dumps({"status": "complete", "groups": len(groups["groups"])}))
        return 0
    except ValidationError as exc:
        print(f"V0 grouping refused: {exc}", file=sys.stderr)
    except Exception as exc:
        print(f"V0 grouping failed ({type(exc).__name__}).", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
