"""Teacher abstraction -> global functional clustering -> consolidation."""
import json
from pathlib import Path

from baselines.agentdistill.collection import checked_pool
from baselines.agentdistill.prompts import CONSTRUCTION, LANGUAGE
from baselines.agentdistill.sandbox import Rejected, validate
from baselines.agentdistill.teacher import read
from sed.provenance import write_provenance
from sed.storage import digest, write_json
from teachers.stage import teacher_stage, count


def parse_clusters(raw, ids):
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"clusters"} or not isinstance(value["clusters"], list):
        raise Rejected("Clustering schema invalid")
    clusters = value["clusters"]
    seen, names = [], set()
    for cluster in clusters:
        if (not isinstance(cluster, dict) or set(cluster) != {"name", "members"}
                or type(cluster["name"]) is not str or not cluster["name"] or cluster["name"] in names
                or not isinstance(cluster["members"], list) or not cluster["members"]
                or any(type(i) is not str for i in cluster["members"])):
            raise Rejected("Clustering membership invalid")
        seen.extend(cluster["members"])
        names.add(cluster["name"])
    if len(seen) != len(set(seen)) or set(seen) != set(ids):
        raise Rejected("Clustering must partition every accepted abstraction exactly once")
    return clusters


@teacher_stage("output", name="agentdistill.abstract/cluster/consolidate")
def build(kind, source, output, tasks, budget, config):
    output, source = Path(output), Path(source)
    output.mkdir(parents=True, exist_ok=False)
    task_map = {t["task_id"]: t for t in tasks}
    rows, rejected = [], []
    if kind == "abstract":
        for row in checked_pool(source / "pool.json", tasks):
            raw = budget.request([kind, row["id"]], kind, task_map[row["task_id"]],
                CONSTRUCTION[kind] + "\n" + LANGUAGE + "\nInput MCP:\n" + json.dumps(row["tool"]))
            try:
                tool = json.loads(raw)
                validate(tool)
                rows.append({**row, "tool": tool, "raw_tool_sha256": digest(row["tool"])})
            except (ValueError, TypeError, KeyError):
                count("rejections")
                rejected.append({"id": row["id"], "reason": "invalid_abstraction", "raw": raw})
        write_json(output / "abstracted.json", rows)
    elif kind == "cluster":
        rows = read(source / "abstracted.json")
        # Only code and anonymous source IDs enter this prompt, not lineage/audit.
        inputs = [{"id": row["id"], "tool": row["tool"]} for row in rows]
        if rows:
            raw = budget.request([kind, digest(inputs)], kind, tasks[0],
                CONSTRUCTION[kind] + "\nInput code:\n" + json.dumps(inputs))
            try:
                clusters = parse_clusters(raw, [r["id"] for r in rows])
            except (ValueError, TypeError, KeyError):
                count("rejections")
                write_json(output / "rejected.json", {"reason": "invalid_partition", "raw": raw})
                raise Rejected("Invalid global partition; retained, no automatic redraw") from None
        else:
            clusters = []
        write_json(output / "clusters.json", clusters)
        write_json(output / "abstracted.json", rows)
    elif kind == "consolidate":
        abstractions = {r["id"]: r for r in read(source / "abstracted.json")}
        clusters = read(source / "clusters.json")
        parse_clusters(json.dumps({"clusters": clusters}), list(abstractions))
        for cluster in clusters:
            inputs = [abstractions[i]["tool"] for i in cluster["members"]]
            parent = abstractions[cluster["members"][0]]["task_id"]
            raw = budget.request([kind, digest(cluster)], kind, task_map[parent],
                CONSTRUCTION[kind] + "\n" + LANGUAGE + "\nCluster:\n" + json.dumps({"name": cluster["name"], "tools": inputs}))
            try:
                tool = json.loads(raw)
                validate(tool)
                if tool["name"] in {r["tool"]["name"] for r in rows}:
                    raise Rejected("Consolidated names must be unique")
                rows.append({"cluster": cluster["name"], "members": cluster["members"], "tool": tool,
                    "sources": [{key: abstractions[i][key] for key in (
                        "id", "task_id", "parent_seed_ids", "run", "source_directory", "receipt_sha256", "raw_tool_sha256")}
                        for i in cluster["members"]],
                    "verification": "static_sandbox_validated; behavior_after_rewrite_unknown"})
            except (ValueError, TypeError, KeyError):
                count("rejections")
                rejected.append({"cluster": cluster, "reason": "invalid_consolidation", "raw": raw})
        if rejected:
            write_json(output / "rejections.json", rejected)
            raise Rejected("A cluster has no valid consolidated tool; box publication refused")
        write_json(output / "box.json", {"schema_version": 1, "label": "different interface",
            "training_exposure": 0, "tools": [r["tool"] for r in rows], "lineage": rows,
            "source_sha256": digest([abstractions, clusters]), "rejected": rejected})
    else:
        raise ValueError("Unknown MCP box stage")
    write_json(output / "rejections.json", rejected)
    write_json(output / "result.json", {"accepted": len(rows), "rejected": len(rejected), "training_exposure": 0})
    write_provenance(output, command="agentdistill_" + kind, config=config,
        inputs=list(source.glob("*.json")), details={"accepted": len(rows), "rejected": len(rejected)})


def load_box(path, sha256):
    from sed.storage import file_hash
    if file_hash(path) != sha256:
        raise ValueError("Published MCP box changed")
    box = read(path)
    if box["label"] != "different interface" or box["training_exposure"] != 0:
        raise ValueError("Unexpected MCP box identity")
    for tool in box["tools"]:
        validate(tool)
    if len({t["name"] for t in box["tools"]}) != len(box["tools"]):
        raise ValueError("Duplicate tool names")
    return box
