"""Parse ALFWorld goal IDs and audit-only typed PDDL facts."""
import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import PurePosixPath
import re


def refuse_sealed(task_id):
    parts = PurePosixPath(task_id).parts
    if "valid_unseen" in parts or "test" in parts:
        raise ValueError("Sealed tasks are forbidden in mechanism analysis")
    if not parts or ".." in parts or PurePosixPath(task_id).is_absolute():
        raise ValueError("Expected a relative task ID")


def normal_type(value):
    return re.sub(r"[^a-z0-9]", "", value.lower()).removesuffix("type")


@dataclass(frozen=True)
class Goal:
    task_type: str
    object_type: str
    movable_receptacle: str | None
    target_receptacle: str

    @property
    def base_object(self):
        return normal_type(self.object_type).removesuffix("sliced")

    @property
    def required(self):
        result = []
        for verb, predicate in (("heat", "ishot"), ("cool", "iscool"), ("clean", "isclean")):
            if f"_{verb}_" in self.task_type:
                result.append(predicate)
        if normal_type(self.object_type).endswith("sliced"):
            result.append("issliced")
        return result


def parse_goal(task_id):
    refuse_sealed(task_id)
    group = next((p for p in PurePosixPath(task_id).parts if p.startswith(("pick_", "look_at_obj"))), None)
    fields = group.split("-") if group else []
    if len(fields) < 5 or not all(fields[:4]):
        raise ValueError("Task ID lacks ALFWorld type/object/movable/target fields")
    return Goal(fields[0], fields[1], None if fields[2] == "None" else fields[2], fields[3])


def parse_fact(text):
    """TextWorld prints predicate(id: sort, id: sort); ignore only sort annotations."""
    match = re.fullmatch(r"\s*([\w]+)\((.*)\)\s*", text)
    if not match:
        raise ValueError("Unrecognized PDDL fact representation")
    args = tuple(a.strip().split(":", 1)[0].strip().lower()
                 for a in match[2].split(",") if a.strip())
    return match[1].lower(), args


class Facts:
    """Audit data. Never pass this object (or its attributes) to a policy."""
    def __init__(self, facts):
        self.rows = [(parse_fact(text), text) for text in facts]
        self.predicates = {}
        self.types = {}
        for (name, args), _ in self.rows:
            self.predicates.setdefault(name, set()).add(args)
            if name in {"objecttype", "receptacletype"} and len(args) == 2:
                self.types[args[0]] = normal_type(args[1])

    def get(self, name):
        return self.predicates.get(name, set())

    def held(self):
        return {args[-1] for args in self.get("holds")}

    def has(self, predicate, obj):
        return any(obj in args for args in self.get(predicate))

    def evidence(self, names):
        return [text for (name, _), text in self.rows if name in names]


def object_types(facts):
    return Facts(facts).types


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--facts", help="JSON fact list or verification blob; read only")
    args = parser.parse_args(argv)
    result = {"goal": asdict(parse_goal(args.task_id))}
    if args.facts:
        from pathlib import Path
        value = json.loads(Path(args.facts).read_text())
        result["object_types"] = object_types(value if isinstance(value, list) else value["audit"]["internal_facts"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
