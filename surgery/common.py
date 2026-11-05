"""Immutable artifacts, provenance, and component-prefix path relocation."""
from pathlib import Path
import json

from repair.p0_pairs import provenance as p0_provenance, ROOT
from sed.storage import digest, file_hash, write_json


def read(path):
    return json.loads(Path(path).read_text())


def seal(value):
    value = {k: v for k, v in value.items() if k != 'sha256'}
    return dict(value, sha256=digest(value))


def checked(value, schema=None):
    if value.get('sha256') != seal(value)['sha256'] or (schema and value.get('schema') != schema):
        raise ValueError('Frozen artifact identity changed')
    return value


def publish(path, value):
    value = seal(value)
    write_json(path, value, exclusive=True)
    return value


def provenance(config=None):
    value = p0_provenance(config)
    paths = sorted((ROOT / 'surgery').glob('*.py')) + [ROOT / 'tools/p1_surgery.py']
    value['code_sha256'].update({str(p.relative_to(ROOT)): file_hash(p) for p in paths if p.exists()})
    return value


def output_guard(path, protected=()):
    path = Path(path).resolve()
    forbidden = [Path('/home/xueqi/hq/projects/ICML27_v0_runs'),
                 ROOT / 'reports/p0_tooling', ROOT / 'reports/p0_runs', *map(Path, protected)]
    if any(path == p.resolve() or path.is_relative_to(p.resolve()) for p in forbidden):
        raise ValueError('P1 output must be outside immutable input/P0/V0 directories')
    return path


class Relocation:
    """Small port of train-tree repair/v2_inputs.Relocation's path contract.

    One lexical OLD component-prefix replacement, nonoverlapping rules, no
    cascading, unchanged relative tail, resolved destination confinement.
    Frozen JSON/hashes are never rewritten. Only path consumers call this.
    """
    def __init__(self, values=()):
        self.rules = []
        for value in values:
            old, sep, new = value.partition('=')
            old, new = Path(old), Path(new)
            if (not sep or not old.is_absolute() or not new.is_absolute()
                    or '..' in old.parts or '..' in new.parts
                    or any(old.is_relative_to(a) or a.is_relative_to(old) for a, _ in self.rules)
                    or any(p.exists() and not p.is_dir() for p in (old, new))):
                raise ValueError('Relocation needs nonoverlapping absolute directory OLD=NEW prefixes')
            self.rules.append((old, new))
        self.boundaries = {new: new.resolve() for _, new in self.rules}

    def path(self, value, *, root=None):
        path = Path(value)
        if path.is_absolute():
            for old, new in self.rules:
                if path.is_relative_to(old):
                    tail = path.relative_to(old)
                    if '..' in tail.parts:
                        raise ValueError('Relocation relative-tail escape')
                    target = (new / tail).resolve()
                    if not target.is_relative_to(self.boundaries[new]):
                        raise ValueError('Relocation symlink escape')
                    return target
        path = (Path(root) / path if root is not None and not path.is_absolute() else path)
        resolved = path.resolve()
        for _, new in self.rules:
            if path.is_absolute() and path.is_relative_to(new) and not resolved.is_relative_to(self.boundaries[new]):
                raise ValueError('Relocation symlink escape')
        return resolved

    def inputs(self, inputs):
        for path, sha in inputs.items():
            if file_hash(self.path(path)) != sha:
                raise ValueError('Relocated source bytes changed')
