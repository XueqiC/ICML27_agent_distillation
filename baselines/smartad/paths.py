"""Explicit filesystem relocation, separate from published pool identities."""
import os
from pathlib import Path

from sed.seed_evidence import artifact_hashes


class PathResolver:
    def __init__(self, entries=()):
        self.pairs = []
        for entry in entries:
            old, separator, new = entry.partition("=")
            if not separator or not old or not new or not all(Path(p).is_absolute() for p in (old, new)):
                raise ValueError("--path-map requires absolute OLD=NEW paths")
            pair = tuple(Path(os.path.normpath(p)) for p in (old, new))
            # Disjoint endpoints make both lookup and published-path recovery
            # unambiguous, and prevent accidental chained/double relocation.
            endpoints = [p for previous in self.pairs for p in previous]
            for endpoint in pair:
                if any(endpoint.is_relative_to(p) or p.is_relative_to(endpoint) for p in endpoints):
                    raise ValueError("--path-map has duplicate or overlapping prefixes")
                endpoints.append(endpoint)
            self.pairs.append(pair)

    @property
    def disclosure(self):
        return [{"old": str(old), "new": str(new)} for old, new in self.pairs]

    def published(self, path):
        """Recover a relocated root's published spelling for score identity."""
        path = Path(path)
        for old, new in self.pairs:
            if path.is_relative_to(new):
                return old / path.relative_to(new)
        return path

    def resolve(self, path, hashes=None):
        path = Path(path)
        for old, new in self.pairs:
            if path.is_relative_to(old):
                path = new / path.relative_to(old)
                break
        if not path.is_dir():
            raise ValueError(f"SmartAD evidence directory does not exist: {path}")
        if hashes is not None and artifact_hashes(path) != hashes:
            raise ValueError(f"SmartAD evidence changed: {path}")
        return path
