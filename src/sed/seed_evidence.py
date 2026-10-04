"""Validate environment evidence separately from model-visible training fields."""
import json
from pathlib import Path

from sed.storage import BlobStore, file_hash


def environment_won(directory, episode):
    if not (episode.get("complete") is True and episode.get("status") == "ok"
            and episode.get("success") is True and episode.get("steps")):
        return False
    final = episode["steps"][-1]
    verification = BlobStore(Path(directory) / "blobs").get(final["verification_ref"])
    return (verification.get("fact_source") == "real_environment"
            and verification.get("executable") is True
            and verification.get("audit", {}).get("won") is True
            and final.get("environment_steps") == 1)


def artifact_hashes(directory):
    directory = Path(directory)
    return {str(p.relative_to(directory)): file_hash(p) for p in sorted(directory.rglob("*"))
            if p.is_file() and not p.name.endswith(".lock")}


def checked_episode(root, receipt):
    directory = (Path(root) / receipt["episode_dir"]).resolve()
    if not directory.is_relative_to(Path(root).resolve()):
        raise ValueError("Episode path escapes collection")
    if artifact_hashes(directory) != receipt["episode_hashes"]:
        raise ValueError("Episode evidence changed")
    episode = json.loads((directory / "episode.json").read_text())
    if not environment_won(directory, episode):
        raise ValueError("Seed lacks a real environment win")
    return episode
