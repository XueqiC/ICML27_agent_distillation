"""Tests cannot use CUDA, DNS, TCP/UDP, or the live OpenAI transport."""

import json
import os
from pathlib import Path
import socket

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pytest


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("Network/API access is forbidden in tests")
    for name in ("connect", "connect_ex", "sendto"):
        monkeypatch.setattr(socket.socket, name, forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "getaddrinfo", forbidden)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TEACHER_RATE_LIMIT_DIR", raising=False)
    monkeypatch.setenv("SED_SECRETS_FILE", str(tmp_path / "nonexistent.env"))
    monkeypatch.setenv("SED_HASH_CACHE_DIR", str(tmp_path / "hash-receipts"))
    try:
        from openai.resources.responses.responses import Responses
    except ImportError:
        pass
    else:
        monkeypatch.setattr(Responses, "create", forbidden)


class FakeBackend:
    def __init__(self, path, seed):
        self.position = 0
        self.closed = False
        self.win_at = json.loads(Path(path).read_text()).get("win_at", 99)

    def state(self):
        return {"feedback": f"Visible room at {self.position}",
                "admissible_commands": ["look", "go north"],
                "won": self.position >= self.win_at, "lost": False,
                "_facts": [f"hidden-position-{self.position}"],
                "extra.expert_plan": ["SECRET_PLAN"], "policy_commands": ["SECRET_POLICY"],
                "facts": ["SECRET_PDDL"], "new_oracle_field": {"leak": "SECRET_ORACLE"}}

    def reset(self):
        self.position = 0
        return self.state()

    def step(self, action):
        self.position += 1
        state = self.state()
        return state, float(state["won"]), state["won"]

    def close(self):
        self.closed = True


@pytest.fixture
def data_root(tmp_path):
    root = tmp_path / "dataset"
    for source in ("train", "valid_seen", "valid_unseen"):
        for group in ("room_a", "room_b"):
            for trial in ("trial_0", "trial_1"):
                directory = root / source / group / trial
                directory.mkdir(parents=True)
                (directory / "traj_data.json").write_text(json.dumps({"task_type": "pick_and_place_simple"}))
                (directory / "game.tw-pddl").write_text(json.dumps({"solvable": True}))
                (directory / "initial_state.pddl").write_text("PRIVATE_PDDL")
    return root


@pytest.fixture
def task():
    return {"task_id": "train/room_a/trial_0", "split": "train", "sealed": False,
            "task_type": "pick_and_place_simple", "instruction": "Follow the visible task"}


@pytest.fixture
def visible():
    return {"observation": "Find the cup", "admissible_commands": ["look", "go north"]}
