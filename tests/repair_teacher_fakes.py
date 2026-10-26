"""Scripted stage responses; no SDK, model, network or hidden-state inference."""
import json

from repair.prompts import unpack_request
from teachers.openai_teacher import TeacherResult


DIAGNOSIS = {"expressed_plan": [{"subgoal": "Finish", "evidence_steps": [0],
    "interpretation_status": "hypothesis"}], "preservable_choices": [], "unmet_requirements": [],
    "candidate_joint_changes": [], "unknowns": ["diagnosis-only-marker"]}


class FakeTeacher:
    def __init__(self, responses):
        self.responses, self.requests = list(responses), []

    def generate(self, history, task, visible_context):
        assert history == []
        prompt = unpack_request(visible_context)
        self.requests.append(prompt)
        response = self.responses.pop(0)
        if callable(response):
            response = response(prompt)
        if isinstance(response, BaseException):
            raise response
        if isinstance(response, TeacherResult):
            return response
        return result(response)


def result(value, **metadata):
    return TeacherResult(value if isinstance(value, str) else json.dumps(value), None,
        {"usage": {"input_tokens": 20, "output_tokens": 30, "cached_input_tokens": 0},
         "response_status": "completed", "provider_model_id": "fake-alias",
         "returned_model_id": "fake-returned-model", "timestamp": "fake-time",
         "cache_hit": False, **metadata})


def proposal(original, *, blocks=None):
    return {"candidates": [{"original_trajectory_id": original.trajectory_id, "blocks": blocks or [
        {"block_id": "key", "op": "replace", "start": 0, "end_exclusive": 1, "actions": ["take key"]},
        {"block_id": "door", "op": "replace", "start": 1, "end_exclusive": 2, "actions": ["open door"]}]}]}


def label(reasoning="Use the visible information.", evidence_steps=None, unsupported=False):
    return {"reasoning": reasoning, "evidence_steps": [-1] if evidence_steps is None else evidence_steps,
            "unsupported": unsupported}
