"""Minimal assistant-action examples; arbitrary audit/event fields are ignored.

Tokenization and model-specific loss masks are not implemented in Stage 1.
This loader provides role-level supervision and refuses dev/test sources.
"""

from sed.manifest import authorize_task
from sed.storage import canonical_bytes
from sed.visibility import visible_history, visible_observation, visible_task


def build_action_example(record):
    authorize_task(record["task"], "generation")
    action = record["action_parsed"]
    if type(action) is not str or not action:
        raise ValueError("Training action must be parsed text")
    payload = {"task": visible_task(record["task"]),
               "history": visible_history(record["history"]),
               "visible_context": visible_observation(record["visible_context"])}
    # Construct every output field explicitly: never dict(record) or **record.
    return {"messages": [
                {"role": "user", "content": canonical_bytes(payload).decode(), "supervised": False},
                {"role": "assistant", "content": action, "supervised": True}],
            "label_provenance": {"event_id": record["event_id"],
                                 "source": record["label_source"]}}
