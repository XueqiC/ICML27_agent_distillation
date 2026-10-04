"""Allowlist at every boundary; unknown fields are never model input."""


def visible_observation(value):
    observation = value["observation"]
    commands = value["admissible_commands"]
    if not isinstance(observation, str):
        raise TypeError("Observation must be text")
    if not isinstance(commands, (list, tuple)) or any(type(c) is not str for c in commands):
        raise TypeError("Admissible commands must be strings")
    return {"observation": observation, "admissible_commands": sorted(set(commands))}


def visible_history(history):
    result = []
    for turn in history:
        item = {"visible": visible_observation(turn["visible"])}
        if "reply" in turn:
            from sed.thinking import split_thinking
            item["reply"] = split_thinking(turn["reply"])[0]
        if "action" in turn:
            if type(turn["action"]) is not str:
                raise TypeError("History action must be text")
            item["action"] = turn["action"]
        result.append(item)
    return result


def visible_task(task):
    # task_id/family, split, file paths, mechanism IDs and audit are NOT prompts:
    # ALFWorld filenames reveal goal objects and layouts. Goal text is already
    # in the environment's initial visible observation.
    instruction = task.get("instruction", "")
    if type(instruction) is not str:
        raise TypeError("Task instruction must be text")
    return {"instruction": instruction}
