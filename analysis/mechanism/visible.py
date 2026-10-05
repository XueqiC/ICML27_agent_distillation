"""Visible-text-only restatement and deterministic information actions (I1/I2)."""
import re

from sed.visibility import visible_history, visible_observation

ENTITY = r"[a-zA-Z][a-zA-Z0-9]* \d+"


def summarize_visible(history, current):
    observations = [h["visible"]["observation"] for h in visible_history(history)]
    observations.append(visible_observation(current)["observation"])
    visited, objects, held, location = [], {}, None, None
    for text in observations:
        arrival = re.search(r"You arrive at (" + ENTITY + r")\.", text, re.I)
        if arrival:
            location = arrival[1].lower()
            if location not in visited:
                visited.append(location)
        named = re.search(r"(?:On|In) the (" + ENTITY + r"), you see ([^.]+)\.", text, re.I)
        implicit = re.search(r"In it, you see ([^.]+)\.", text, re.I)
        opened = re.search(r"You open the (" + ENTITY + r")\.", text, re.I)
        if opened:
            location = opened[1].lower()
            if location not in visited:
                visited.append(location)
        if named:
            location = named[1].lower()
            if location not in visited:
                visited.append(location)
        where = named[1].lower() if named else location
        content = named[2] if named else implicit[1] if implicit else None
        if content is not None and where:
            # Latest explicit contents observation supersedes older contents there.
            objects = {o: r for o, r in objects.items() if r != where}
            for obj in re.findall(ENTITY, content):
                objects[obj.lower()] = where
        taken = re.search(r"You pick up the (" + ENTITY + r") from the (" + ENTITY + r")\.", text, re.I)
        moved = re.search(r"You (?:move|put) the (" + ENTITY + r") (?:to|in|on) the (" + ENTITY + r")\.", text, re.I)
        if taken:
            held = taken[1].lower()
            objects[held] = "held"
        if moved:
            objects[moved[1].lower()] = moved[2].lower()
            if held == moved[1].lower():
                held = None
        if "You are not carrying anything." in text:
            if held:
                objects.pop(held, None)
            held = None
    return {"receptacles_visited": visited, "objects_seen_where": dict(sorted(objects.items())),
            "held": held, "last_location": location}


def restatement(history, current):
    summary = summarize_visible(history, current)
    objects = "; ".join(f"{o}: {r}" for o, r in summary["objects_seen_where"].items()) or "none recorded"
    return ("Facts so far:\nReceptacles visited: " + (", ".join(summary["receptacles_visited"]) or "none recorded")
            + "\nObjects seen where (latest observation): " + objects
            + "\nHeld per visible feedback: " + (summary["held"] or "none recorded"))


def information_action(history, current):
    current = visible_observation(current)
    summary = summarize_visible(history, current)
    commands = current["admissible_commands"]
    if "Nothing happens." in current["observation"] and summary["held"]:
        selected = "inventory"
    elif summary["last_location"] and "examine " + summary["last_location"] in commands:
        selected = "examine " + summary["last_location"]
    else:
        selected = "look"
    if selected not in commands:
        raise ValueError("Fixed I2 information action is not admissible")
    return selected
