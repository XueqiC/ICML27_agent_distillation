"""Deterministic HAD text from visible feedback, commands and realized history.

No environment internals, rewards, success labels or teacher predictions are read.
The runner owns execution, parsing and termination; this module only adds text.
"""
from collections import Counter
import re

from baselines.had import RECIPE
from sed.react import action_outcome, build_messages
from sed.storage import digest
from sed.visibility import visible_history, visible_observation


def records(history, visible, *, budget=40):
    history, visible = visible_history(history), visible_observation(visible)
    if not 0 <= len(history) < budget:
        raise ValueError("HAD requires a live policy turn within the H0 budget")
    location, holdings = None, []  # ALFWorld resets with empty inventory.
    places, visits, seen, failures = set(), [], set(), Counter()
    previous_key, previous_action, last_new = None, None, 0
    for turn, page in enumerate([h["visible"] for h in history] + [visible]):
        if turn and visits:
            visits[-1]["commands"] += int(previous_action is not None)
        text, commands = page["observation"], page["admissible_commands"]
        places.update(c[6:] for c in commands if c.startswith("go to "))
        match = re.search(r"You (?:arrive at|are at) (?:the )?([^\n.]+)\.", text)
        if match:
            location = match[1]
        pick = re.search(r"You pick up (?:the )?(.+?) from (?:the )?[^\n.]+\.", text)
        put = re.search(r"You put (?:the )?(.+?) (?:in/on|on|in) (?:the )?[^\n.]+\.", text)
        if pick:
            holdings = [pick[1]]
        if put:
            holdings = [obj for obj in holdings if obj != put[1]]
        inventory = re.search(r"You are (?:carrying|holding):? ([^\n.]+)\.", text)
        if inventory:
            holdings = ([] if inventory[1] in {"nothing", "nothing at all"} else
                        sorted(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]* \d+", inventory[1])))
        if location:
            places.add(location)
            if not visits or visits[-1]["location"] != location:
                if visits:
                    visits[-1]["left_turn"] = turn
                visits.append({"location": location, "arrived_turn": turn, "commands": 0})
        key = digest([location, sorted(holdings), commands])
        if turn:
            if previous_action is not None and (key == previous_key or "Nothing happens." in text):
                failures[(previous_key, previous_action)] += 1
        if key not in seen:
            last_new = turn
            seen.add(key)
        previous_key = key
        if turn < len(history):
            previous_action = action_outcome(history[turn].get("reply", "Action: " + history[turn].get("action", "")), commands)[0]
    visited = sorted({v["location"] for v in visits})
    return {"available_actions": visible["admissible_commands"], "location": location,
            "inventory": sorted(holdings), "visits": visits, "visited_locations": visited,
            "unvisited_locations": sorted(places - set(visited)), "state_key": previous_key,
            "step": len(history) + 1, "budget": budget, "remaining": budget - len(history) - 1,
            "budget_warning": budget - len(history) - 1 < RECIPE["budget_warning"],
            "failed_actions": [{"action": action, "attempts": count}
                for (state, action), count in sorted(failures.items())
                if state == previous_key and count >= RECIPE["failed_attempts"]],
            "stall": len(history) - last_new >= RECIPE["stall_window"],
            "steps_without_new_state": len(history) - last_new}


def render(record):
    lines = ["[HAD state tracking]",
        "Available actions: " + " / ".join(record["available_actions"]),
        "Location: " + (record["location"] or "unknown"),
        "Inventory: " + (", ".join(record["inventory"]) or "empty"),
        "Visited locations: " + (", ".join(record["visited_locations"]) or "none"),
        "Unvisited locations: " + (", ".join(record["unvisited_locations"]) or "none known")]
    for visit in record["visits"]:
        lines.append(f"Visit {visit['location']}: arrived turn {visit['arrived_turn']}, "
                     f"{visit['commands']} commands, left turn {visit.get('left_turn', 'current')}.")
    lines.append(f"Step {record['step']} of {record['budget']}.")
    if record["budget_warning"]:
        lines.append(f"Budget warning: {record['remaining']} steps remain after this turn.")
    for failed in record["failed_actions"]:
        lines.append(f"Repeated action at this state: {failed['action']!r} had no effect "
                     f"on {failed['attempts']} attempts. Choose a different available action.")
    if record["stall"]:
        lines.append(f"Stall: no new state in {record['steps_without_new_state']} steps.")
    return "\n".join(lines)


def augment(history, visible):
    page = visible_observation(visible)
    return {"observation": page["observation"] + "\n\n" + render(records(history, page)),
            "admissible_commands": page["admissible_commands"]}


def messages(history, task, visible):
    # Appendix B.1: harness on the latest page; historical replies stay realized.
    return build_messages(history, task, augment(history, visible))


def validity(action, record):
    """Check only the preferred action, against pre-action records."""
    if action not in record["available_actions"]:
        return False, "unavailable_action"
    held = record["inventory"]
    needs = re.match(r"(?:put|heat|cool|clean) (.+?) (?:in/on|in|on|with) ", action)
    tool = re.match(r"slice .+? with (.+)$", action)
    if ((needs and needs[1] not in held) or (tool and tool[1] not in held)
            or (action.startswith("take ") and held)):
        return False, "inventory_contradiction"
    if action in [r["action"] for r in record["failed_actions"]]:
        return False, "repeated_no_effect"
    return True, "no_recorded_contradiction"
