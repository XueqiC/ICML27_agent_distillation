"""One parser shared by teacher and scripted policies."""

import re


class ActionParseError(ValueError):
    pass


def parse_action(raw):
    if type(raw) is not str:
        raise ActionParseError("Action output must be text")
    value = raw.strip()
    markers = list(re.finditer(r"(?im)^\s*ACTION\s*:\s*", value))
    if len(markers) == 1:
        value = value[markers[0].end():].strip()
    elif markers:
        raise ActionParseError("Multiple action markers")
    if not value or "\n" in value or "\r" in value or value.startswith("```"):
        raise ActionParseError("Expected exactly one command")
    return value
