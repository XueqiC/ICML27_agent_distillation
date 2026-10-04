"""Compatibility import; ReAct has one shared parser in sed.react."""
from sed.react import ActionParseError, parse_action

__all__ = ["ActionParseError", "parse_action"]
