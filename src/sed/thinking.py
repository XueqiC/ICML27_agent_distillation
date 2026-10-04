"""Separate Gemma student thinking from visible ReAct (no vLLM/GPU imports).

Boundary spellings checked against installed vLLM 0.27.1 parser/gemma4.py.
Unclosed thought channels are thinking-only, including on token-limit finishes.
"""

import re


def split_thinking(text):
    if text is None:
        text = ""
    if type(text) is not str:
        raise TypeError("Completion content must be text")
    thinking, visible = [], []
    remaining = text
    while "<|channel>" in remaining or "<channel|>" in remaining:
        start = remaining.find("<|channel>")
        end = remaining.find("<channel|>")
        if end >= 0 and (start < 0 or end < start):
            # Opener can be in the prompt, outside returned completion text.
            thinking.append(remaining[:end])
            remaining = remaining[end + len("<channel|>"):]
        elif start >= 0:
            visible.append(remaining[:start])
            remaining = remaining[start + len("<|channel>"):]
            end = remaining.find("<channel|>")
            if end < 0:
                thinking.append(remaining)
                remaining = ""
            else:
                thinking.append(remaining[:end])
                remaining = remaining[end + len("<channel|>"):]
    # A bare thought channel prefix is never executable ReAct.
    if remaining.lstrip().startswith("thought\n"):
        thinking.append(remaining)
    else:
        visible.append(remaining)
    clean = "".join(visible).strip()
    clean = re.sub(r"(?:\s*(?:<turn\|>|<eos>))+$", "", clean).strip()
    blocks = [re.sub(r"^thought(?:\r?\n|$)", "", t.strip()).strip() for t in thinking]
    return clean, "\n".join(blocks)
