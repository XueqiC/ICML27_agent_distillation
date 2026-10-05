"""Kang SAG action vote; only the selected candidate reaches the environment.

Adapted from agent-distillation 8884b80, src/smolagents/agents.py:1435-1515.
Integrity uses our shared ReAct parser (Thought is optional) and admissibility.
Neither hidden verification nor speculative environment execution is used.
"""
from sed.react import action_outcome
from students.vllm_student import Student, separate_response


def vote(replies, commands):
    if not replies:
        raise ValueError("SAG requires candidates")
    groups, candidates = {}, []
    for index, reply in enumerate(replies):
        action, outcome = action_outcome(reply, commands)
        passing = outcome == "legal"
        candidates.append({"sample": index, "reply": reply, "action": action,
                           "action_outcome": outcome, "passing": passing})
        if passing:
            groups.setdefault(action, []).append(index)
    if not groups:
        selected, reason = 0, "no_passing_sample_0"
    else:
        # dict insertion order implements the official stable group tie break.
        winner = max(groups, key=lambda action: len(groups[action]))
        selected = groups[winner][0]
        reason = "all_singletons_first_passing" if all(len(g) == 1 for g in groups.values()) else "largest_group"
    return {"candidates": candidates, "groups": [{"action": a, "samples": g} for a, g in groups.items()],
            "selected_sample": selected, "reason": reason}


class SAGStudent(Student):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.spec["max_tokens"] != 1024 or self.spec["chat_template_kwargs"]["enable_thinking"]:
            raise ValueError("SAG fixes thinking OFF and max_tokens=1024")
        self.spec.update(self.sampling(), decoding="sag")

    def sampling(self):
        return {"n": 8, "temperature": 0.4, "top_p": 0.8}

    def select_choice(self, result, visible_context, blobs):
        choices = result["choices"]
        if len(choices) != 8:
            raise ValueError("SAG requires exactly eight returned samples")
        separated = [separate_response(c["message"]) for c in choices]
        selection = vote([s[0] for s in separated], visible_context["admissible_commands"])
        for candidate, choice, (_, thinking, shape) in zip(selection["candidates"], choices, separated):
            candidate.update(finish_reason=choice.get("finish_reason"), response_index=choice.get("index"),
                thinking_chars=len(thinking), thinking_return_shape=shape,
                raw_response_ref=blobs.put({k: choice["message"].get(k) for k in
                                          ("content", "reasoning", "reasoning_content", "refusal")}))
        thinking_tokens = ((result.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens")
        if thinking_tokens is None and all(not s[1] and c.get("finish_reason") == "stop"
                                           for s, c in zip(separated, choices)):
            thinking_tokens = 0
        return choices[selection["selected_sample"]], {"decoding": "sag", "sag": selection,
            "thinking_chars": sum(len(s[1]) for s in separated), "thinking_tokens": thinking_tokens}
