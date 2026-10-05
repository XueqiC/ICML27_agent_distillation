"""SFT-model MC requests with independent, worker-invariant rollout seeds."""
from sed.storage import digest
from students.vllm_student import Student


def rollout_seed(sampling_seed, parent_seed_id, task_id, kind, iteration, j, sample):
    return int(digest([sampling_seed, parent_seed_id, task_id, kind, iteration, j, sample])[:8], 16)


class MCStudent(Student):
    def __init__(self, *, sampling_seed, **kwargs):
        super().__init__(**kwargs)
        if self.spec["max_tokens"] != 512 or self.spec["chat_template_kwargs"]["enable_thinking"]:
            raise ValueError("IPR MC fixes OFF and 512 output tokens")
        self.sampling_seed = sampling_seed
        self.spec.update(self.sampling(), sampling_seed=sampling_seed,
                         request_seed_rule="sha256([rollout_seed, zero_based_turn])[:8]")

    def sampling(self):
        return {"temperature": 1, "top_p": 1, "top_k": -1}

    def request_sampling(self, history, task):
        return {**self.sampling(), "seed": int(digest([self.sampling_seed, len(history)])[:8], 16)}
