"""Single sampled action through the unchanged vLLM student interface."""

from sed.storage import digest
from students.vllm_student import Student


class SampledStudent(Student):
    def __init__(self, *, sampling_seed=0, task_id=None, **kwargs):
        super().__init__(**kwargs)
        if type(sampling_seed) is not int:
            raise ValueError("Sampling seed must be an integer")
        self.sampling_seed, self.task_id = sampling_seed, task_id
        self.spec.update(self.sampling(), decoding="sampled", sampling_seed=sampling_seed,
                         request_seed_rule="sha256([sampling_seed, task_id, zero_based_turn])[:8]")

    def sampling(self):
        return {"temperature": 0.7, "top_p": 0.95, "n": 1}

    def request_sampling(self, history, task):
        # Identical canonical JSON, truncation and unsigned conversion to SAG.
        # Identity is accounting metadata; it never enters the visible prompt.
        task_id = self.task_id or task.get("task_id")
        if not task_id:
            raise ValueError("Sampled decoding requires a task identity")
        seed = int(digest([self.sampling_seed, task_id, len(history)])[:8], 16)
        return {**self.sampling(), "seed": seed}
