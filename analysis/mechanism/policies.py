"""Analysis-only prompt overlays, seeded sampling and capped existing teacher client."""
from copy import deepcopy
from pathlib import Path

from analysis.mechanism.common import output_guard, read_json
from sed.react import PROMPT_TEMPLATE
from sed.storage import digest, locked, write_json
from students.vllm_student import Student
from teachers.openai_teacher import Teacher, TeacherCallError, rate_limit_options

ADVICE_INSTRUCTION = (
    "For this turn only, write a short visible plan or suggestion for the next decision. "
    "Do not issue an action or an Action: line. Use only the visible conversation."
)
ADVICE_PROMPT_TEMPLATE = (
    "You advise an agent acting in an ALFWorld household environment. The task is "
    "in the initial observation. Observations and admissible_commands are supplied "
    "as JSON in user turns. " + ADVICE_INSTRUCTION
)


class Overlay:
    """Append to one user turn; replay the identical appended text in future requests."""
    def __init__(self):
        self.additions = {}

    def append(self, turn, text):
        if turn in self.additions:
            raise ValueError("Only one intervention per user turn")
        self.additions[turn] = text

    def apply(self, messages):
        messages = deepcopy(messages)
        for turn, text in self.additions.items():
            index = 1 + 2 * turn
            if index < len(messages):
                if messages[index]["role"] != "user":
                    raise ValueError("Unexpected shared conversation layout")
                messages[index]["content"] += "\n\n" + text
        return messages


class BranchStudent(Student):
    def __init__(self, *, sample_index=0, sampling_seed=0, overlay=None, **kwargs):
        if type(sample_index) is not int or sample_index < 0:
            raise ValueError("sample_index=0 is greedy; positive indices sample at T=0.7")
        super().__init__(**kwargs)
        if self.spec["max_tokens"] != 1024 or self.spec["chat_template_kwargs"]["enable_thinking"]:
            raise ValueError("Mechanism students require OFF and 1024 tokens")
        self.sample_index, self.sampling_seed = sample_index, sampling_seed
        self.overlay = overlay or Overlay()
        self.spec.update(self.sampling(), sample_index=sample_index)

    def sampling(self):
        return {"temperature": .7, "seed": self.sampling_seed} if self.sample_index else {"temperature": 0}

    def messages(self, history, task, visible_context):
        return self.overlay.apply(super().messages(history, task, visible_context))


class BudgetExceeded(TeacherCallError):
    incomplete = True


class TeacherBudget:
    """Durable conservative HTTP-attempt caps, shared across probes/arms/restarts."""
    def __init__(self, path, *, max_episode, max_run):
        if any(type(n) is not int or n < 1 for n in (max_episode, max_run)):
            raise ValueError("Explicit positive per-episode and per-run teacher caps required")
        self.path = Path(path)
        output_guard(self.path.parent)
        self.limits = {"max_episode": max_episode, "max_run": max_run}

    def read(self):
        state = read_json(self.path) if self.path.exists() else {
            "limits": self.limits, "total": 0, "episodes": {}, "reservations": []}
        if state["limits"] != self.limits:
            raise ValueError("Teacher caps changed on resume")
        return state

    def reserve(self, episode_id, namespace):
        with locked(str(self.path) + ".lock"):
            state = self.read()
            count = state["episodes"].get(episode_id, 0)
            if state["total"] >= self.limits["max_run"] or count >= self.limits["max_episode"]:
                raise BudgetExceeded("Explicit teacher HTTP-attempt cap reached")
            state["total"] += 1
            state["episodes"][episode_id] = count + 1
            state["reservations"].append({"episode": episode_id, "namespace": namespace})
            write_json(self.path, state)

    def snapshot(self):
        """Stable aggregate view; keep the durable accounting log in arrival order."""
        state = self.read()
        state["reservations"] = sorted(state["reservations"], key=lambda r: (r["episode"], digest(r["namespace"])))
        return state


class CappedLimiter:
    def __init__(self, limiter, budget, episode_id, namespace):
        self.limiter, self.budget = limiter, budget
        self.path = limiter.path
        self.episode_id, self.namespace = episode_id, namespace
        self.exhausted = False

    def acquire(self, tokens):
        # Reserve before waiting so interruption never refunds uncertain work.
        try:
            self.budget.reserve(self.episode_id, self.namespace)
        except BudgetExceeded:
            self.exhausted = True
            raise
        return self.limiter.acquire(tokens)

    def defer_until(self, deadline):
        return self.limiter.defer_until(deadline)


class BranchTeacher(Teacher):
    def __init__(self, *, namespace, sample_index, budget, episode_key, overlay=None,
                 advice=False, **kwargs):
        # This is an advice-only teacher call, not the evaluation/action policy.
        # Avoid contradictory system instructions requiring an Action: line.
        kwargs.setdefault("prompt_template", ADVICE_PROMPT_TEMPLATE if advice else PROMPT_TEMPLATE)
        super().__init__(purpose="diagnosis", **kwargs)
        self.namespace, self.sample_index = namespace, sample_index
        self.overlay, self.advice = overlay or Overlay(), advice
        self.limiter = CappedLimiter(self.limiter, budget, episode_key,
                                    {"branch": namespace, "sample_index": sample_index, "advice": advice})
        self.spec = {"provider_model_id": self.provider_model_id, "sampling": self.sampling,
                     "sample_index": sample_index, "namespace": namespace, "advice": advice,
                     "advice_template_sha256": digest(ADVICE_PROMPT_TEMPLATE) if advice else None}

    def prompt(self, history, task, visible_context):
        messages = self.overlay.apply(super().prompt(history, task, visible_context))
        if self.advice:
            messages[-1]["content"] += "\n\n" + ADVICE_INSTRUCTION
        return messages

    def cache_key(self, history, task, visible_context):
        return digest({"base_key": super().cache_key(history, task, visible_context),
                       "namespace": self.namespace, "sample_index": self.sample_index, "advice": self.advice})

    def generate(self, history, task, visible_context):
        try:
            result = super().generate(history, task, visible_context)
            return result._replace(metadata=dict(result.metadata, mechanism_sample_index=self.sample_index,
                                                   mechanism_namespace=self.namespace))
        except Exception:
            if self.limiter.exhausted:
                raise BudgetExceeded("Teacher cap reached; stage NOT complete") from None
            raise


def make_teacher(config, workspace, budget, episode_key, *, namespace, sample_index, advice=False):
    directory = Path(workspace)
    spec = config["models"]["teacher"]
    return BranchTeacher(namespace=namespace, sample_index=sample_index, budget=budget,
        episode_key=episode_key, advice=advice, cache_dir=directory / "teacher_cache",
        usage_ledger=directory / "teacher_usage.jsonl", run_id=read_json(directory/"provenance.json")["run_id"],
        provider_model_id=spec["provider_model_id"], sampling=spec["sampling"],
        max_attempts=spec["max_attempts"], timeout_seconds=spec["timeout_seconds"],
        secrets_file=config["paths"]["secrets_file"],
        **rate_limit_options(spec, directory / "teacher_rate_limit"))


def student_spec(config, args):
    from tools.check_lora_serving import validate_report
    model = config["models"]["student"]
    if args.server_provenance:
        config["server_provenance"] = read_json(args.server_provenance)
    if args.lora_check:
        config["lora_equivalence"] = read_json(args.lora_check)
    if args.model_name != model["requested_repo"]:
        if not config.get("server_provenance") or not config.get("lora_equivalence"):
            raise ValueError("Adapter continuation requires a passing equivalence gate")
        validate_report(config["lora_equivalence"], config["server_provenance"], args.model_name, args.endpoint)
        checked = config["lora_equivalence"]["identity"]["adapter"]
        if checked["base_model"] != model["requested_repo"] or checked["revision"] != model["revision"]:
            raise ValueError("Adapter gate differs from configured student")
    return {"endpoint": args.endpoint, "model": args.model_name, "revision": model["revision"],
            "enable_thinking": False, "max_tokens": 1024}


def validate_continuation(config, spec, parent_path, *, subject="student", verified=None):
    """Bind S to its evaluated checkpoint and I5 to S's actual training collection."""
    from sed.storage import file_hash
    parent_config = read_json(Path(parent_path).parent / "provenance.json")["config"]
    parent_gate = parent_config.get("lora_equivalence", {}).get("identity", {})
    current_gate = config.get("lora_equivalence", {}).get("identity", {})
    base = config["models"]["student"]["requested_repo"]
    if subject == "base":
        if spec["model"] != base or config.get("server_provenance", {}).get("config", {}).get("server", {}).get("merged_model"):
            raise ValueError("Base controls require the original configured base checkpoint")
    elif parent_gate.get("adapter"):
        if parent_gate["adapter"] != current_gate.get("adapter"):
            raise ValueError("Continuation checkpoint differs from the evaluated student")
    elif spec["model"] != parent_config.get("suite", {}).get("student", {}).get("model"):
        raise ValueError("Continuation model differs from evaluated student")
    if verified is not None:
        if not parent_gate.get("adapter"):
            raise ValueError("I5 needs an explicit training provenance binding for this student")
        server = parent_config["server_provenance"]["config"]["server"]
        name = parent_config["suite"]["student"]["model"]
        source = next(a for a in server["lora_adapters"] if a["name"] == name)
        training_path = Path(source["path"]).parent / "provenance.json"
        if file_hash(training_path) != parent_gate["adapter"]["training_provenance_sha256"]:
            raise ValueError("Student training provenance changed")
        training = read_json(training_path)
        trained_collection = Path(training["config"]["sft"]["verified"]) / "manifest.json"
        saved_hash = training["inputs"].get(str(trained_collection))
        if not saved_hash or file_hash(Path(verified)/"manifest.json") != saved_hash:
            raise ValueError("I5 collection differs from S's actual training set")


def add_student_args(parser, *, required=True):
    parser.add_argument("--endpoint", required=required)
    parser.add_argument("--model-name", required=required)
    parser.add_argument("--server-provenance")
    parser.add_argument("--lora-check")


def add_budget_args(parser):
    parser.add_argument("--max-teacher-calls", type=int, required=True, help="Per parent episode HTTP-attempt cap")
    parser.add_argument("--max-teacher-calls-run", type=int, required=True)
    parser.add_argument("--budget-dir", help="Owned mechanism workspace sharing caps across FCE and arms")
