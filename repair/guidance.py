"""Condition-guidance probe: public prompts, bounded labels and paired evaluation.

Independent of repair prompt/schema versions. Nothing here produces training data.
"""
from copy import deepcopy
from dataclasses import dataclass, replace
from hashlib import sha256
import csv
import json
from pathlib import Path
import re
import uuid

from repair.artifacts import read_json
from repair.budgets import GlobalTeacherBudget
from repair.execute import execute_actions, environment_version, task_fingerprint
from repair.ledger import BudgetLedger, teacher_token_usage
from repair.records import TrajectoryRecord
from repair.runtime import CaseStopped, RepairTeacher
from sed.manifest import authorize_task
from sed.provenance import utc_now
from sed.react import action_outcome, build_messages
from sed.storage import canonical_bytes, digest, file_hash, locked, write_json
from sed.thinking import split_thinking
from sed.visibility import visible_observation
from teachers.openai_teacher import Teacher, TeacherResult
from teachers.stage import require_completed

GROUPS = (
    'look_at_obj_in_light-Newspaper-None-DeskLamp-224',
    'pick_and_place_simple-Kettle-None-Shelf-1',
    'pick_clean_then_place_in_recep-Kettle-None-DiningTable-7',
    'pick_clean_then_place_in_recep-Lettuce-None-CounterTop-18',
    'pick_cool_then_place_in_recep-Potato-None-SideTable-21',
    'pick_cool_then_place_in_recep-Apple-None-Microwave-19',
    'pick_cool_then_place_in_recep-Egg-None-Microwave-5',
)
BRANCH_INDEX = 10
OUTPUT_CAP = 8192
GENERAL_REMINDER = 'Review your progress so far, then continue to complete the task.'
SUGGESTION_JOIN = ' A next step that can gather this information: '
TEMPLATE_VERSION = 'cg-probe/condition_guidance/1'
TEMPLATE = """Analyze ONLY this public student history. Infer the current plan and subgoal
from the student's own visible Thoughts, not from a replacement plan of your own.
Use only the supplied task, initial observation, ten prefix replies/actions/observations,
current admissible commands, and step budget. Never infer hidden state or target location.
Return JSON only with exactly these four fields:
{"student_plan":"current plan and subgoal inferred from the student's Thoughts",
 "conditions":[{"condition":"premise required by that plan",
 "status":"undetermined", "evidence_steps":[-1]}],
 "condition_hint":"at most 70 English words",
 "action_suggestion":{"command":"exact admissible command", "reason":"at most 30 words"}}.
Each condition has status undetermined (not established), known_unsatisfied (known but
not yet met), refuted (ruled out by observations), or satisfied (observationally supported).
Evidence steps are zero-based 0 through 9; -1 denotes the initial observation. An empty
evidence_steps list is appropriate for a premise with no observational support.
condition_hint must name the student's current subgoal, the premise without any
observational support, and what the observations so far have ruled out. It MUST NOT
suggest where to look, name unvisited locations, give any command, or guess the location.
It must not contain any current admissible command string or mention any receptacle
instance or type (singular or plural) not visited by a legal go to/open/examine action
in the prefix. Keep C separate from the action suggestion; do not put advice in C.
action_suggestion.command must be copied EXACTLY from admissible_commands_now, and its
verb must be go to, open, examine, or look. Its reason may use only public information
and general household knowledge. Teacher statements are hypotheses, not environment truth.
"""


@dataclass(frozen=True)
class GuidancePrompt:
    payload: dict
    retry_reason: str | None = None
    retry_index: int = 0
    kind: str = 'condition_guidance'

    @property
    def template_sha256(self):
        return sha256((TEMPLATE_VERSION + '\n' + TEMPLATE).encode()).hexdigest()

    def to_json(self):
        return dict(kind=self.kind, template_version=TEMPLATE_VERSION,
                    template_sha256=self.template_sha256, format_retry=self.retry_reason is not None,
                    retry_reason=self.retry_reason, retry_index=self.retry_index, payload=self.payload)

    def messages(self):
        retry = (f'\nFormat retry {self.retry_index} of 2: preceding output rejected: ' + self.retry_reason
                 + '. Return a corrected JSON object following all constraints.') if self.retry_reason else ''
        return [{'role': 'system', 'content': TEMPLATE + retry},
                {'role': 'user', 'content': canonical_bytes(self.payload).decode()}]


class GuidanceTeacher(RepairTeacher):
    """Reuse repair's structured-response transport, with an isolated envelope."""
    def __init__(self, **kwargs):
        Teacher.__init__(self, prompt_template=TEMPLATE_VERSION + '\n' + TEMPLATE, **kwargs)

    def prompt(self, history, task, visible_context):
        value = json.loads(visible_context['observation'])
        prompt = GuidancePrompt(value['payload'], value['retry_reason'], value['retry_index'])
        if history or prompt.to_json() != value:
            raise ValueError('Invalid condition_guidance envelope')
        return prompt.messages()


def make_guidance_teacher(config, output, *, run_id):
    # Same factory/transport used by runtime.make_teachers; no OUTPUT_CAPS change.
    from teachers.capped import from_config
    spec = config['models']['teacher']
    if spec['requested_name'] != 'gpt-6-luna' or spec['provider_model_id'] != 'gpt-6-luna':
        raise ValueError('CG probe requires configured gpt-6-luna')
    output = Path(output)
    return from_config(config, output=output, cache_dir=output / 'cache',
        usage_ledger=output / 'usage.jsonl', run_id=run_id, cap=3 * len(GROUPS),
        teacher_class=GuidanceTeacher,
        sampling={**spec['sampling'], 'max_output_tokens': OUTPUT_CAP})


def target_pattern(target):
    return re.compile(r'\b' + re.escape(target) + r' \d+\b', re.I)


def prefix_sha(original):
    return digest([s['action_raw'] for s in original.steps[:BRANCH_INDEX]])


def guidance_prompt(original):
    """Only preregistered public scalar fields, never full records or task paths."""
    initial = original.initial['visible']['observation']
    match = re.search(r'Your task is to:\s*(.*)', initial, re.I | re.S)
    task = match[1].strip() if match else initial
    return GuidancePrompt({'task': task, 'initial_observation': initial,
        'prefix': [{'action_index': i, 'reply': split_thinking(s['action_raw'])[0],
                    'action': s['action_parsed'], 'observation': s['visible']['observation']}
                   for i, s in enumerate(original.steps[:BRANCH_INDEX])],
        'admissible_commands_now': list(original.steps[9]['visible']['admissible_commands']),
        'steps_used': BRANCH_INDEX, 'step_limit': 40})


def receptacles(initial_observation):
    # Parse only the public room inventory, never the goal or game metadata.
    listing = re.search(r'\byou see\s+(.+?)(?:\.|\n|$)', initial_observation, re.I)
    if not listing:
        raise ValueError('Missing public receptacle inventory (you see ...)')
    instances = re.findall(r'\b(?:a|an)\s+([a-z][a-z ]*?\s+\d+)\b', listing[1], re.I)
    if not instances:
        raise ValueError('Empty public receptacle inventory')
    return {re.sub(r'\s+', ' ', name.lower()) for name in instances}


def visit_object(step):
    if step.get('action_outcome') != 'legal':
        return None
    match = re.fullmatch(r'(?:go to|open|examine)\s+(.+?\s+\d+)', step.get('action_parsed') or '', re.I)
    return re.sub(r'\s+', ' ', match[1].lower()) if match else None


def type_pattern(kind):
    # ALFWorld uses closed compounds (countertop, sidetable); also recognize
    # ordinary English spacing and regular/irregular plural forms in a hint.
    plurals = {kind + ('es' if kind.endswith(('s', 'x', 'ch', 'sh')) else 's')}
    if kind.endswith('y') and len(kind) > 1 and kind[-2] not in 'aeiou':
        plurals.add(kind[:-1] + 'ies')
    if kind.endswith('shelf'):
        plurals.add(kind[:-5] + 'shelves')
    forms = {kind, *plurals}
    patterns = []
    for form in forms:
        spaced = form
        for left, right in [('counter', 'top'), ('side', 'table'), ('dining', 'table'),
                            ('coffee', 'table'), ('sink', 'basin'), ('bathtub', 'basin'),
                            ('garbage', 'can'), ('desk', 'lamp')]:
            spaced = spaced.replace(left + right, left + ' ' + right)
        patterns.extend([re.escape(form), re.escape(spaced).replace(r'\ ', r'[\s-]+')])
    return r'\b(?:' + '|'.join(sorted(set(patterns))) + r')\b'


class GuidanceValidationError(ValueError):
    pass


def validate_guidance(raw, original):
    """Automatic checks only; human approval still checks indirect advice/status."""
    def require(ok, reason):
        if not ok:
            raise GuidanceValidationError(reason)
    try:
        value = json.loads(raw)
    except (ValueError, TypeError):
        raise GuidanceValidationError('invalid_json') from None
    require(type(value) is dict and set(value) == {
        'student_plan', 'conditions', 'condition_hint', 'action_suggestion'}, 'top_level_schema')
    require(type(value['student_plan']) is str and bool(value['student_plan'].strip()), 'student_plan_schema')
    require(type(value['conditions']) is list and bool(value['conditions']), 'conditions_schema')
    for condition in value['conditions']:
        require(type(condition) is dict and set(condition) == {'condition', 'status', 'evidence_steps'}, 'condition_schema')
        require(type(condition['condition']) is str and bool(condition['condition'].strip()), 'condition_text')
        require(type(condition['status']) is str and condition['status'] in {
            'undetermined', 'known_unsatisfied', 'refuted', 'satisfied'}, 'condition_status')
        ids = condition['evidence_steps']
        require(type(ids) is list and all(type(i) is int and -1 <= i <= 9 for i in ids), 'evidence_step_ids')
    hint = value['condition_hint']
    require(type(hint) is str and bool(hint.strip()), 'condition_hint_schema')
    require(len(hint.split()) <= 70, 'condition_hint_word_limit')
    commands = original.steps[9]['visible']['admissible_commands']
    require(not any(c.lower() in hint.lower() for c in commands), 'condition_hint_contains_admissible_command')
    inventory = receptacles(original.initial['visible']['observation'])
    visited = {visit_object(s) for s in original.steps[:10]} & inventory
    visited_types = {r.rsplit(' ', 1)[0] for r in visited}
    for kind in sorted({r.rsplit(' ', 1)[0] for r in inventory}):
        for match in re.finditer(type_pattern(kind) + r'\s+(\d+)\b', hint, re.I):
            require(kind + ' ' + match[1] in visited, 'condition_hint_unvisited_instance')
    for kind in sorted({r.rsplit(' ', 1)[0] for r in inventory} - visited_types):
        require(not re.search(type_pattern(kind), hint, re.I), 'condition_hint_unvisited_type')
    suggestion = value['action_suggestion']
    require(type(suggestion) is dict and set(suggestion) == {'command', 'reason'}, 'suggestion_schema')
    command, reason = suggestion['command'], suggestion['reason']
    require(type(command) is str and command in commands, 'suggestion_not_admissible')
    require(bool(re.match(r'^(?:go to|open|examine)(?:\s+\S.*)$|^look(?:\s+\S.*)?$', command)), 'suggestion_verb')
    require(type(reason) is str and bool(reason.strip()), 'suggestion_reason_schema')
    require(len(reason.split()) <= 30, 'suggestion_reason_word_limit')
    return value


class _GuidanceReservations:
    """Every underlying HTTP retry reserves before transport, as CaseSession does."""
    def __init__(self, session, prior, bound, attempt):
        self.session, self.prior, self.bound, self.attempt = session, prior, bound, attempt
        self.calls = 0

    def reserve(self, **kwargs):
        if self.calls:
            self.session.budget.reserve(self.bound, OUTPUT_CAP)
            self.session.ledger.append('teacher_transport_retry', task_id=self.session.task['task_id'],
                parent_attempt_id=self.attempt, supervision={'teacher_calls': {'retry': 1},
                'teacher_input_tokens': None, 'teacher_output_tokens': None,
                'teacher_reasoning_tokens': None, 'teacher_visible_tokens': None})
        self.calls += 1
        self.session.ledger.append('transport_reservation', settled=False,
            parent_attempt_id=self.attempt, task_id=self.session.task['task_id'],
            details={'request_id': kwargs['request_id'], 'input_bound': self.bound, 'output_cap': OUTPUT_CAP})
        if self.prior is not None:
            self.prior.reserve(**kwargs)


class GuidanceSession:
    def __init__(self, teacher, task, ledger, budget):
        authorize_task(task, 'generation')
        if task.get('evaluation_only'):
            raise ValueError('Evaluation-only task cannot supply hints')
        self.teacher, self.task, self.ledger, self.budget = teacher, task, ledger, budget

    def request(self, original):
        prompt, attempts = guidance_prompt(original), []
        for retry in range(3):
            bound = len(canonical_bytes(prompt.messages())) + 32
            attempt, kind = str(uuid.uuid4()), 'retry' if retry else 'diagnosis'
            try:
                self.budget.reserve(bound, OUTPUT_CAP)
            except CaseStopped as exc:
                attempts.append({'retry': retry, 'status': 'budget_stopped', 'reason': str(exc)})
                break
            self.ledger.append('teacher_request', settled=False, task_id=self.task['task_id'],
                attempt_id=attempt, details={'kind': prompt.kind, 'request': prompt.to_json(),
                'request_sha256': digest(prompt.messages()), 'input_bound': bound, 'output_cap': OUTPUT_CAP})
            underlying = getattr(self.teacher, 'teacher', self.teacher)
            prior = getattr(underlying, 'reservation_budget', None)
            if isinstance(underlying, Teacher):
                if underlying.sampling['max_output_tokens'] != OUTPUT_CAP:
                    raise ValueError('CG teacher output cap differs')
                underlying.reservation_budget = _GuidanceReservations(self, prior, bound, attempt)
            try:
                result = self.teacher.generate([], self.task,
                    {'observation': canonical_bytes(prompt.to_json()).decode(), 'admissible_commands': []})
                require_completed(result)
            except Exception as exc:
                self.ledger.append('teacher_failure', task_id=self.task['task_id'], attempt_id=attempt,
                    reason=type(exc).__name__, supervision={'teacher_calls': {kind: 1},
                    'teacher_input_tokens': None, 'teacher_output_tokens': None,
                    'teacher_reasoning_tokens': None, 'teacher_visible_tokens': None})
                attempts.append({'retry': retry, 'status': 'teacher_failed', 'reason': type(exc).__name__})
                # No SDK exception text. Unknown reservation remains charged; stop run.
                self.budget.stop_reason = self.budget.stop_reason or 'teacher_transport_failed'
                break
            finally:
                if isinstance(underlying, Teacher):
                    underlying.reservation_budget = prior
            usage = result.metadata['usage']
            billed = teacher_token_usage(result.metadata.get('billed_usage', usage))
            accounted = billed if result.metadata.get('cache_hit') else usage
            self.ledger.append('teacher_result', task_id=self.task['task_id'], attempt_id=attempt,
                supervision={'teacher_calls': {kind: int(not result.metadata.get('cache_hit'))},
                    'teacher_input_tokens': billed['input_tokens'], 'teacher_output_tokens': billed['output_tokens'],
                    'teacher_reasoning_tokens': billed['reasoning_tokens'], 'teacher_visible_tokens': billed['visible_tokens'],
                    'teacher_cached_tokens': usage.get('cached_input_tokens', 0),
                    'teacher_billed_cost': result.metadata.get('billed_cost')},
                details={'kind': prompt.kind, 'raw_output': result.raw_text, 'teacher_metadata': result.metadata,
                         'template_sha256': prompt.template_sha256})
            row = {'retry': retry, 'raw_output': result.raw_text, 'metadata': result.metadata}
            attempts.append(row)
            try:
                self.budget.settle(bound, OUTPUT_CAP, accounted)
            except CaseStopped as exc:
                row.update(status='budget_stopped', reason=str(exc))
                break
            try:
                if result.truncated or usage['output_tokens'] > OUTPUT_CAP:
                    raise GuidanceValidationError('output_cap')
                value = validate_guidance(result.raw_text, original)
                row.update(status='valid', reason=None)
                return value, attempts
            except GuidanceValidationError as exc:
                row.update(status='invalid', reason=str(exc))
                self.ledger.append('schema_rejection', task_id=self.task['task_id'], reason=str(exc),
                                   details={'request_attempt': attempt, 'kind': prompt.kind})
                # Distinct messages prevent a rejected cached retry from being
                # returned again instead of making the second format attempt.
                prompt = replace(prompt, retry_reason=str(exc), retry_index=retry + 1)
        return None, attempts


def branch_texts(value):
    hint = value['condition_hint'] if value else None
    suggestion = value['action_suggestion'] if value else None
    return {'A': None, 'B': GENERAL_REMINDER, 'C': hint,
            'D': hint + SUGGESTION_JOIN + suggestion['command'] + ' (' + suggestion['reason'] + ')' if value else None}


def approve_hints(hints_path, decisions_path):
    hints = read_json(hints_path)
    with Path(decisions_path).open(newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        if not {'task_id', 'decision', 'note'} <= set(reader.fieldnames or []):
            raise ValueError('Decisions TSV requires task_id, decision, note')
        rows = list(reader)
    ids = [c['task_id'] for c in hints['cases']]
    if (len(rows) != len(ids) or {r['task_id'] for r in rows} != set(ids)
            or any(r['decision'] not in {'approve', 'reject'} for r in rows)):
        raise ValueError('Exactly one approve/reject decision required for every case')
    value = {'schema_version': 1, 'hints_sha256': file_hash(hints_path), 'timestamp': utc_now(),
             'decisions_sha256': file_hash(decisions_path),
             'decisions': [{k: r[k] for k in ('task_id', 'decision', 'note')} for r in rows]}
    path = Path(hints_path).parent / 'hints_approval.json'
    write_json(path, value, exclusive=True)
    return path


def approved_cases(hints_path, approval_path):
    hints, approval = read_json(hints_path), read_json(approval_path)
    if approval.get('hints_sha256') != file_hash(hints_path):
        raise ValueError('Approval hints SHA256 mismatch')
    rows = approval.get('decisions', [])
    decisions = {r['task_id']: r for r in rows}
    ids = [c['task_id'] for c in hints['cases']]
    if len(set(ids)) != len(ids) or len(decisions) != len(rows) or set(decisions) != set(ids):
        raise ValueError('Approval must cover each case exactly once')
    result = []
    for case in hints['cases']:
        decision = decisions[case['task_id']]['decision']
        if decision not in {'approve', 'reject'}:
            raise ValueError('Case has no explicit approve/reject decision')
        missing = case['texts']['C'] is None or case['texts']['D'] is None
        result.append({**case, 'branches': ['A', 'B'] if missing or decision == 'reject' else list('ABCD'),
                       'cd_status': 'missing' if missing else 'rejected' if decision == 'reject' else 'approved'})
    return hints, result


class GuidancePolicy:
    def __init__(self, policy, task_id, text, branch_index=BRANCH_INDEX):
        self.policy, self.task_id, self.text, self.branch_index = policy, task_id, text, branch_index

    def __call__(self, history, visible):
        if self.text is not None:
            history, visible = deepcopy(history), deepcopy(visible)
            if len(history) < self.branch_index:
                raise ValueError('Guidance policy called before branch point')
            turn = visible if len(history) == self.branch_index else history[self.branch_index]['visible']
            turn['observation'] += '\n\nGuidance: ' + self.text
        return self.policy.generate(history, {'task_id': self.task_id}, visible).raw_text


def derive_metrics(execution, target, *, branch, suggested_command=None, branch_index=BRANCH_INDEX):
    pattern = target_pattern(target)
    suffix = execution.steps[branch_index:]
    observations = [(s.get('visible') or {}).get('observation', '') for s in suffix]
    found_at = next((i + 1 for i, obs in enumerate(observations) if pattern.search(obs)), None)
    take = re.compile(r'^take ' + re.escape(target) + r' \d+ from .+$', re.I)
    first = suffix[0] if suffix else {}
    raw = split_thinking(first.get('action_raw') or '')[0]
    thought = re.search(r'(?ms)^\s*Thought:\s*(.*?)(?=^\s*Action:|\Z)', raw)
    cleared, revisits = set(), 0
    for i, step in enumerate(execution.steps):
        place = visit_object(step)
        observation = (step.get('visible') or {}).get('observation', '')
        if place:
            if i >= branch_index and (step.get('action_parsed') or '').lower().startswith('go to ') and place in cleared:
                revisits += 1
            if pattern.search(observation):
                cleared.discard(place)
            else:
                cleared.add(place)
    suggested_observations = [obs for s, obs in zip(suffix, observations)
                              if s.get('action_parsed') == suggested_command] if branch == 'D' else []
    return {'found': found_at is not None, 'steps_to_find': found_at,
        'picked': any(s.get('action_outcome') == 'legal' and take.fullmatch(s.get('action_parsed') or '')
                      and obs.startswith('You pick up') for s, obs in zip(suffix, observations)),
        'success': execution.success, 'inadmissible_count': sum(s.get('action_outcome') == 'inadmissible' for s in suffix),
        'first_post_branch_action': first.get('action_parsed'),
        'first_post_branch_Thought': thought[1].strip() if thought else None,
        'followed_suggestion': bool(suffix and first.get('action_parsed') == suggested_command) if branch == 'D' else None,
        'suggestion_reveals_target': any(pattern.search(obs) for obs in suggested_observations) if suggested_observations else None,
        'revisits_of_cleared': revisits}


def rollout_key(case, branch, decoding, seed):
    return digest([case, branch, decoding, seed])


def execute_rollout(case, original, *, branch, decoding, seed, env_factory, policy_factory,
                    output, run_sha256):
    """Per-key lock + atomic immutable result; interrupted call logs are retained."""
    output = Path(output)
    key = rollout_key(case['task_id'], branch, decoding, seed)
    result_path = output / 'records' / (key + '.json')
    with locked(output / 'locks' / (key + '.lock')):
        if result_path.exists():
            result = read_json(result_path)
            if result['run_sha256'] != run_sha256 or result['rollout_key'] != key:
                raise ValueError('Resume rollout identity mismatch')
            return result
        directory = output / 'calls' / key / str(uuid.uuid4())
        directory.mkdir(parents=True, exist_ok=False)
        ledger = BudgetLedger(directory / 'budget.jsonl')
        # Runner episodes have environment_revision, not repair.environment_version.
        # Bind the imported reference to the executor's current adapter/version;
        # original hashes and EVERY reset/prefix transition are still compared.
        env = env_factory()
        try:
            version = environment_version(env)
        finally:
            env.close()
        reference = replace(original, environment_version=version,
            fingerprint=task_fingerprint(version, original.game_sha256, original.initial['visible'], original.env_seed))
        policy = policy_factory(directory / 'student_calls')
        text = case['texts'][branch]
        reached = False
        wrapped = GuidancePolicy(policy, original.task_id, text)
        def continuation(history, visible):
            nonlocal reached
            reached = True  # execute_actions has verified the prefix before this callback.
            return wrapped(history, visible)
        try:
            execution = execute_actions(env_factory, original.task_id,
                [s['action_raw'] for s in original.steps[:10]], original.max_steps,
                prefix_reference=reference, continuation=continuation,
                ledger=ledger, interaction_kind='evaluation_steps')
        finally:
            if hasattr(policy, 'close'):
                policy.close()
        reason = None if reached else execution.error_type or 'prefix_not_completed'
        if execution.termination_reason == 'ENV_ERROR':
            reason = execution.error_type or 'execution_error'
        suggestion = (case.get('teacher_json') or {}).get('action_suggestion', {}).get('command')
        row = {'schema_version': 1, 'rollout_key': key, 'run_sha256': run_sha256,
            'case': original.task_id, 'group': case['group'], 'branch': branch, 'decoding': decoding,
            'seed': seed, 'guidance_text': text, 'guidance_sha256': sha256(text.encode()).hexdigest() if text is not None else None,
            'cd_status': case['cd_status'], 'prefix_verified': reached, 'void': reason is not None,
            'void_reason': reason, 'execution': execution.to_json(),
            'student_call_log_dir': str((directory / 'student_calls').relative_to(output)),
            'budget': ledger.summary(), 'metrics': derive_metrics(execution, case['target'],
                branch=branch, suggested_command=suggestion) if reason is None else None}
        write_json(result_path, row, exclusive=True)
        return row


def decision_rule(x, y, left, right):
    if not x or len(x) != len(y):
        return 'no clear difference'
    difference = sum(a - b for a, b in zip(x, y)) / len(x)
    balance = sum(a > b for a, b in zip(x, y)) - sum(a < b for a, b in zip(x, y))
    return f'{left} better than {right}' if difference >= .15 and balance >= 2 else 'no clear difference'


def cluster_bootstrap(differences):
    import numpy as np
    if not differences:
        return None
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(0)
    draws = values[rng.integers(0, len(values), size=(10000, len(values)))].mean(axis=1)
    return [float(v) for v in np.quantile(draws, [.025, .975])]


class DryTeacher:
    """Synthetic labels only; runs the same validation/reservation path."""
    def generate(self, history, task, visible_context):
        payload = json.loads(visible_context['observation'])['payload']
        commands = payload['admissible_commands_now']
        command = next((c for c in commands if re.match(r'^(?:go to |open |examine |look$)', c)), '')
        value = {'student_plan': 'SYNTHETIC: seeking the task object.',
            'conditions': [{'condition': 'The object has been observed.', 'status': 'undetermined', 'evidence_steps': []}],
            'condition_hint': 'Your subgoal is to obtain the object. Its presence has no observational support. Observations have excluded the inspected contents so far.',
            'action_suggestion': {'command': command, 'reason': 'This can provide additional public information.'}}
        return TeacherResult(json.dumps(value), 'structured_repair_response', {
            'usage': {'input_tokens': 100, 'output_tokens': 100, 'reasoning_tokens': 0},
            'response_status': 'completed', 'returned_model_id': 'FAKE_NO_MODEL',
            'timestamp': utc_now(), 'cache_hit': False, 'billed_cost': 0})


class DryEnvironment:
    """Replays recorded public transitions; fake continuation always fails.

    This is explicitly synthetic and cannot establish real reset consistency.
    No hidden game content or stored audit blobs are read.
    """
    def __init__(self, original):
        self.original, self.seed, self.max_episode_steps = original, original.env_seed, 40
        self.game_hash, self.index = original.game_sha256, 0

    def reset(self, task_id):
        self.index = 0
        return self._result(self.original.initial)

    def _result(self, step):
        # Synthetic original must have matching synthetic audit hashes.
        return deepcopy(step['visible']), step['reward'], step['terminated'], step['truncated'], {'won': False}

    def step(self, action):
        self.index += 1
        if self.index <= 10:
            return self._result(self.original.steps[self.index - 1])
        return ({'observation': 'Synthetic continuation: no target observed.', 'admissible_commands': ['look']},
                0.0, False, self.index >= 40, {'won': False})

    def close(self):
        pass


class DryPolicy:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)

    def generate(self, history, task, visible_context):
        write_json(self.log_dir / f'{len(history):02d}.json',
                   {'messages': build_messages(history, task, visible_context), 'model': 'FAKE_NO_MODEL'}, exclusive=True)
        return TeacherResult('Thought: Synthetic continuation.\nAction: look', 'look', {})
