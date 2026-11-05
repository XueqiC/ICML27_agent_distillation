"""E0.4 public-history supervision; no environment execution or training.

The interchange is deliberately independent of the E0.5 loader. Evidence is
never a prompt: only visible_observation and executed visible replies enter it.
"""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import re

from repair.facts import FactTable, INSTANCE
from repair.labels import instance_ids, prefix_checks
from repair.prompts import rationale_prompt
from repair.schemas import SchemaError, parse_label
from sed.react import ActionParseError, PROMPT_TEMPLATE, build_messages, parse_action
from sed.storage import canonical_bytes, digest, file_hash, JsonlLedger, write_json
from sed.thinking import split_thinking
from sed.visibility import visible_observation

SCHEMA = 'v2-train-interchange/1'
ARMS = ('JOINT', 'JOINT_COMPLETE_ONLY', 'PREFIX_EXPERT', 'OFFLINE', 'CONTINUE')
GUIDANCE = re.compile(r'\b(?:guidance|hints?|suggest(?:ed|ion|ions|s|ing)?)\b', re.I)
EVENTS = ('target_seen', 'target_taken', 'hands_freed', 'transform_done',
          'placed', 'lamp_used_with_target', 'task_success')


class InvalidExample(ValueError):
    """A stable exclusion reason, never raw model/SDK exception text."""


def public_visible(value):
    result = visible_observation(value)
    # G appends exactly this suffix. Retain all bytes before it, including any
    # original trailing whitespace. Also accept an inline recorded suffix.
    result['observation'] = re.split(r'\n\nGuidance:|Guidance:', result['observation'], maxsplit=1)[0]
    return result


def thought(reply):
    lines = re.findall(r'^[ \t]*Thought[ \t]*:[ \t]*([\s\S]*?)(?=^[ \t]*Action[ \t]*:|\Z)', reply, re.M)
    return '\n'.join(lines).strip()


def action_line(reply):
    return re.findall(r'^[ \t]*Action[ \t]*:[^\r\n]*', reply, re.M)[-1]


def group_id(task_id):
    parts = task_id.split('/')
    if len(parts) != 3 or parts[0] != 'train' or any(p in ('', '.', '..') for p in parts):
        raise InvalidExample('non_train_task')
    return '/'.join(parts[:2])


class SplitPolicy:
    """Exact trial whitelist plus group/name-level evaluation exclusions."""
    def __init__(self, splits, k20_ids, task_types=None):
        pool = splits['sets']['V2_pool']
        if pool.get('training_eligible') is not True:
            raise InvalidExample('pool_not_training_eligible')
        self.pool, self.k20 = set(pool['task_ids']), set(k20_ids)
        if len(self.pool) != len(pool['task_ids']) or not self.pool or not self.k20:
            raise InvalidExample('invalid_split_membership')
        if 'list_sha256' in pool and pool['list_sha256'] != digest(pool['task_ids']):
            raise InvalidExample('split_list_hash')
        self.forbidden = set()
        for name, view in splits['sets'].items():
            if name in {'U_dev', 'U_mech', 'D_probe', 'D_unseen_quick', 'dev140'}:
                self.forbidden.update(view.get('group_ids', []))
                self.forbidden.update(t.rsplit('/', 1)[0] for t in view.get('task_ids', []))
        for source in splits.get('exclusion_sources', []):
            if source['category'] in {'U_dev', 'U_mech', 'D_probe', 'D_unseen_quick', 'dev140'}:
                self.forbidden.update(source['group_ids'])
        # V2's additional name exclusions do not retroactively change theta0's
        # K20 view. TRAIN and valid_seen with the same name are distinct groups.
        self.pool_forbidden_names = {g.split('/')[-1] for g in self.forbidden}
        self.pool_forbidden_names.update(splits.get('task_name_exclusions', {}).get('excluded_task_names', []))
        pg, kg = ({group_id(t) for t in ids} for ids in (self.pool, self.k20))
        if pg & kg or len(pg) != len(self.pool):
            raise InvalidExample('split_group_overlap')
        self.types = task_types or {}
        for t in self.pool | self.k20:
            if group_id(t) in self.forbidden or (t in self.pool and group_id(t).split('/')[-1] in self.pool_forbidden_names):
                raise InvalidExample('evaluation_group_overlap')

    def check(self, task_id, source):
        group = group_id(task_id)
        if group in self.forbidden or (source != 'k20_replay' and group.split('/')[-1] in self.pool_forbidden_names):
            raise InvalidExample('evaluation_task')
        if task_id not in (self.k20 if source == 'k20_replay' else self.pool):
            raise InvalidExample('task_outside_frozen_view')
        return group, self.types.get(task_id, group.split('/')[-1].split('-')[0])


def fact_state_hash(table):
    """State, not accumulating evidence timestamps/counters, defines a repeat."""
    def values(value):
        if isinstance(value, dict):
            return {k: values(v) for k, v in value.items() if k != 'evidence_steps'}
        if isinstance(value, list):
            return [values(v) for v in value]
        return value
    state = table.to_json()
    keys = ('goal', 'holding', 'object_locations', 'receptacles', 'state_changes', 'lamps_on', 'conditions')
    return digest({**values({k: state[k] for k in keys}), 'current': table.current,
                   'refused_actions': sorted(table.refused_actions)})


def _type(instance):
    return instance.rsplit(' ', 1)[0]


def inspect_steps(execution):
    """Derive local deltas solely from the executed public record."""
    table = FactTable(public_visible(execution['initial']['visible']))
    visible, rows, seen, refused = public_visible(execution['initial']['visible']), [], set(), set()
    observed_instances = instance_ids(visible['observation'])
    for i, step in enumerate(execution['steps']):
        raw = split_thinking(step['action_raw'])[0]
        try:
            action = parse_action(raw)
        except ActionParseError:
            action = None
        if action != step['action_parsed']:
            raise InvalidExample('executed_action_mismatch')
        after_visible = public_visible(step['visible'])
        rejected = (step['action_outcome'] == 'inadmissible' or
                    bool(re.search(r'\bnothing happens\.', after_visible['observation'], re.I)))
        legal = (action is not None and action in visible['admissible_commands'] and
                 step['action_outcome'] == 'legal' and not rejected and step.get('environment_steps') == 1)
        state_hash, before = fact_state_hash(table), table.to_json()
        repeated_refused = action in refused
        repeat = (action, state_hash) in seen
        seen.add((action, state_hash))
        if rejected:
            refused.add(action)
        table.update({'action_parsed': action, 'action_outcome': 'legal' if legal else 'inadmissible',
                      'visible': after_visible})
        after, events = table.to_json(), []
        goal = before['goal']['value']
        if legal and goal:
            objtype = goal['object_type']
            new_objects = set(after['object_locations']) - set(before['object_locations']) - observed_instances
            if (action.startswith(('go to ', 'open ', 'examine ')) or action == 'look') and any(
                    _type(o) == objtype for o in new_objects):
                events.append('target_seen')
            take = re.fullmatch(r'take (' + INSTANCE + r') from (' + INSTANCE + ')', action)
            move = re.fullmatch(r'(?:move|put) (' + INSTANCE + r') (?:to|in|on) (' + INSTANCE + ')', action)
            transform = re.fullmatch(r'(heat|cool|clean) (' + INSTANCE + r') with (' + INSTANCE + ')', action)
            if take and _type(take[1]) == objtype and take[1] not in (before['holding']['value'] or []) and take[1] in (after['holding']['value'] or []):
                events.append('target_taken')
            if (move and _type(move[1]) != objtype and
                    before['conditions']['hands_free']['status'] == 'known_unsatisfied' and
                    after['conditions']['hands_free']['status'] == 'satisfied'):
                events.append('hands_freed')  # Qualified by the NEXT take below.
            if transform and _type(transform[2]) == objtype:
                state = {'heat': 'hot', 'cool': 'cool', 'clean': 'clean'}[transform[1]]
                if (not before['state_changes'].get(transform[2], {}).get(state, {}).get('value') and
                        after['state_changes'].get(transform[2], {}).get(state, {}).get('value') is True):
                    events.append('transform_done')
            if move and _type(move[1]) == objtype and _type(move[2]) == goal['target_receptacle_type']:
                required = goal['required_state']
                if (before['object_locations'].get(move[1], {}).get('value') != move[2] and
                        after['object_locations'].get(move[1], {}).get('value') == move[2] and
                        (required is None or before['state_changes'].get(move[1], {}).get(required, {}).get('value') is True)):
                    events.append('placed')
            if re.fullmatch(r'use desklamp \d+', action):
                if any(_type(o) == objtype and
                       not before['state_changes'].get(o, {}).get('lit', {}).get('value') and
                       after['state_changes'].get(o, {}).get('lit', {}).get('value') is True
                       for o in before['holding']['value'] or []):
                    events.append('lamp_used_with_target')
        if legal and execution['success'] is True and i == len(execution['steps']) - 1:
            events.append('task_success')
        rows.append(dict(step=i, action=action, raw=raw, legal=legal, refused=rejected,
                         repeated_refused=repeated_refused, no_progress_repeat=repeat,
                         state_sha256=state_hash, before_sha256=digest(before), after_sha256=digest(after),
                         local_events=events, goal=goal, visible=visible, after_visible=after_visible))
        observed_instances.update(instance_ids(after_visible['observation']))
        visible = after_visible
    for i, row in enumerate(rows):
        if 'hands_freed' in row['local_events']:
            pattern = r'take ' + re.escape(row['goal']['object_type']) + r' \d+ from '
            following = next((s for s in rows[i + 1:] if re.match(pattern, s['action'] or '')), None)
            if following is None or not following['legal']:
                row['local_events'].remove('hands_freed')
            else:
                row['next_legal_target_take'] = following['step']
    return rows


class ByteBoundTokenizer:
    """Conservative L-label cap when a local tokenizer is unavailable."""
    def encode(self, value, **kwargs):
        return list(value.encode('utf-8'))


def _public_ids(messages):
    # Prior Thought claims cannot make an invented instance become grounded.
    text = []
    for message in messages:
        if message['role'] == 'user':
            text.append(message['content'])
        elif message['role'] == 'assistant':
            text.append(parse_action(message['content']))
    return instance_ids('\n'.join(text))


def _grounded(reply, messages):
    return not (instance_ids(thought(reply)) - _public_ids(messages))


def _example(arm, source, task_id, ref, step, messages, evidence, policy, kind='decision'):
    group, task_type = policy.check(task_id, source)
    assistants = [i for i, m in enumerate(messages) if m['role'] == 'assistant']
    return dict(example_id=digest([arm, source, task_id, ref, step]), arm=arm, source=source,
                kind=kind, task_id=task_id, group_id=group, task_type=task_type, messages=messages,
                supervised_assistant_indices=assistants[-1:] if kind == 'decision' else assistants,
                weight=1.0, evidence=dict(episode_ref=ref, step=step, **evidence))


def validate_example(example, policy, *, expected_messages=None, source_record=None):
    group, task_type = policy.check(example['task_id'], example['source'])
    if (example['arm'] not in ARMS or example['group_id'] != group or example['task_type'] != task_type
            or example['weight'] != 1.0 or example['kind'] not in ('decision', 'trajectory')):
        raise InvalidExample('example_contract')
    e = example['evidence']
    if example['example_id'] != digest([example['arm'], example['source'], example['task_id'], e['episode_ref'], e['step']]):
        raise InvalidExample('example_id')
    messages = example['messages']
    if expected_messages is not None and canonical_bytes(messages) != canonical_bytes(expected_messages):
        raise InvalidExample('executed_prefix_mismatch')
    if not messages or messages[0] != {'role': 'system', 'content': PROMPT_TEMPLATE} or len(messages) % 2 != 1:
        raise InvalidExample('message_roles')
    assistants = list(range(2, len(messages), 2))
    wanted = assistants[-1:] if example['kind'] == 'decision' else assistants
    if not wanted or example['supervised_assistant_indices'] != wanted:
        raise InvalidExample('supervision_mask')
    for i, m in enumerate(messages):
        if set(m) != {'role', 'content'} or type(m['content']) is not str or 'Guidance:' in m['content']:
            raise InvalidExample('guidance_or_message_fields')
        if i == 0:
            continue
        if m['role'] != ('user' if i % 2 else 'assistant'):
            raise InvalidExample('message_roles')
        if i % 2:
            value = json.loads(m['content'])
            if canonical_bytes(visible_observation(value)).decode() != m['content']:
                raise InvalidExample('user_visible_whitelist')
        else:
            try:
                action = parse_action(m['content'])
            except ActionParseError:
                raise InvalidExample('unparseable_prefix') from None
            if split_thinking(m['content'])[1]:
                raise InvalidExample('hidden_thinking')
            # Trajectories retain every main-grid teacher turn unchanged. These
            # two decision-only checks become informational manifest counts.
            if example['kind'] == 'decision' and i in wanted:
                if action not in json.loads(messages[i - 1]['content'])['admissible_commands']:
                    raise InvalidExample('supervised_action_inadmissible')
                if not _grounded(m['content'], messages[:i]):
                    raise InvalidExample('ungrounded_rationale')
    if source_record is not None:
        execution = source_record['execution']
        end = e['step']
        if len(messages) != 3 + 2 * end or execution['task_id'] != example['task_id']:
            raise InvalidExample('executed_prefix_mismatch')
        for turn in range(end + 1):
            visible = execution['initial']['visible'] if turn == 0 else execution['steps'][turn - 1]['visible']
            if messages[1 + 2 * turn]['content'] != canonical_bytes(public_visible(visible)).decode():
                raise InvalidExample('executed_observation_mismatch')
            if parse_action(messages[2 + 2 * turn]['content']) != execution['steps'][turn]['action_parsed']:
                raise InvalidExample('executed_action_mismatch')
            if turn < end:
                actual = messages[2 + 2 * turn]['content']
                raw = split_thinking(execution['steps'][turn]['action_raw'])[0]
                decisions = {s['step']: s for s in source_record.get('steps', [])}
                executor = decisions.get(turn, {}).get('executor', 'student')
                dependent = executor in ('teacher', 'student_after_guidance') or GUIDANCE.search(thought(raw))
                if example['arm'] == 'PREFIX_EXPERT' or not dependent:
                    if actual != raw:
                        raise InvalidExample('executed_history_reply_mismatch')
                else:
                    rewrites = {s['step']: s for s in e.get('history_rewrites', [])}
                    if (action_line(actual) != action_line(raw) or
                            rewrites.get(turn, {}).get('reply_sha256') != digest(actual)):
                        raise InvalidExample('history_rewrite_mismatch')
    return example


def build_decisions(record, *, arm, episode_ref, policy, labeler=None, tokenizer=None, audit=None):
    """Return examples and reason counts; malformed/foreign records fail closed.

    A labeler receives ONLY a repair.prompts.Prompt. It returns raw label JSON.
    audit receives every requested label, raw response, check and rejection.
    """
    if arm not in ARMS[:3]:
        raise ValueError('Decision arm required')
    policy.check(record['task_id'], 'prefix_expert' if arm == 'PREFIX_EXPERT' else 'joint_local')
    excluded = Counter()
    if record.get('void') is not False or record.get('prefix_verified') is not True or record.get('status') != 'completed':
        return [], Counter({'void_or_unverified_episode': 1})
    if arm == 'PREFIX_EXPERT':
        if record['mode'] not in ('takeover', 'from_start', 'prefix'):
            return [], Counter({'wrong_mode': 1})
        if record['success'] is not True:
            return [], Counter({'unsuccessful_takeover': 1})
    elif record['mode'] != 'G':
        return [], Counter({'wrong_mode': 1})
    execution = record['execution']
    if execution['task_id'] != record['task_id'] or execution['success'] != record['success']:
        raise InvalidExample('execution_identity')
    if execution.get('termination_reason') == 'ENV_ERROR':
        return [], Counter({'environment_error': 1})
    if execution['success'] is True and execution.get('termination_reason') != 'SUCCESS':
        raise InvalidExample('success_not_environment_verified')
    rows = inspect_steps(execution)
    branch = record['branch_index']
    if type(branch) is not int or not 0 <= branch < len(rows):
        raise InvalidExample('branch_index')
    decisions = {r['step']: r for r in record.get('steps', []) if r.get('executed_reply') is not None}
    if arm != 'PREFIX_EXPERT':
        if len(decisions) != len([r for r in record.get('steps', []) if r.get('executed_reply') is not None]) or set(decisions) != set(range(branch, len(rows))):
            raise InvalidExample('missing_or_duplicate_executors')
        for i, d in decisions.items():
            if (split_thinking(d['executed_reply'])[0] != rows[i]['raw'] or
                    d['executed_action'] != rows[i]['action'] or
                    d['executor'] not in ('teacher', 'student', 'student_after_guidance')):
                raise InvalidExample('executed_reply_mismatch')
    initial = public_visible(execution['initial']['visible'])['observation']
    task = {'instruction': initial}
    history, prefix, examples, replacements = [], [], [], []
    broken_prefix = False
    for i, row in enumerate(rows):
        executor = ('teacher' if arm == 'PREFIX_EXPERT' and i >= branch else
                    decisions[i]['executor'] if i in decisions else 'student')
        raw, action = row['raw'], row['action']
        messages = build_messages(history, {}, row['visible'])
        dependent = executor in ('teacher', 'student_after_guidance') or bool(GUIDANCE.search(thought(raw)))
        reply, source, check = raw, 'teacher_reply' if arm == 'PREFIX_EXPERT' else 'student_thought', 'passed'
        if arm != 'PREFIX_EXPERT' and dependent and action is not None and not broken_prefix:
            reply, source, check = action_line(raw), 'action_only_no_label', 'not_available'
            prompt = rationale_prompt(task, initial, prefix, action)
            log = dict(episode_ref=episode_ref, step=i, executor=executor, request=prompt.to_json(),
                       prefix_sha256=digest(messages), raw_output=None)
            if labeler is not None:
                raw_label = labeler(prompt)  # Transport/budget failures abort, never become labels.
                log['raw_output'] = raw_label
                try:
                    label = parse_label(raw_label, tokenizer or ByteBoundTokenizer())
                    checks = prefix_checks(task, initial, prefix, action, label, i)
                    log['grounding'] = checks
                    candidate = 'Thought: ' + label.reasoning + '\n' + action_line(raw)
                    if not checks['structural_pass'] or not _grounded(candidate, messages) or GUIDANCE.search(label.reasoning):
                        raise InvalidExample('label_grounding_failed')
                    reply, source, check = candidate, 'teacher_rationale', 'passed'
                except (SchemaError, InvalidExample) as exc:
                    source, check = 'action_only_invalid_label', str(exc)
                    excluded['invalid_rationale_label'] += 1
            log.update(label_source=source, label_check=check)
            if audit is not None:
                audit(log)
            replacements.append(dict(step=i, label_source=source, label_check=check, reply_sha256=digest(reply)))
        target = reply
        if arm != 'PREFIX_EXPERT' and dependent and action is not None:
            target = ('Thought: ' + thought(reply) + '\n' if thought(reply) else '') + 'Action: ' + action
        if arm != 'PREFIX_EXPERT' and not dependent and action is not None:
            r = thought(raw)
            target = ('Thought: ' + r + '\n' if r else '') + 'Action: ' + action
            if not r:
                source = 'action_only_no_thought'
            if not broken_prefix and not _grounded(target, messages):
                target, source, check = 'Action: ' + action, 'action_only_ungrounded_student', 'label_instances_not_public'
        reason = None
        local = row['local_events'] if row['legal'] and not row['repeated_refused'] else []
        complete = execution['success'] is True and not row['no_progress_repeat']
        if arm == 'PREFIX_EXPERT' and i < branch:
            reason = 'student_prefix_context_only'
        elif not row['legal']:
            reason = 'refused_or_inadmissible' if row['refused'] or action else 'unparseable_action'
        elif arm != 'PREFIX_EXPERT' and row['repeated_refused']:
            reason = 'repeated_refused_action'
        elif arm != 'PREFIX_EXPERT' and not (complete or (arm == 'JOINT' and local)):
            reason = 'no_progress_repeat' if row['no_progress_repeat'] else 'no_qualifying_progress'
        if broken_prefix or action is None:
            reason = 'unparseable_prefix'
        if reason:
            excluded[reason] += 1
        else:
            src = 'prefix_expert' if arm == 'PREFIX_EXPERT' else 'joint_local' if local and arm == 'JOINT' else 'joint_complete'
            rendered = messages + [{'role': 'assistant', 'content': target}]
            evidence = dict(executor=executor, local_event=local[0] if local else None,
                local_events=local, episode_success=execution['success'], label_source=source, label_check=check,
                complete_success_eligible=complete, public_prefix_sha256=digest(messages),
                fact_state_sha256=row['state_sha256'], facts_before_sha256=row['before_sha256'],
                facts_after_sha256=row['after_sha256'], history_rewrites=deepcopy([r for r in replacements if r['step'] < i]),
                next_legal_target_take=row.get('next_legal_target_take'))
            example = _example(arm, src, record['task_id'], episode_ref, i, rendered, evidence, policy)
            try:
                validate_example(example, policy, source_record=record)
            except InvalidExample as exc:
                excluded[str(exc)] += 1
            else:
                examples.append(example)
        history.append({'visible': row['visible'], 'reply': reply})
        prefix.append(dict(action_index=i, reply=reply, action=action,
                           observation=row['after_visible']['observation']))
        broken_prefix |= action is None
    return examples, excluded


def trajectory_messages(episode):
    """The message-level loop of train.dataset.episode_examples, unchanged.

    That renderer exposes token arrays, not messages. Both paths call the shared
    build_messages; tests capture its encode_turn input for byte comparison.
    """
    if episode.get('schema_version') != 2 or not episode['steps']:
        raise InvalidExample('react_trajectory_required')
    history, visible = [], visible_observation(episode['initial']['visible'])
    for step in episode['steps']:
        reply, hidden = split_thinking(step['action_raw'])
        if hidden:
            raise InvalidExample('hidden_thinking')
        messages = build_messages(history, {}, visible)
        history.append({'visible': visible, 'reply': reply})
        visible = visible_observation(step['visible'])
    return messages + [{'role': 'assistant', 'content': reply}]


def build_trajectory(episode, *, arm, source, episode_ref, policy):
    if arm not in ARMS or source not in ('offline_teacher', 'k10_replay', 'k20_replay', 'k30_replay'):
        raise ValueError('Unknown trajectory arm/source')
    if not (episode.get('success') is True and episode.get('complete') is True and
            episode.get('status') == 'ok' and episode.get('split') == 'train' and episode.get('sealed') is False):
        raise InvalidExample('unverified_teacher_trajectory')
    messages = trajectory_messages(episode)
    example = _example(arm, source, episode['task_id'], episode_ref, None, messages,
        dict(executor='teacher', local_event=None, episode_success=True, label_source='teacher_reply',
             label_check='passed'), policy, kind='trajectory')
    return validate_example(example, policy, expected_messages=messages)


def trajectory_diagnostics(examples):
    """Count affected supervised turns per trajectory, never reject or rewrite.

    Grounding uses the same pre-turn public evidence as decision validation;
    future sightings and earlier Thought claims cannot ground an instance.
    A turn can contribute to both counts, but multiple unseen IDs count once.
    """
    diagnostics = {}
    for example in sorted(examples, key=lambda e: e['example_id']):
        if example['kind'] != 'trajectory':
            continue
        messages = example['messages']
        counts = dict(inadmissible_supervised_turns=0, not_yet_seen_instance_supervised_turns=0)
        for i in example['supervised_assistant_indices']:
            reply = messages[i]['content']
            counts['inadmissible_supervised_turns'] += int(
                parse_action(reply) not in json.loads(messages[i - 1]['content'])['admissible_commands'])
            counts['not_yet_seen_instance_supervised_turns'] += int(not _grounded(reply, messages[:i]))
        diagnostics[example['example_id']] = counts
    return diagnostics


class FakeLabelClient:
    """Explicit synthetic transport, never imports or calls the OpenAI client."""
    def __init__(self):
        from types import SimpleNamespace
        self.responses = SimpleNamespace(with_raw_response=self)

    def create(self, **kwargs):
        from types import SimpleNamespace
        result = SimpleNamespace(output_text=json.dumps(dict(reasoning='Act on the public evidence.',
            evidence_steps=[-1], unsupported=False)), model='FAKE_NO_MODEL', id='synthetic', status='completed',
            usage={'input_tokens': 100, 'output_tokens': 30, 'output_tokens_details': {'reasoning_tokens': 0}})
        return SimpleNamespace(status_code=200, headers={}, parse=lambda: result)


class RationaleLabeler:
    """Existing factory/structured L transport with shared per-attempt hard caps."""
    def __init__(self, config, task_id, directory, budget, *, dry_run=False, client=None):
        from repair.runtime import RepairTeacher, OUTPUT_CAPS
        from repair.takeover import _Reservations, _UsageLedger, _QuotaGuard
        from teachers.capped import from_config
        from teachers.openai_teacher import rate_limit_options
        from teachers.rate_limit import RateLimiter
        spec = config['models']['teacher']
        if spec['requested_name'] != 'gpt-6-luna' or spec['provider_model_id'] != 'gpt-6-luna':
            raise ValueError('L requires the configured gpt-6-luna teacher')
        self.directory, self.budget = Path(directory), budget
        self.task = dict(task_id=task_id, split='train', sealed=False)
        self.output_cap = OUTPUT_CAPS['rationale']
        self.teacher = from_config(config, output=self.directory, cache_dir=self.directory / 'cache',
            usage_ledger=self.directory / 'usage.jsonl', run_id=self.directory.name, cap=80,
            teacher_class=RepairTeacher, sampling={**spec['sampling'], 'max_output_tokens': self.output_cap})
        self.reservation = _Reservations(budget, self.output_cap)
        native = self.teacher.teacher
        native.reservation_budget = self.reservation
        native.ledger = _UsageLedger(native.ledger, budget)
        if dry_run or client is not None:
            native.client.client = FakeLabelClient() if dry_run else client
        original = native._client
        native._client = lambda: _QuotaGuard(original(), budget)
        options = rate_limit_options(spec, budget.path.parent / ('label_fake_rate_limit' if dry_run else 'label_teacher_rate_limit'))
        if dry_run:
            options.update(requests_per_minute=1_000_000, tokens_per_minute=1_000_000_000)
        native.limiter = RateLimiter(options.pop('rate_limit_dir'), spec['provider_model_id'], **options)

    def __call__(self, prompt):
        from teachers.stage import require_completed
        from teachers.openai_teacher import TeacherCallError
        from sed.provenance import utc_now
        row = dict(timestamp=utc_now(), request=prompt.to_json(), status='started')
        ledger = JsonlLedger(self.directory / 'requests.jsonl')
        ledger.append(row)
        try:
            self.budget.check()
            self.reservation.input_bound = len(canonical_bytes(prompt.messages())) + 16 * (len(prompt.messages()) + 1)
            result = self.teacher.generate([], self.task,
                {'observation': canonical_bytes(prompt.to_json()).decode(), 'admissible_commands': []})
            require_completed(result, allow_truncated=False)
            if result.metadata.get('cache_hit') or not result.raw_text.strip() or result.metadata['usage']['output_tokens'] > self.output_cap:
                raise TeacherCallError('Invalid L completion')
            ledger.append(dict(row, status='completed', raw_output=result.raw_text, metadata=result.metadata))
            return result.raw_text
        except Exception as exc:
            reason = self.budget.summary()['stop_reason'] or 'label_transport_failed'
            self.budget.halt(reason)
            ledger.append(dict(row, status='failed', error_type=type(exc).__name__, reason=reason))
            raise TeacherCallError('L halted; see sanitized label ledgers') from None


def dataset_manifest(examples, *, arm, inputs, splits_sha256, theta0, excluded, tokenizer=None,
                     tokenizer_info=None, code_revision=None, **extra):
    from sed.provenance import utc_now
    turns = [e['messages'][i]['content'] for e in examples for i in e['supervised_assistant_indices']]
    total = sum(len(tokenizer.encode(t, add_special_tokens=False)) if tokenizer else (len(t) + 3) // 4 for t in turns)
    return dict(schema=SCHEMA, arm=arm, created=utc_now(), code_revision=code_revision, inputs=dict(sorted(inputs.items())),
        counts=dict(by_source=dict(Counter(e['source'] for e in examples)), by_kind=dict(Counter(e['kind'] for e in examples)),
                    by_task_type=dict(Counter(e['task_type'] for e in examples)),
                    by_local_event=dict(Counter(v for e in examples for v in e['evidence'].get('local_events', []))),
                    examples=len(examples), unique_instances=len({e['task_id'] for e in examples}),
                    task_groups=len({e['group_id'] for e in examples})),
        supervised_turns=len(turns), supervised_tokens=total,
        trajectory_diagnostics=trajectory_diagnostics(examples),
        supervised_token_estimate=dict(value=total, method='local_tokenizer' if tokenizer else 'ceil_char/4',
            is_estimate=tokenizer is None, scope='assistant_text_only_before_chat_template', tokenizer=tokenizer_info),
        example_ids_sha256=digest([e['example_id'] for e in examples]), splits_sha256=splits_sha256, theta0=theta0,
        training_exposure_tasks=sorted({e['group_id'] for e in examples}),
        excluded=[dict(reason=k, count=v) for k, v in sorted(excluded.items()) if v], **extra)


def seal_dataset(directory, examples, *, policy, **metadata):
    """Call only in an exclusively reserved build directory; manifest seals last."""
    from sed.storage import atomic_write
    directory = Path(directory)
    examples = sorted(examples, key=lambda e: e['example_id'])
    if len({e['example_id'] for e in examples}) != len(examples):
        raise InvalidExample('duplicate_example_id')
    for e in examples:
        validate_example(e, policy)
    for path, expected in metadata['inputs'].items():
        if file_hash(path) != expected:
            raise InvalidExample('input_changed_during_build')
    manifest = dataset_manifest(examples, **metadata)
    atomic_write(directory / 'examples.jsonl', b''.join(canonical_bytes(e) + b'\n' for e in examples), exclusive=True)
    atomic_write(directory / 'examples.sha256', (file_hash(directory / 'examples.jsonl') + '\n').encode(), exclusive=True)
    write_json(directory / 'manifest.json', manifest, exclusive=True)
    return manifest
