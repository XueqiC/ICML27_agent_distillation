"""P0 train-side pair freeze. No discovery of evaluation directories or assets.

All policy inputs use the existing ReAct whitelist and decision_context. Audit
references are separate; an episode repair is not proof of a local action pair.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
import subprocess

from repair.patch import Patch, compile_patch
from repair.records import TrajectoryRecord
from repair.v0_common import read, inside
from sed.provenance import utc_now, versions
from sed.react import build_messages, action_outcome
from sed.storage import digest, file_hash

SCHEMA = 'p0-pairs/3'
SPLIT_RULE = ('Task-instance groups: sorted unique eligible group_id; '
              'random.Random(0).sample(groups, len(groups)); for each sorted CATEGORIES, '
              'add the first unchosen group containing that category until diagnosis has >=4 '
              'pairs of that category; all eligible pairs of chosen groups are diagnosis, '
              'all remaining eligible pairs are test.')
REVISION = 'ee0ef6023621cff504d758262d4e04895a5af4a2'
MODEL = 'google/gemma-4-E4B-it'
DEFAULT_ROUND = '/home/xueqi/hq/projects/ICML27_v0_runs/v0_k20_r1'
DEFAULT_BASE = str(Path.home() / '.cache/huggingface/hub/models--google--gemma-4-E4B-it/snapshots' / REVISION)
DEFAULT_ADAPTER = ('/home/xueqi/hq/projects/ICML27_agent_distillation/reports/runs/baselines_loni/'
                   'e4b-tsft-k20-e32-lr5e-5/stages/train-sft/attempt-000/output/checkpoint')
CATEGORIES = ('goal_misread', 'missing_precondition', 'refused_action',
              'search_incomplete_or_loop', 'wrong_object_or_target')
ROOT = Path(__file__).resolve().parents[1]


def provenance(config=None):
    paths = list((ROOT / 'repair').glob('p0_*.py')) + list((ROOT / 'tools').glob('p0_*.py'))
    paths += [ROOT / p for p in ('src/sed/react.py', 'src/sed/visibility.py', 'src/sed/thinking.py',
        'repair/execute.py', 'repair/records.py', 'repair/align.py', 'repair/patch.py',
        'train/dataset.py', 'harness/alfworld_env/environment.py', 'repair/v0_transport.py',
        'teachers/openai_teacher.py', 'teachers/capped.py')]
    return dict(created=utc_now(), code_revision=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        code_sha256={str(p.relative_to(ROOT)): file_hash(p) for p in sorted(paths)},
        dependencies=versions(), effective_config=config or {}, training_exposure=0,
        K=20, teacher_alias='gpt-6-luna', teacher_returned_model_id=None,
        teacher_timestamp=None, student_revision=REVISION)


def stratify(pairs):
    groups = defaultdict(list)
    for p in pairs:
        groups[p['group_id']].append(p)
    ids = sorted(groups)
    order = random.Random(0).sample(ids, len(ids))
    selected, counts = set(), Counter()
    for category in sorted(CATEGORIES):
        while counts[category] < 4:
            group = next((g for g in order if g not in selected
                          and any(p['category'] == category for p in groups[g])), None)
            if group is None:
                raise ValueError('Insufficient category pairs: ' + category)
            selected.add(group)
            counts.update(p['category'] for p in groups[group])
    output = [dict(p, split='diagnosis' if p['group_id'] in selected else 'test')
              for p in sorted(pairs, key=lambda p: (p['category'], p['pair_id']))]
    assert_disjoint(output)
    return output


def assert_disjoint(pairs):
    for key in ('group_id', 'task_id', 'state'):
        def values(split):
            return {p['state']['sha256'] if key == 'state' else p[key]
                    for p in pairs if p['split'] == split}
        if values('diagnosis') & values('test'):
            raise ValueError('Diagnosis/test overlap: ' + key)


def resolve_category(breaking_points, t):
    """First exact match; otherwise nearest, with ties going to the later step."""
    if not breaking_points:
        raise ValueError('Missing breaking points')
    bp = min(breaking_points, key=lambda b: (abs(b['step'] - t), -b['step']))
    source = ('at_divergence' if bp['step'] == t else
              'nearest_following' if bp['step'] > t else 'nearest_preceding')
    return {k: bp[k] for k in ('step', 'category', 'reason')}, source, abs(bp['step'] - t)


def local_repair(original, review, *, repaired_execution, executors, prefix_length=0,
                 review_rounds=1):
    """First parsed-action divergence of the FINAL verified execution.

    Round-0 compilation supplies audit metadata only, never the repair action.
    """
    patch = Patch.from_json(review['patch'])
    if patch.original_trajectory_id != original.trajectory_id:
        raise ValueError('Review does not refer to original attempt')
    compiled = compile_patch([s['action_raw'] for s in original.steps], patch)
    round_zero = [action_outcome(a if 'Action:' in a else 'Action: ' + a, [])[0] for a in compiled]
    repaired = TrajectoryRecord.from_json(repaired_execution)
    oa = [s['action_parsed'] for s in original.steps]
    ea = [s['action_parsed'] for s in repaired.steps]
    if len(executors) != len(ea):
        raise ValueError('Final execution/executor lengths differ')
    t = next((i for i, (a, b) in enumerate(zip(oa, ea)) if a != b), None)
    detail = dict(context_only_prefix_length=prefix_length, original_actions=len(oa),
                  executed_actions=len(ea), compiled_actions=len(round_zero),
                  review_rounds=review_rounds, state_match='unavailable')
    if t is None:
        reason = ('no_alternative_action_at_divergence' if len(ea) < len(oa) else
                  'no_original_action_at_divergence' if len(ea) > len(oa) else 'no_action_divergence')
        return t, None, reason, detail
    mismatches = [i for i in range(t) if original.steps[i]['visible'] != repaired.steps[i]['visible']]
    initial_equal = (original.initial is not None and repaired.initial is not None
                     and original.initial['visible'] == repaired.initial['visible'])
    detail.update(prefix_equal=oa[:t] == ea[:t], initial_visible_equal=initial_equal,
                  mismatching_prefix_steps=mismatches,
                  state_match='verified' if initial_equal and not mismatches else 'mismatch',
                  executor_at_divergence=executors[t],
                  round_zero_action_at_divergence=round_zero[t] if t < len(round_zero) else None)
    reason = ('divergence_not_teacher_patch' if executors[t] != 'patch' else
              'divergence_in_context_only_prefix' if t < prefix_length else
              'repaired_prefix_state_mismatch' if detail['state_match'] == 'mismatch' else None)
    return t, ea[t], reason, detail


def extract_pairs(round_dir=DEFAULT_ROUND):
    root = Path(round_dir).resolve()
    paths = [root / p for p in ('review/records.json', 'attempt/records.json',
                               'attempt/plan.json', 'synth/manifest.json')]
    reviews, attempts, plan, manifest = map(read, paths)
    # The manifest is an authorized input, but sealed task assets are never read.
    tasks = {r['task_id']: r for r in manifest['tasks'] if r['split'] == 'train'
             and r['sealed'] is False and r['heldout_probe'] is False}
    attempts = {r['attempt_id']: r for r in attempts if r['heldout_probe'] is False}
    if manifest['K'] != 20 or len(manifest['seed_task_ids']) != 20:
        raise ValueError('Expected original K20 source')
    identity = plan['identity']
    if identity['H'] != 40 or identity['serving']['enable_thinking'] is not False:
        raise ValueError('Unexpected round horizon/thinking')
    if (identity['adapter']['base_model'], identity['adapter']['revision']) != (MODEL, REVISION):
        raise ValueError('Student substitution forbidden')
    # Pin reused behavior against the original round, without changing old code.
    for rel in ('src/sed/react.py', 'src/sed/visibility.py', 'repair/execute.py',
                'repair/align.py', 'harness/alfworld_env/environment.py'):
        if file_hash(ROOT / rel) != identity['code_sha256'][rel]:
            raise ValueError('Harness code differs from round: ' + rel)
    pairs, excluded = [], []
    first_categories, resolved_categories, state_matches = Counter(), Counter(), Counter()
    inputs = {str(p): file_hash(p) for p in paths}
    for row in reviews:
        if row['verified'] is not True or row['success'] is not True:
            continue
        aid = row['attempt_id']
        attempt = attempts[aid]
        task = tasks[row['task_id']]
        if attempt['task_id'] != row['task_id'] or attempt['success'] is not False:
            raise ValueError('R bucket requires a failed original execution')
        if row['execution']['success'] is not True:
            raise ValueError('Repair lacks environment success')
        path = inside(root, f'review/failures/{aid}/round-0.json')
        inputs[str(path)] = file_hash(path)
        round_zero = read(path)
        review = round_zero['review']
        first_categories[review['breaking_points'][0]['category']] += 1
        original = TrajectoryRecord.from_json(attempt['execution'])
        if original.max_steps != 40:
            raise ValueError('Unexpected episode horizon')
        t, ar, reason, detail = local_repair(original, review,
            prefix_length=attempt['prefix_length'], repaired_execution=row['execution'],
            executors=row['executors'], review_rounds=len(row['rounds']))
        state_matches[detail['state_match']] += 1
        category_info = {}
        reasons = [reason] if reason else []
        if t is not None:
            bp, source, distance = resolve_category(review['breaking_points'], t)
            resolved_categories[bp['category']] += 1
            category_info = dict(category=bp['category'], review=bp,
                                 category_source=source, category_distance=distance)
            if bp['category'] == 'other':
                reasons.append('category_other')
            elif bp['category'] not in CATEGORIES:
                raise ValueError('Unexpected category')
        if reasons:
            excluded.append(dict(attempt_id=aid, task_id=row['task_id'], step=t,
                reason=reasons[0], reasons=reasons, state_match=detail['state_match'],
                alignment_audit=detail, review_path=str(path), review_sha256=inputs[str(path)],
                repaired_execution_sha256=digest(row['execution']), **category_info))
            continue
        context = original.decision_context(t)
        messages = build_messages(context['history'], {'task_id': row['task_id']}, context['visible_context'])
        pairs.append(dict(pair_id=aid, attempt_id=aid, step=t, **category_info,
            task_id=row['task_id'], parent_seed_task_id=task['parent_seed_task_id'],
            group_id=task['group_id'], decoding=attempt['decoding'], sampling_seed=attempt['sampling_seed'],
            state=dict(messages=messages, context=context, sha256=digest(messages)),
            a_f=original.steps[t]['action_parsed'], a_r=ar,
            original_action_raw=original.steps[t]['action_raw'],
            alignment='final_execution_first_divergence', alignment_audit=detail, state_match=detail['state_match'],
            reproduction=dict(status='not_run', greedy_action=None, matches=None),
            audit=dict(original_trajectory_id=original.trajectory_id,
                original=original.to_json(), review_path=str(path), review_sha256=inputs[str(path)],
                patch=review['patch'], env_seed=original.env_seed,
                repaired_execution_sha256=digest(row['execution']),
                review_round_status=round_zero['status'],
                verification_scope='final repaired episode; not an independent local-action success label',
                environment_version=original.environment_version, game_sha256=original.game_sha256)))
    pairs = stratify(pairs)
    splits = {p['pair_id']: p['split'] for p in pairs}
    diag_tasks = {p['task_id'] for p in pairs if p['split'] == 'diagnosis'}
    test_tasks = {p['task_id'] for p in pairs if p['split'] == 'test'}
    diag_states = {p['state']['sha256'] for p in pairs if p['split'] == 'diagnosis'}
    test_states = {p['state']['sha256'] for p in pairs if p['split'] == 'test'}
    parents = {s: {p['parent_seed_task_id'] for p in pairs if p['split'] == s}
               for s in ('diagnosis', 'test')}
    shared_parents = parents['diagnosis'] & parents['test']
    split_counts = {s: dict(pairs=sum(p['split'] == s for p in pairs),
        categories={c: sum(p['split'] == s and p['category'] == c for p in pairs) for c in CATEGORIES},
        category_source=dict(Counter(p['category_source'] for p in pairs if p['split'] == s)),
        state_match=dict(Counter(p['state_match'] for p in pairs if p['split'] == s)),
        unique_instances=len({p['task_id'] for p in pairs if p['split'] == s}),
        task_groups=len({p['group_id'] for p in pairs if p['split'] == s}),
        unique_states=len({p['state']['sha256'] for p in pairs if p['split'] == s}),
        parent_seed_task_ids=len(parents[s]),
        pairs_with_shared_parent=sum(p['split'] == s and p['parent_seed_task_id'] in shared_parents for p in pairs))
        for s in ('diagnosis', 'test')}
    value = dict(schema=SCHEMA, seed=0, diagnosis_min_per_category=4, pairs=pairs, excluded=excluded,
        provenance=provenance(dict(round_dir=str(root), H=40, dtype='bfloat16', seed=0)),
        inputs=inputs, student=identity['adapter'], serving=identity['serving'],
        manifest_path=str(paths[-1]), manifest_sha256=file_hash(paths[-1]),
        counts=dict(verified_repairs=sum(first_categories.values()),
            first_listed_categories=dict(first_categories), resolved_categories=dict(resolved_categories),
            categories=dict(Counter(p['category'] for p in pairs)),
            category_source=dict(Counter(p['category_source'] for p in pairs)),
            state_match=dict(state_matches), eligible_state_match=dict(Counter(p['state_match'] for p in pairs)),
            excluded=len(excluded), excluded_by_reason=dict(Counter(p['reason'] for p in excluded)),
            exclusion_reason_occurrences=dict(Counter(r for p in excluded for r in p['reasons'])),
            eligible=len(pairs), pairs=len(pairs), diagnosis=split_counts['diagnosis']['pairs'],
            test=split_counts['test']['pairs'], splits=split_counts,
            alignment=dict(Counter(p['alignment'] for p in pairs)),
            unique_instances=len({p['task_id'] for p in pairs}),
            task_groups=len({p['group_id'] for p in pairs}),
            unique_states=len(diag_states | test_states),
            diagnosis_test_state_overlap=len(diag_states & test_states),
            diagnosis_test_instance_overlap=len(diag_tasks & test_tasks),
            diagnosis_test_group_overlap=len({p['group_id'] for p in pairs if p['split'] == 'diagnosis'}
                & {p['group_id'] for p in pairs if p['split'] == 'test'}),
            diagnosis_test_parent_seed_task_id_overlap=len(shared_parents),
            source_decoding=dict(Counter(p['decoding'] for p in pairs)),
            diagnosis_alignment=dict(Counter(p['alignment'] for p in pairs if p['split'] == 'diagnosis'))),
        split_sha256=digest(splits), split_rule=SPLIT_RULE)
    value['sha256'] = digest(value)
    return value


def load_pairs(path):
    value = read(path)
    if value.get('schema') != SCHEMA or value.get('sha256') != digest({k: v for k, v in value.items() if k != 'sha256'}):
        raise ValueError('Pair freeze changed')
    if value['split_sha256'] != digest({p['pair_id']: p['split'] for p in value['pairs']}):
        raise ValueError('Split freeze changed')
    assert_disjoint(value['pairs'])
    if any(p.get('alignment') != 'final_execution_first_divergence' for p in value['pairs']):
        raise ValueError('Final execution first-divergence alignment required')
    for source, expected in value['inputs'].items():
        if file_hash(source) != expected:
            raise ValueError('Pair source changed')
    return value


FRAMINGS = ('original_thought', 'greedy_thought', 'no_thought')


def thought_split(raw):
    """Preserve the exact public reply split; never request hidden reasoning."""
    raw = raw if isinstance(raw, str) else ''
    before, separator, after = raw.partition('\nAction:')
    available = raw.startswith('Thought:') and bool(separator)
    return dict(prefix=before + '\n' if available else '',
                split_index=len(before) if separator else None,
                separator=separator, before_action=before, after_separator=after,
                raw_sha256=digest(raw),
                fallback_reason=None if available else 'missing_thought_or_action_delimiter')


def state_digest(state):
    # Same rendered history can have different recorded/current Thoughts.
    return digest({k: state[k] for k in ('text', 'spans', 'origin_pair_id', 'original_thought', 'thought_override') if k in state})


def render_state(pair, tokenizer):
    messages = pair['state']['messages']
    text = tokenizer.apply_chat_template(messages, tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
    spans, cursor, obs_index = [], 0, -1
    for msg in messages:
        content = msg['content']
        start = text.find(content, cursor)
        if start < 0:
            if msg['role'] == 'assistant':
                # Native template may remove embedded channel markup. Locate
                # Thought in rendered text separately below, never rewrite input.
                continue
            raise ValueError('Cannot align native rendered user message')
        cursor = start + len(content)
        if msg['role'] != 'user':
            continue
        visible = json.loads(content)
        for key in ('observation', 'admissible_commands'):
            value = json.dumps(visible[key], ensure_ascii=False, separators=(',', ':'))
            at = content.index('"' + key + '":') + len(key) + 3
            if content[at:at + len(value)] != value:
                raise ValueError('Visible JSON span mismatch')
            label = ('latest_observation' if obs_index == pair['step'] - 1 else f'observation_step_{obs_index}') if key == 'observation' else f'admissible_commands_step_{obs_index}'
            spans.append(dict(id=label, label=label, start=start + at, end=start + at + len(value)))
            if key == 'observation':
                match = re.search(r'Your task is to:[^"\\]*', value)
                if match:
                    spans.append(dict(id=f'task_goal_{obs_index}', label='task_goal',
                                      start=start + at + match.start(), end=start + at + match.end()))
        obs_index += 1
    for i, match in enumerate(re.finditer(r'Thought:[^\n]*|<\|channel>thought\n<channel\|>', text)):
        spans.append(dict(id=f'Thought_{i}', label='Thought', start=match.start(), end=match.end()))
    state = dict(text=text, spans=spans, pair_id=pair['pair_id'],
                 origin_pair_id=pair['pair_id'], counterfactual=False,
                 original_thought=thought_split(pair['original_action_raw']))
    state['sha256'] = state_digest(state)
    return state


def teacher_pair(pair, state):
    """Explicit prompt whitelist: no audit row, verifier labels or patch suffix."""
    initial = pair['state']['context']['history']
    visible = initial[0]['visible'] if initial else pair['state']['context']['visible_context']
    match = re.search(r'Your task is to:\s*([^\n]+)', visible['observation'])
    return dict(pair_id=pair['pair_id'], task=match[1] if match else visible['observation'],
        state=state['text'], state_id=state.get('canonical_state_id', state['sha256']), spans=state['spans'],
        a_f=pair['a_f'], a_r=pair['a_r'],
        framing=state.get('framing', 'no_thought'),
        framing_record=state.get('framing_record'), review=dict(
            {k: pair['review'][k] for k in ('step', 'category', 'reason')}, decision_step=pair['step']))


def collateral_sample(round_dir, count=200, seed=0):
    rows = read(Path(round_dir) / 'attempt/records.json')
    candidates = []
    for row in sorted(rows, key=lambda r: r['attempt_id']):
        if row['success'] is not True or row['heldout_probe'] is not False:
            continue
        record = TrajectoryRecord.from_json(row['execution'])
        for t in range(row['prefix_length'], len(record.steps)):
            candidates.append((row['attempt_id'], record, t))
    sample = random.Random(seed).sample(candidates, count)
    return [dict(pair_id=f'S-{aid}-{t}', task_id=r.task_id, step=t,
        original_action_raw=r.steps[t]['action_raw'],
        state=dict(context=r.decision_context(t), messages=build_messages(
            r.decision_context(t)['history'], {'task_id': r.task_id}, r.decision_context(t)['visible_context'])))
        for aid, r, t in sample]
