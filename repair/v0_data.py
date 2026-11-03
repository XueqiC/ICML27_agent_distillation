"""V0 decision rows with the existing V2 public rendering and L checks.

The small decision loop is adapted from repair.supervision.build_decisions:
V0 labels patch decisions only, includes every accepted successful decision,
and keeps injected recovery prefixes context-only. No existing builder changes.
"""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil

from repair.labels import prefix_checks
from repair.prompts import rationale_prompt
from repair.schemas import SchemaError, parse_label
from repair.supervision import (ByteBoundTokenizer, GUIDANCE, InvalidExample,
    _grounded, _example, action_line, inspect_steps, public_visible, thought, validate_example)
from repair.v0_common import bind, check_inputs, read
from repair.v0_transport import TeacherOutputTruncated, TeacherTimeout
from sed.provenance import utc_now
from sed.react import build_messages
from sed.storage import atomic_write, canonical_bytes, digest, file_hash, write_json


class RowPolicy:
    def __init__(self, manifest):
        self.manifest = manifest

    def check(self, task_id, source):
        row = self.manifest.authorize(task_id)
        return row['group_id'], row['task_type']


def decision_rows(episode, manifest, *, bucket, episode_ref, labeler=None, audit=None, tokenizer=None):
    task = manifest.authorize(episode['task_id'])
    if bucket not in ('R', 'S', 'G'):
        raise ValueError('V0 root bucket must be R, S or G')
    execution = episode['execution']
    if (episode['success'] is not True or execution['success'] is not True
            or execution['termination_reason'] != 'SUCCESS' or execution['task_id'] != episode['task_id']):
        raise ValueError('V0 requires an environment-verified successful episode')
    rows = inspect_steps(execution)
    origins = episode['executors']
    branch = episode['prefix_length']
    if len(origins) != len(rows) or any(x not in ('student', 'patch', 'handback-student') for x in origins):
        raise ValueError('V0 executor coverage differs')
    if bucket == 'S' and any(x != 'student' for x in origins):
        raise ValueError('V0 S bucket must contain only the student own decisions')
    if type(branch) is not int or not 0 <= branch <= len(rows):
        raise ValueError('V0 invalid context-only prefix length')
    history, prefix, examples, excluded, rewrites = [], [], [], Counter(), []
    initial = public_visible(execution['initial']['visible'])['observation']
    broken = False
    policy = RowPolicy(manifest)
    for i, row in enumerate(rows):
        action, raw = row['action'], row['raw']
        messages = build_messages(history, {}, row['visible'])
        edit = origins[i] == 'patch'
        reply, source, check = raw, 'student_thought', 'passed'
        if edit and i >= branch and action is not None and not broken:
            reply, source, check = action_line(raw), 'action_only_no_label', 'not_available'
            prompt = rationale_prompt({'instruction': initial}, initial, prefix, action)
            log = dict(episode_id=episode['episode_id'], episode_ref=episode_ref, step=i,
                       request=prompt.to_json(), raw_output=None, prefix_sha256=digest(messages))
            if labeler is not None and row['legal'] and not row['repeated_refused']:
                try:
                    raw_label = labeler(prompt)  # Budget/transport failures must stop the build.
                    log['raw_output'] = raw_label
                    label = parse_label(raw_label, tokenizer or ByteBoundTokenizer())
                    checks = prefix_checks({'instruction': initial}, initial, prefix, action, label, i)
                    log['grounding'] = checks
                    candidate = 'Thought: ' + label.reasoning + '\n' + action_line(raw)
                    if not checks['structural_pass'] or not _grounded(candidate, messages) or GUIDANCE.search(label.reasoning):
                        raise InvalidExample('label_grounding_failed')
                    reply, source, check = candidate, 'teacher_rationale', 'passed'
                except TeacherOutputTruncated as exc:
                    source, check = 'action_only_truncated_label', 'teacher_output_truncated'
                    log['teacher_attempts'] = exc.attempts
                    excluded['truncated_rationale_label'] += 1
                except TeacherTimeout as exc:
                    source, check = 'action_only_timeout_label', 'teacher_timeout'
                    log['teacher_attempts'] = exc.attempts
                    excluded['timeout_rationale_label'] += 1
                except (SchemaError, InvalidExample) as exc:
                    source, check = 'action_only_invalid_label', str(exc)
                    excluded['invalid_rationale_label'] += 1
            log.update(label_source=source, label_check=check)
            if audit:
                audit(log)
            rewrites.append(dict(step=i, reply_sha256=digest(reply), label_source=source, label_check=check))
        target = reply
        if action is not None:
            rationale = thought(reply)
            target = ('Thought: ' + rationale + '\n' if rationale else '') + 'Action: ' + action
            if not edit and not rationale:
                source = 'action_only_no_thought'
            if not broken and (not _grounded(target, messages) or GUIDANCE.search(rationale)):
                target, source, check = 'Action: ' + action, 'action_only_ungrounded_student', 'label_instances_not_public'
        reason = ('prefix_context_only' if i < branch else 'unparseable_prefix' if broken or action is None else
                  'refused_or_inadmissible' if not row['legal'] else
                  'repeated_refused_action' if row['repeated_refused'] else None)
        if reason:
            excluded[reason] += 1
        else:
            evidence = dict(executor=origins[i], local_event=next(iter(row['local_events']), None),
                episode_success=True, label_source=source, label_check=check,
                history_rewrites=deepcopy(rewrites), public_prefix_sha256=digest(messages))
            # Validate with V2 before adding the requested V0 extension.
            example = _example('JOINT', 'v0_' + bucket, episode['task_id'], episode_ref, i,
                               messages + [dict(role='assistant', content=target)], evidence, policy)
            validate_example(example, policy)
            example.update(arm='V0', example_id=digest(['V0', bucket, episode['episode_id'], i]),
                v0=dict(bucket=bucket, episode_id=episode['episode_id'], generator=task['generator'],
                        bin=task['bin'], edit_decision=edit, synthetic_manifest_sha256=manifest.sha256))
            examples.append(example)
        history.append(dict(visible=row['visible'], reply=reply))
        prefix.append(dict(action_index=i, reply=reply, action=action, observation=row['after_visible']['observation']))
        broken |= action is None
    return examples, dict(excluded)


def choose_episodes(attempts, repairs, *, expanded=False):
    selected = []
    for episode in repairs:
        if episode.get('verified') and episode['success'] is True:
            selected.append(('G' if expanded else 'R', episode))
    successes = {}
    for episode in attempts:
        if not episode['heldout_probe'] and episode['success'] is True:
            successes.setdefault(episode['task_id'], []).append(episode)
    for tid in sorted(successes):
        episode = min(successes[tid], key=lambda r: (len(r['execution']['steps']), r['episode_id']))
        selected.append(('G' if expanded else 'S', episode))
    return selected


def check_replay(directory, seed_ids, theta0=None):
    """Read a sealed V2 CONTINUE bundle without changing any row/mask bytes."""
    directory = Path(directory).resolve()
    manifest = read(directory / 'manifest.json')
    raw_hash = file_hash(directory / 'examples.jsonl')
    if (manifest['schema'] != 'v2-train-interchange/1' or manifest['arm'] != 'CONTINUE'
            or (directory / 'examples.sha256').read_text().strip() != raw_hash):
        raise ValueError('V0 replay requires a sealed V2 CONTINUE bundle')
    check_inputs(manifest['inputs'])
    rows = [json.loads(line) for line in (directory / 'examples.jsonl').read_text().splitlines()]
    if (len(rows) != 20 or sorted(r['task_id'] for r in rows) != sorted(seed_ids)
            or digest([r['example_id'] for r in rows]) != manifest['example_ids_sha256']
            or any(r['kind'] != 'trajectory' or r['source'] != 'k20_replay'
                   or r['evidence']['episode_success'] is not True for r in rows)):
        raise ValueError('V0 replay differs from executed K20 seed receipts')
    if theta0 and manifest['theta0']['adapter_weights_sha256'] != theta0['adapter_weights_sha256']:
        raise ValueError('V0 replay and served theta0 differ')
    return manifest, rows


def dataset_counts(rows, replay_rows):
    return {bucket: dict(rows=len(selected), episodes=len({r['v0']['episode_id'] for r in selected}),
                        by_bin=dict(Counter(r['v0']['bin'] for r in selected)),
                        by_generator=dict(Counter(r['v0']['generator'] for r in selected)))
            for bucket in ('R', 'G', 'S', 'D')
            for selected in [[r for r in rows if r['v0']['bucket'] == bucket] if bucket != 'D' else [
                dict(v0=dict(episode_id=r['example_id'], bin='seed', generator='seed')) for r in replay_rows]]}


def publish_dataset(directory, rows, *, manifest, replay, inputs, metadata, inherited=None):
    directory, replay = Path(directory), Path(replay)
    replay_manifest, replay_rows = check_replay(replay, manifest.seeds, metadata.get('theta0'))
    rows = deepcopy(rows)
    if inherited is not None:
        rows = deepcopy(inherited) + rows
    for row in rows:
        task = manifest.authorize(row['task_id'])
        row['v0']['synthetic_manifest_sha256'] = manifest.sha256
        if (row['arm'] != 'V0' or row['v0']['generator'] != task['generator'] or row['v0']['bin'] != task['bin']
                or row['group_id'] != task['group_id'] or row['task_type'] != task['task_type']
                or set(row['v0']) != {'bucket', 'episode_id', 'generator', 'bin', 'edit_decision', 'synthetic_manifest_sha256'}
                or type(row['v0']['edit_decision']) is not bool):
            raise ValueError('V0 dataset identity differs from synthetic manifest')
    rows.sort(key=lambda r: r['example_id'])
    if len({r['example_id'] for r in rows}) != len(rows):
        raise ValueError('V0 duplicate example IDs')
    inputs = {**inputs, **replay_manifest['inputs'], **bind([manifest.path, replay / 'manifest.json',
                                                         replay / 'examples.jsonl', replay / 'examples.sha256'])}
    check_inputs(inputs)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'manifest.json').exists():
        raise ValueError('V0 dataset is already sealed')
    target = directory / 'replay'
    if target.exists():
        if any((target / name).read_bytes() != (replay / name).read_bytes()
               for name in ('manifest.json', 'examples.jsonl', 'examples.sha256')):
            raise ValueError('V0 partial replay differs on resume')
    else:
        # Includes any nested audit artifacts from V2; exact source bytes.
        shutil.copytree(replay, target)
    atomic_write(directory / 'examples.jsonl', b''.join(canonical_bytes(r) + b'\n' for r in rows))
    value = dict(metadata, schema='v0-train-interchange/1', arm='V0', created=utc_now(),
        examples_sha256=file_hash(directory / 'examples.jsonl'), replay_manifest_sha256=file_hash(target / 'manifest.json'),
        splits_sha256=replay_manifest['splits_sha256'], theta0=replay_manifest['theta0'], inputs=inputs,
        synthetic_manifest=str(manifest.path), counts=dataset_counts(rows, replay_rows),
        K=20, unique_seed_instances=len(manifest.seeds), training_exposure=0,
        extension='Explicit 2026-11-02 user contract: required v0.bin and G5; older five-key train reader needs updating')
    check_inputs(inputs)
    write_json(directory / 'manifest.json', value, exclusive=True)
    return value


def load_dataset(directory):
    directory = Path(directory).resolve()
    manifest = read(directory / 'manifest.json')
    if manifest['schema'] != 'v0-train-interchange/1' or file_hash(directory / 'examples.jsonl') != manifest['examples_sha256']:
        raise ValueError('V0 dataset seal mismatch')
    if file_hash(directory / 'replay/manifest.json') != manifest['replay_manifest_sha256']:
        raise ValueError('V0 replay seal mismatch')
    check_inputs(manifest['inputs'])
    return manifest, [json.loads(line) for line in (directory / 'examples.jsonl').read_text().splitlines()]


def consumer_dataset_hash(directory, rows, *, seed_bin=False):
    """Current train-side v0_data.load_dataset identity, including virtual D rows."""
    directory = Path(directory)
    replay = [json.loads(line) for line in (directory / 'replay/examples.jsonl').read_text().splitlines()]
    for row in replay:
        row['arm'] = 'V0'
        row['v0'] = dict(bucket='D', episode_id='seed/' + row['example_id'], generator='seed',
                         edit_decision=False, synthetic_manifest_sha256='')
        if seed_bin:
            row['v0']['bin'] = 'seed'
    return digest(dict(manifest=file_hash(directory / 'manifest.json'), rows=rows + replay,
                       replay_manifest=file_hash(directory / 'replay/manifest.json')))
