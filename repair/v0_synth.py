"""Distance-stratified seed-only factory, including composite G5.

The private worker reuses synth_validation's real planner/executor. Witnesses
and PDDL never leave this validation boundary for policy construction.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
import json
import multiprocessing
import os
from pathlib import Path
import random
import subprocess
import sys

from repair import synth_tasks as factory
from repair.align import normalized_distance
from repair.v0_common import BINS, QUOTAS, ROOT, bind, check_inputs, checked_k, completed, finish, phase, read
from sed.storage import digest, file_hash, write_json

ERRORS = ('refused_action', 'wrong_object', 'useless_search', 'wrong_receptacle')


def candidates(parent, seeds, generator, bin_name, seed):
    rng = random.Random(digest([seed, parent.task_id, generator, bin_name]))
    if generator == 'G3':
        for index in range(4 * len(parent.actions)):
            error = ERRORS[index % 4]
            ident = digest(['v0', parent.task_id, error, index, seed])[:24]
            game = deepcopy(parent.game)
            game.update(walkthrough=[], solvable=False)
            yield game, dict(id=ident, task_id='train/synth/' + ident, generator='G3',
                parent_seed_task_id=parent.task_id, task_text=factory.task_text(game),
                fingerprint=factory.fingerprint(parent.group, game), error_type=error, prefix_index=index,
                edit=dict(error_type=error, prefix_index=index), pddl_diff='', grammar_diff='', new_task=False)
        return
    for game, row in factory.generate_candidates(parent, seeds, 'G2' if generator == 'G5' else generator, rng):
        if game is None:
            yield game, row
            continue
        typ = row['fingerprint']['tuple'][0]
        if generator in ('G2', 'G5'):
            same = typ == factory.group_parts(parent.group)[0]
            if same != (bin_name == 'mid'):
                continue
            # Mid changes the object or receptacle, not merely a grammar rule.
            if same and row['fingerprint']['group'] == parent.group:
                continue
        if generator != 'G5':
            yield game, row
            continue
        # A cross-type goal and relocation of that NEW goal's object. No
        # synthetic Seed object is created, even internally.
        facts = factory.init_facts(game)
        objects = factory.fact_map(facts, 'objectType')
        receptacles = factory.fact_map(facts, 'receptacleType')
        pairs = factory.legal_pairs(seeds)
        options = [(obj, dest, ot, rt) for obj, ot in sorted(objects.items())
                   if ot == row['edit']['object_type'] + 'Type' and ('pickupable', obj) in facts
                   for dest, rt in sorted(receptacles.items()) if (ot, rt) in pairs
                   and factory.fact_map(facts, 'inReceptacle').get(obj) != dest]
        rng.shuffle(options)
        for obj, dest, ot, rt in options:
            edit = dict(goal=deepcopy(row['edit']), object=obj, destination=dest,
                        destination_receptacle_type=rt[:-4], goal_task_type=typ,
                        pair_evidence=pairs[(ot, rt)])
            try:
                changed = factory.relocate(game, obj, dest)
            except ValueError as exc:
                yield None, dict(generator='G5', parent_seed_task_id=parent.task_id,
                                 edit=edit, rejection=str(exc))
                continue
            yield changed, factory.candidate_record(parent, changed, row['fingerprint']['group'], 'G5', edit)


def edit_kind(row):
    """Expansion restrictions refer to seed edits, never the synthetic parent."""
    kind = dict(generator=row['generator'], bin=row['bin'])
    if row['generator'] in ('G2', 'G5'):
        kind['goal_task_type'] = row['task_type']
    elif row['generator'] == 'G1':
        kind['destination_receptacle_type'] = row['edit']['destination_receptacle_type']
    else:
        kind['error_type'] = row['edit']['error_type']
    return kind


def annotate(row, game, bin_name):
    row = deepcopy(row)
    row['bin'] = bin_name
    if game is not None:
        row['task_type'] = row['fingerprint']['tuple'][0]
        row['edit']['goal_task_type'] = row['task_type']
        if row['generator'] == 'G1':
            types = factory.fact_map(factory.init_facts(game), 'receptacleType')
            row['edit']['destination_receptacle_type'] = types[row['edit']['destination']][:-4]
    return row


def validate_candidate(row, parent, directory, source_root, seed, parent_cache, env_factory=None, execution_factory=None):
    """Every acceptance has a replayed seed and two matching successful witnesses."""
    from repair.synth_validation import Execution, ValidationFailure, validate_game, validate_prefix
    from train.dataset import EMPTY_THOUGHT
    execution = (execution_factory or Execution)(seed, **({'env_factory': env_factory} if env_factory else {}))
    validation, rejection = {}, None
    try:
        if file_hash(parent.provenance['game_path']) != parent.provenance['game_sha256']:
            raise ValidationFailure('parent_game_hash_changed')
        if parent.task_id not in parent_cache:
            execution.reset(source_root, parent.task_id, 'verify_original_successful_seed_actions')
            execution.replay(parent.actions)
            parent_cache[parent.task_id] = deepcopy(execution.current)
        parent_episode = parent_cache[parent.task_id]
        if not parent_episode['steps'] or parent_episode['steps'][-1].get('won') is not True:
            raise ValidationFailure('original_seed_actions_do_not_win')
        job = dict(row, parent_actions=parent.actions)
        if row['generator'] == 'G3':
            # Validate on the original seed; the later model run uses only the
            # authorized synthetic alias and verifies this recorded prefix.
            job['task_id'] = parent.task_id
            validation = validate_prefix(job, directory, source_root, execution, parent_episode)
            prefix_path = directory / validation['prefix_path']
            prefix = read(prefix_path)
            events = [json.loads(line) for line in Path(parent.provenance['events_path']).read_text().splitlines()]
            if any(e.get('action_raw') is not None and e.get('environment_steps') != 1 for e in events):
                raise ValidationFailure('seed_parser_turns_cannot_be_silently_dropped')
            seed_replies = [e['action_raw'] for e in events if e.get('environment_steps') == 1]
            n = prefix['error_action_offset']
            row['edit'].update(error_action_offset=n, injected_actions=prefix['prefix_actions'][n:],
                               prefix_actions_sha256=digest(prefix['prefix_actions']))
            prefix.update(task_id=row['task_id'], prefix_replies=seed_replies[:n] + [
                EMPTY_THOUGHT + '\nAction: ' + a for a in prefix['prefix_actions'][n:]],
                prefix_turns_context_only=True)
            write_json(prefix_path, prefix)
            row.update(prefix_path=validation['prefix_path'], prefix_sha256=file_hash(prefix_path),
                       item_fingerprint=digest([row['fingerprint']['problem_sha256'], validation['state_sha256']]))
            row['fingerprint']['recovery_state_sha256'] = validation['state_sha256']
            witness = next(e for e in execution.episodes if e['purpose'] == 'controlled_error_recovery_repeat')
        else:
            validation = validate_game(job, directory, source_root, execution)
            row['item_fingerprint'] = row['fingerprint']['problem_sha256']
            witness = next(e for e in execution.episodes if e['purpose'] == 'witness_second_reset')
        validation['behavioural_distance'] = normalized_distance(parent.actions, [s['action'] for s in witness['steps']])
        rejection = validation.get('failure_reason')
    except ValidationFailure as exc:
        rejection = str(exc)
    except Exception as exc:
        rejection = 'validation_exception:' + type(exc).__name__
    finally:
        execution.close()
    validation['environment_transitions'] = execution.transitions
    path = directory / 'witnesses' / (row['id'] + '.json')
    write_json(path, dict(never_for_training=True, episodes=execution.episodes, validation=validation,
                         rejection=rejection, parent_seed_task_id=parent.task_id))
    row.update(validation=validation, witness_path=str(path.relative_to(directory)), witness_sha256=file_hash(path),
               status='rejected' if rejection else 'valid')
    if rejection:
        row['rejection'] = rejection
    else:
        row['behavioural_distance'] = validation['behavioural_distance']
        row['witness_nll'] = None  # No student/model call is permitted in the factory.
    return row


class SeedFactory:
    def __init__(self, parent, seeds, directory, root, seed, heldout, *, excluded=(), env_factory=None):
        self.parent, self.seeds, self.directory = parent, seeds, Path(directory)
        self.root, self.seed, self.heldout = root, seed, set(heldout)
        self.seen, self.env_factory, self.parent_cache = set(excluded), env_factory, {}
        self.seed_hashes = {factory.problem_hash(s.game) for s in seeds}
        self.streams, self.records = {}, []

    def next_valid(self, generator, bin_name, restriction=None):
        key = generator, bin_name, digest(restriction)
        if key not in self.streams:
            self.streams[key] = iter(candidates(self.parent, self.seeds, generator, bin_name, self.seed))
        for game, raw in self.streams[key]:
            row = annotate(raw, game, bin_name)
            if restriction and (game is None or edit_kind(row) != restriction):
                continue
            row.setdefault('id', digest([self.parent.task_id, generator, bin_name, row.get('edit')])[:24])
            row.setdefault('new_task', generator != 'G3')
            result_path = self.directory / 'candidates' / (row['id'] + '.json')
            if result_path.exists():
                row = read(result_path)
                for k in ('witness_path', 'game_path', 'prefix_path'):
                    if k in row and file_hash(self.directory / row[k]) != row[k.replace('_path', '_sha256')]:
                        raise ValueError('V0 candidate evidence changed on resume')
                for path, expected in row.get('validation_journals', {}).items():
                    if file_hash(self.directory / path) != expected:
                        raise ValueError('V0 validation transition journal changed on resume')
            else:
                reason = row.get('rejection')
                if game is not None:
                    reason = reason or ('heldout_group' if row['fingerprint']['group'] in self.heldout else None)
                    if generator != 'G3':
                        reason = reason or factory.rejection_reason(row, set(), self.seed_hashes, self.seen)
                    row['game_path'] = f"data_root/{row['task_id']}/game.tw-pddl"
                    write_json(self.directory / row['game_path'], game)
                    row['game_sha256'] = file_hash(self.directory / row['game_path'])
                if reason:
                    row.update(status='rejected', rejection=reason)
                else:
                    if self.env_factory is None:
                        from repair.v0_validation import isolated_validation
                        row = isolated_validation(row, self.parent, self.directory, self.root, self.seed, self.parent_cache)
                    else:
                        row = validate_candidate(row, self.parent, self.directory, self.root,
                                                 self.seed, self.parent_cache, self.env_factory)
                    if row['status'] == 'valid' and row['item_fingerprint'] in self.seen:
                        row.update(status='rejected', rejection='duplicate_effective_state')
                # Rejected games remain audit evidence, outside the runtime root.
                if row['status'] == 'rejected' and row.get('game_path'):
                    source = self.directory / row['game_path']
                    target = self.directory / 'rejected' / row['id'] / 'game.tw-pddl'
                    target.parent.mkdir(parents=True, exist_ok=True)
                    source.replace(target)
                    row['game_path'] = str(target.relative_to(self.directory))
                write_json(result_path, row, exclusive=True)
            self.records.append(row)
            if row['status'] == 'valid' and row['item_fingerprint'] not in self.seen:
                self.seen.add(row['item_fingerprint'])
                return deepcopy(row)
        return None


def allocate_seed(factory_instance, *, quotas=QUOTAS, probes=True):
    """Alternate G1/G3. Exhaust a generator before making an explicit fallback."""
    accepted, allocation = [], []
    for bin_name, quota in quotas.items():
        for slot in range(quota + int(probes)):
            probe = probes and slot == 0  # Reserve a probe before training selection.
            index = slot - int(probes)
            preferred = ('G1' if probe or index % 2 == 0 else 'G3') if bin_name == 'near' else (
                'G5' if bin_name == 'very_far' else 'G2')
            row = factory_instance.next_valid(preferred, bin_name)
            fallback = None
            if row is None and bin_name == 'near':
                fallback = 'G3' if preferred == 'G1' else 'G1'
                row = factory_instance.next_valid(fallback, bin_name)
            allocation.append(dict(bin=bin_name, slot=slot, heldout_probe=probe,
                preferred_generator=preferred, fallback_generator=fallback,
                fallback_reason='preferred_generator_has_no_remaining_valid_candidate' if fallback else None,
                item_id=row['id'] if row else None, status='filled' if row else 'candidate_space_exhausted'))
            if row:
                row.update(heldout_probe=probe, split='holdout' if probe else 'train', sealed=probe,
                           group_id='synth/' + row['item_fingerprint'], allocation=allocation[-1])
                accepted.append(row)
    return accepted, allocation


def seed_worker(job):
    from repair.synth_validation import offline
    offline()
    request, task_id = job
    seeds, _ = factory.load_seeds(request['seeds'], request['data_root'])
    parent = next(s for s in seeds if s.task_id == task_id)
    directory = Path(request['out']) / 'parents' / digest(task_id)[:16]
    sf = SeedFactory(parent, seeds, directory, request['data_root'], request['seed'], request['heldout'])
    accepted, allocation = allocate_seed(sf)
    result = dict(parent_seed_task_id=task_id, tasks=accepted, allocation=allocation, candidates=sf.records,
                  asset_root=str(directory.resolve()))
    write_json(directory / 'result.json', result)
    print(json.dumps(dict(parent=task_id, accepted=len(accepted), attempts=len(sf.records))), flush=True)
    return result


def stats(tasks, candidates, allocation, seed_ids):
    return dict(K=len(seed_ids), unique_seed_instances=len(set(seed_ids)),
        seed_task_groups=len({factory.group_name(t) for t in seed_ids}),
        training_items=sum(not t['heldout_probe'] for t in tasks), heldout_probes=sum(t['heldout_probe'] for t in tasks),
        by_bin={b: dict(training=sum(t['bin'] == b and not t['heldout_probe'] for t in tasks),
                           probes=sum(t['bin'] == b and t['heldout_probe'] for t in tasks),
                           by_generator=dict(Counter(t['generator'] for t in tasks if t['bin'] == b))) for b in BINS},
        by_generator=dict(Counter(t['generator'] for t in tasks)),
        rejections=dict(Counter(t['rejection'] for t in candidates if t.get('rejection'))),
        unfilled_slots=sum(a['item_id'] is None for a in allocation),
        environment_steps=sum(t.get('validation', {}).get('environment_transitions', 0) for t in candidates),
        recovery_situations=sum(t['generator'] == 'G3' for t in tasks),
        new_executable_games=sum(t['generator'] != 'G3' for t in tasks),
        teacher_calls=0, student_calls=0, teacher_tokens=0, training_exposure=0)


def synthesis_freeze(data_root, main_checkout, split_freeze=None):
    """Allow exact frozen inputs across worktrees, never weaker exclusions.

    Absolute source paths are part of the historical manifest hash. A supplied
    freeze may retain those paths only when every source content hash and the
    complete exclusion set still match a fresh identity-only scan.
    """
    current = factory.freeze_heldout(data_root, main_checkout, ROOT)
    if split_freeze is None:
        return current
    frozen = read(split_freeze)
    def contents(value):
        return Counter((Path(s['path']).name, s['sha256'], s['groups'], s['identity_only'])
                       for s in value['sources'])
    if ({k: v for k, v in frozen.items() if k != 'sources'} !=
            {k: v for k, v in current.items() if k != 'sources'} or contents(frozen) != contents(current)):
        raise ValueError('V0 supplied split freeze differs from current held-out identities or source contents')
    check_inputs({s['path']: s['sha256'] for s in frozen['sources'] if Path(s['path']).is_file()})
    for source in frozen['sources']:
        path = Path(source['path'])
        if not path.is_file():
            names = sorted(p.name for p in path.iterdir() if p.is_dir() and factory.GROUP_RE.fullmatch(p.name))
            if factory.digest(names) != source['sha256']:
                raise ValueError('V0 frozen held-out directory identities changed')
    return frozen


def synthesize(*, seeds_path, data_root, output, main_checkout, seed=0, workers=4, resume=False,
               split_freeze=None):
    output, data_root, main_checkout = map(lambda p: Path(p).resolve(), (output, data_root, main_checkout))
    if seed != 0 or workers < 1:
        raise ValueError('V0 first round requires seed 0 and positive CPU workers')
    for protected in (data_root, main_checkout):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError('V0 output overlaps read-only input')
    seeds, source = factory.load_seeds(seeds_path, data_root)
    k = checked_k(source['K'])
    if len(seeds) != k:
        raise ValueError('V0 K differs from executed seed trajectory count')
    if any(s.provenance['environment_seed'] != seed for s in seeds):
        raise ValueError('V0 seed trajectories use a different environment seed')
    freeze = synthesis_freeze(data_root, main_checkout, split_freeze)
    inputs = bind([seeds_path, *[p for s in seeds for p in (
        s.provenance['game_path'], s.provenance['episode_path'], s.provenance['events_path'])],
        *[s['path'] for s in freeze['sources'] if Path(s['path']).is_file()]])
    directory = output / 'synth'
    phase(directory, dict(command='synth', seed=seed, workers=workers, quotas=QUOTAS, probes_per_bin=1,
        inputs=inputs, data_root=str(data_root), split_freeze=freeze, K=k,
        validation_seconds_per_candidate=120, validation_timeout_policy='unknown feasibility; retain partial transitions',
        G5_definition='cross-type seed goal plus relocation of the new goal object; explicit user specification'), resume=resume)
    if completed(directory):
        return read(directory / 'stats.json')
    write_json(directory / 'split_freeze.json', freeze)
    request = dict(seeds=str(Path(seeds_path).resolve()), data_root=str(data_root), out=str(directory),
                   seed=seed, workers=workers, heldout=freeze['groups'], seed_task_ids=[s.task_id for s in seeds])
    write_json(directory / 'request.json', request)
    # Same offline, credential-free subprocess convention as synth_tasks.py.
    worker_env = dict(PATH=os.defpath, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1',
        PYTHONHASHSEED='0', PYTHONPATH=str(ROOT / 'src') + os.pathsep + str(ROOT),
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    subprocess.run([sys.executable, '-m', 'repair.v0_synth', '--request', str(directory / 'request.json')],
                   cwd=directory, env=worker_env, check=True, timeout=7200)
    tasks, all_candidates, allocation, seen = [], [], [], set()
    for parent in seeds:
        result = read(directory / 'parents' / digest(parent.task_id)[:16] / 'result.json')
        all_candidates.extend(result['candidates'])
        allocation.extend(result['allocation'])
        for row in result['tasks']:
            if row['item_fingerprint'] in seen:
                all_candidates.append(dict(row, status='rejected', rejection='duplicate_across_seed_parents'))
                next(a for a in allocation if a['item_id'] == row['id'])['item_id'] = None
                continue
            seen.add(row['item_fingerprint'])
            # Publish independent runtime copies, never symlinks into read-only inputs.
            for key in ('game_path', 'prefix_path', 'witness_path'):
                if key not in row:
                    continue
                relative = row[key]
                target = directory / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((Path(result['asset_root']) / relative).read_bytes())
            tasks.append(row)
    manifest = dict(schema='v0-synthetic-tasks/1', seed_task_ids=[s.task_id for s in seeds], tasks=tasks,
        split_freeze_sha256=file_hash(directory / 'split_freeze.json'), source_seed_manifest_sha256=file_hash(seeds_path),
        seed_sources={s.task_id: s.provenance for s in seeds}, K=k, seed=seed,
        group_rule='effective initial state/goal or recovery-state fingerprint; native groups remain in fingerprint.group',
        witness_policy='private validation only; never a policy/label input')
    write_json(directory / 'attempts.json', all_candidates)
    write_json(directory / 'allocation.json', allocation)
    write_json(directory / 'manifest.json', manifest)
    report = stats(tasks, all_candidates, allocation, manifest['seed_task_ids'])
    report['synthetic_manifest_sha256'] = file_hash(directory / 'manifest.json')
    write_json(directory / 'stats.json', report)
    finish(directory, **report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    request = read(parser.parse_args().request)
    jobs = [(request, task_id) for task_id in request['seed_task_ids']]
    if request['workers'] == 1:
        for job in jobs:
            seed_worker(job)
    else:
        with ProcessPoolExecutor(max_workers=request['workers'], mp_context=multiprocessing.get_context('spawn')) as pool:
            list(pool.map(seed_worker, jobs))


if __name__ == '__main__':
    main()
