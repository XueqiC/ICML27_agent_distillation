"""V0 expansion allocation and independent seed edits; input datasets are read-only."""
import argparse
from collections import defaultdict
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from repair import synth_tasks as factory
from repair.v0_common import ROOT, SyntheticManifest, bind, completed, finish, phase, read
from repair.v0_data import consumer_dataset_hash, load_dataset
from repair.v0_synth import SeedFactory, edit_kind
from sed.storage import digest, file_hash, write_json


def allocate(total, masses):
    """Largest remainder; lexical ties; never clip negative consistency."""
    if type(total) is not int or total < 0 or any(not math.isfinite(v) or v < 0 for v in masses.values()):
        raise ValueError('V0 allocation requires nonnegative finite mass')
    if total == 0:
        return {key: 0 for key in sorted(masses)}
    if not masses or sum(masses.values()) <= 0:
        raise ValueError('V0 no positive eligible expansion mass')
    exact = {key: total * value / sum(masses.values()) for key, value in masses.items()}
    counts = {key: math.floor(value) for key, value in exact.items()}
    for key in sorted(exact, key=lambda k: (-(exact[k] - counts[k]), k))[:total - sum(counts.values())]:
        counts[key] += 1
    return counts


def expansion_plan(dataset, rows, attempts, manifest, *, variant, groups=None, weights=None, seed=0):
    failed = sorted({r['task_id'] for r in attempts if not r['heldout_probe'] and r['success'] is False})
    for task_id in failed:
        manifest.authorize(task_id)
    episodes = {}
    for row in rows:
        if row['v0']['bucket'] == 'R':
            episodes.setdefault(row['v0']['episode_id'], row)
    representatives = {}
    for eid, row in episodes.items():
        episode = read(row['evidence']['episode_ref'])
        if episode['episode_id'] != eid or not episode['verified'] or episode['success'] is not True:
            raise ValueError('V0 expansion source is not a verified repair')
        representatives[eid] = episode
    eligible, masses, ignored = defaultdict(list), {}, []
    if variant == 'D-data':
        if groups is None or weights is None:
            raise ValueError('D-data requires groups.json and weights.json')
        group_doc, weight_doc = read(groups), read(weights)
        identities = {consumer_dataset_hash(dataset, rows, seed_bin=include_bin) for include_bin in (False, True)}
        if (group_doc['schema'] != 'v0-groups/1' or weight_doc['schema'] != 'v0-episode-weights/1'
                or weight_doc.get('status') != 'valid' or group_doc.get('smoke_subset') or weight_doc.get('smoke_subset')
                or any(group_doc[k] != weight_doc[k] for k in (
                    'dataset_sha256', 'adapter_sha256', 'parameter_names_sha256', 'features_index_sha256'))
                or group_doc['dataset_sha256'] not in identities
                or group_doc['adapter_sha256'] != read(Path(dataset) / 'manifest.json')['theta0']['adapter_weights_sha256']):
            raise ValueError('V0 group/weight source identity mismatch')
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in weight_doc['weights'].values()):
            raise ValueError('V0 invalid episode weights')
        for group in group_doc['groups']:
            gid = group['group_id']
            if any(group_doc['assignment'].get(eid) != gid for eid in group['members']):
                raise ValueError('V0 group membership mismatch')
            eligible[gid] = sorted(set(group['members']) & representatives.keys())
            if eligible[gid]:
                if any(eid not in weight_doc['weights'] for eid in eligible[gid]):
                    raise ValueError('V0 repaired episode missing weight')
                masses[gid] = group['consistency']
            else:
                ignored.append(gid)
    else:
        if variant != 'D-param':
            raise ValueError('Unknown V0 expansion variant')
        for eid, episode in representatives.items():
            categories = {p['category'] for r in episode['rounds'] if r.get('review')
                          for p in r['review']['breaking_points']}
            for category in categories:
                eligible[category].append(eid)
                masses[category] = 1.0
    # Keep every eligible repair, including ones not assigned an initial unit.
    # Fallback changes only the seed-edit source, never the group allocation.
    sources = {}
    for group in sorted(masses):
        sources[group] = []
        for eid in sorted(set(eligible[group])):
            task = manifest.authorize(episodes[eid]['task_id'])
            sources[group].append(dict(source_episode_id=eid,
                source_synthetic_task_id=task['task_id'], preferred_seed=task['parent_seed_task_id'],
                restriction=edit_kind(task)))
    counts = allocate(len(failed), masses)
    units = []
    for group, count in sorted(counts.items()):
        members = sorted(eligible[group], key=lambda eid: (digest([seed, group, eid]), eid))
        for index in range(count):
            eid = members[index % len(members)]
            task = manifest.authorize(episodes[eid]['task_id'])
            units.append(dict(unit=len(units), group=group, source_episode_id=eid,
                source_synthetic_task_id=task['task_id'], preferred_seed=task['parent_seed_task_id'],
                restriction=edit_kind(task)))
    return dict(variant=variant, failed_training_task_ids=failed, requested_items=len(failed),
        allocation=counts, units=units, group_sources=sources, ignored_groups_without_repaired_episodes=ignored,
        fallback_rule='after exhausting all K20 seeds, try other repairs in the same group/category by episode_id',
        rule='largest remainder over eligible repaired-episode groups/categories; deterministic lexical ties', seed=seed)


def allocation_summary(plan, execution):
    return dict(requested_items=plan['requested_items'],
        generated_items=sum(row['status'] == 'filled' for row in execution),
        fallback_count=sum(row.get('fallback_to') is not None for row in execution),
        fallback_attempted_count=sum(row.get('fallback_from') is not None for row in execution),
        unfilled_count=sum(row['status'] == 'unfilled_after_fallback' for row in execution))


def generate(request, *, env_factory=None):
    """Exhaust seed edits, then same-group repairs; record any remaining shortfall."""
    seeds, _ = factory.load_seeds(request['seeds'], request['data_root'])
    if (set(s.task_id for s in seeds) != set(request['seed_task_ids'])
            or any(s.task_id.startswith('train/synth/') for s in seeds)):
        raise ValueError('V0 expansion seeds differ from frozen K20')
    if request.get('seed_sources') and any(any(s.provenance[k] != request['seed_sources'][s.task_id][k]
            for k in ('game_sha256', 'episode_sha256', 'events_sha256', 'environment_seed')) for s in seeds):
        raise ValueError('V0 expansion seed execution/game receipts changed')
    out = Path(request['out'])
    seen = set(request['excluded_fingerprints'])
    factories = {s.task_id: SeedFactory(s, seeds, out / 'parents' / digest(s.task_id)[:16],
        request['data_root'], request['seed'], request['heldout'], excluded=seen, env_factory=env_factory) for s in seeds}
    selected, unit_log = [], []
    for unit in request['plan']['units']:
        original = {key: unit[key] for key in (
            'source_episode_id', 'source_synthetic_task_id', 'preferred_seed', 'restriction')}
        alternatives = sorted((s for s in request['plan'].get('group_sources', {}).get(unit['group'], [])
                               if s['source_episode_id'] != original['source_episode_id']),
                              key=lambda s: s['source_episode_id'])
        chosen = None
        source_attempts, fallback_from, fallback_to = [], None, None
        for source_index, source in enumerate([original] + alternatives):
            parents = sorted(factories, key=lambda tid: (tid != source['preferred_seed'],
                digest([request['seed'], unit['unit'], tid]), tid))
            tried = []
            for parent in parents:
                tried.append(parent)
                sf = factories[parent]
                sf.seen.update(seen)
                kind = source['restriction']
                chosen = sf.next_valid(kind['generator'], kind['bin'], kind)
                if chosen:
                    chosen.update(parent_seed_task_id=parent, split='train', sealed=False, heldout_probe=False,
                        group_id='synth/' + chosen['item_fingerprint'], expansion_unit=unit,
                        expansion_source=source, asset_root=str(sf.directory.resolve()))
                    seen.add(chosen['item_fingerprint'])
                    selected.append(chosen)
                    break
            source_attempts.append(dict(source=source, tried_seed_task_ids=tried,
                status='filled' if chosen else 'candidate_space_exhausted'))
            if chosen:
                if source_index:
                    fallback_to = source
                break
            fallback_from = original
        unit_log.append(dict(unit=unit, generated_task_id=chosen['task_id'] if chosen else None,
            status='filled' if chosen else 'unfilled_after_fallback',
            fallback_from=fallback_from, fallback_to=fallback_to, source_attempts=source_attempts))
        write_json(out / 'allocation_execution.json', unit_log)
    write_json(out / 'allocation_execution.json', unit_log)
    write_json(out / 'attempts.json', [r for sf in factories.values() for r in sf.records])
    write_json(out / 'generated.json', selected)
    write_json(out / 'summary.json', dict(allocation_summary(request['plan'], unit_log),
        strict_fill=request.get('strict_fill', False)))
    if request.get('strict_fill', False) and len(selected) != request['plan']['requested_items']:
        raise ValueError('V0 expansion candidate space exhausted; requested total not met')
    return selected


def expand(args, root, manifest):
    from tools.v0_round import (build_dataset, budget_for, run_attempts, run_labels, run_reviews,
                                sealed_records, source_inputs)
    dataset = Path(args.dataset).resolve()
    directory = root / ('expand-' + args.variant)
    if directory.is_relative_to(dataset) or dataset.is_relative_to(directory):
        raise ValueError('V0 expansion output overlaps input dataset')
    input_manifest, rows = load_dataset(dataset)
    if Path(input_manifest['synthetic_manifest']).resolve() != manifest.path:
        raise ValueError('V0 expansion requires the input dataset synthetic manifest')
    if (manifest.value.get('source_seed_manifest_sha256') is not None and
            file_hash(args.seeds) != manifest.value['source_seed_manifest_sha256']):
        raise ValueError('V0 expansion must use the original K20 seed receipt manifest')
    attempts_dir = Path(args.attempts or root / 'attempt')
    attempts = sealed_records(attempts_dir, 'records.json')
    if any(r['synthetic_manifest_sha256'] != manifest.sha256 for r in attempts):
        raise ValueError('V0 expansion attempt source differs')
    plan = expansion_plan(dataset, rows, attempts, manifest, variant=args.variant,
                          groups=args.groups, weights=args.weights, seed=args.seed)
    # Reuse the ORIGINAL identity freeze, never inspect evaluation executions.
    freeze_path = manifest.path.parent / 'split_freeze.json'
    freeze = read(freeze_path)
    paths = [dataset / 'manifest.json', dataset / 'examples.jsonl', manifest.path, args.seeds, freeze_path]
    paths += [p for p in (args.groups, args.weights) if p]
    inputs = {**input_manifest['inputs'], **bind(paths), **source_inputs(attempts_dir)}
    strict_fill, dry_run = getattr(args, 'strict_fill', False), getattr(args, 'dry_run', False)
    phase(directory, dict(command='expand', plan=plan, inputs=inputs, seed=args.seed,
        strict_fill=strict_fill, dry_run=dry_run), resume=args.resume)
    if completed(directory):
        return read(directory / 'summary.json')
    write_json(directory / 'allocation.json', plan)
    synth = directory / 'synth'
    phase(synth, dict(command='expansion-synth', inputs=inputs, plan=plan, strict_fill=strict_fill), resume=args.resume)
    if not completed(synth):
        request = dict(seeds=str(Path(args.seeds).resolve()), data_root=str(Path(args.data_root).resolve()),
            out=str(synth), seed=args.seed, heldout=freeze['groups'], seed_task_ids=sorted(manifest.seeds),
            seed_sources=manifest.value.get('seed_sources'),
            excluded_fingerprints=[t['item_fingerprint'] for t in manifest.tasks.values()], plan=plan,
            strict_fill=strict_fill)
        write_json(synth / 'request.json', request)
        env = dict(PATH=os.defpath, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0',
            PYTHONPATH=str(ROOT / 'src') + os.pathsep + str(ROOT), HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
        subprocess.run([sys.executable, '-m', 'repair.v0_expand', '--request', str(synth / 'request.json')],
                       env=env, cwd=synth, check=True, timeout=7200)
        old_tasks = [dict(t, asset_root=str(Path(t.get('asset_root', manifest.path.parent)).resolve()))
                     for t in manifest.tasks.values()]
        combined = dict(manifest.value, tasks=old_tasks + read(synth / 'generated.json'),
                        expansion_variant=args.variant, parent_manifest_sha256=manifest.sha256)
        write_json(synth / 'manifest.json', combined)
        write_json(synth / 'split_freeze.json', freeze)
        finish(synth, **allocation_summary(plan, read(synth / 'allocation_execution.json')))
    summary = allocation_summary(plan, read(synth / 'allocation_execution.json'))
    if dry_run:
        result = dict(summary, dry_run=True, strict_fill=strict_fill,
            synthetic_manifest=str(synth / 'manifest.json'), student_calls=0, teacher_calls=0, training_exposure=0)
        write_json(directory / 'summary.json', result)
        finish(directory, **result)
        load_dataset(dataset)
        return result
    expanded_manifest = SyntheticManifest(synth / 'manifest.json', manifest.seeds)
    # When importing a dataset into a new round-output root, retain its already
    # settled teacher expenditure under the SAME caps. No writes to that input.
    budget = budget_for(root)
    prior = input_manifest.get('round_teacher_budget')
    if prior and budget.summary()['reserved_attempts'] == 0:
        if prior['unsettled_attempts'] or prior['stop_reason']:
            raise ValueError('V0 imported round has unresolved teacher accounting')
        usage = prior['reported_usage']
        budget.reserve('v0-imported-round', usage['input_tokens'], usage['output_tokens'],
                       source_dataset_sha256=file_hash(dataset / 'manifest.json'))
        budget.settle('v0-imported-round', usage)
    current_args = deepcopy(args)
    current_args.task_ids = [t['task_id'] for t in read(synth / 'generated.json')]
    run_attempts(current_args, root, expanded_manifest, directory=directory / 'attempt')
    run_reviews(current_args, root, expanded_manifest, directory / 'attempt', directory=directory / 'review')
    run_labels(current_args, root, expanded_manifest, directory / 'attempt', directory / 'review',
               directory=directory / 'label', expanded=True)
    current_args.replay = str(dataset / 'replay')
    built = build_dataset(current_args, root, expanded_manifest, directory / 'label',
        directory=directory / 'dataset', inherited=rows,
        extra_inputs={**inputs, **source_inputs(synth, directory / 'attempt', directory / 'review')})
    result = dict(summary, strict_fill=strict_fill, dataset=str(directory / 'dataset'), counts=built['counts'])
    write_json(directory / 'summary.json', result)
    finish(directory, **result)
    # Reverify every original byte after publication.
    load_dataset(dataset)
    return result


def main(argv=None):
    from repair.synth_validation import offline
    offline()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', required=True)
    parser.add_argument('--strict-fill', action='store_true',
                        help='Fail if any unit remains unfilled after same-group fallback (default: off)')
    args = parser.parse_args(argv)
    request = read(args.request)
    request['strict_fill'] = args.strict_fill or request.get('strict_fill', False)
    generate(request)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(json.dumps(dict(status='error', error_type=type(exc).__name__)), file=sys.stderr)
        raise SystemExit(1) from None
