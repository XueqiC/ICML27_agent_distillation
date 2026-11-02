"""V0 student collection uses the shared parser, prompt and measured continuation."""
from pathlib import Path
import math

from repair.execute import execute_actions
from repair.ledger import BudgetLedger
from repair.records import TrajectoryRecord
from repair.v0_common import DECODINGS, read
from sed.storage import digest, write_json


def consistent_prefix(reference, actual, length):
    keys = ('visible', 'audit_sha256', 'reward', 'terminated', 'truncated')
    return (actual.termination_reason != 'ENV_ERROR' and reference.initial is not None and actual.initial is not None
        and actual.task_id == reference.task_id and actual.env_seed == reference.env_seed
        and actual.environment_version == reference.environment_version and actual.game_sha256 == reference.game_sha256
        and len(reference.steps) >= length and len(actual.steps) >= length
        and all(reference.initial[k] == actual.initial[k] for k in keys)
        and all(all(left[k] == right[k] for k in (*keys, 'action_parsed', 'action_raw'))
                for left, right in zip(reference.steps[:length], actual.steps[:length])))


def attempt_id(manifest, task_id, decoding, sampling_seed):
    return digest([manifest.sha256, task_id, decoding, sampling_seed])[:32]


def attempt(manifest, task_id, *, decoding, sampling_seed, env_factory, student_factory,
            directory, probes=False):
    task = manifest.authorize(task_id, probes=probes)  # Before any environment/model construction.
    if (decoding, sampling_seed) not in DECODINGS:
        raise ValueError('V0 requires greedy or sampled seeds 0, 1, 2')
    directory = Path(directory)
    ident = attempt_id(manifest, task_id, decoding, sampling_seed)
    path = directory / 'record.json'
    if path.exists():
        row = read(path)
        if row['attempt_id'] != ident or row['synthetic_manifest_sha256'] != manifest.sha256:
            raise ValueError('V0 attempt resume identity mismatch')
        return row
    directory.mkdir(parents=True, exist_ok=True)
    ledger = BudgetLedger(directory / 'environment.jsonl')
    replies, prefix = [], None
    if task['generator'] == 'G3':
        prefix = read(manifest.asset(task, 'prefix_path'))
        replies = prefix['prefix_replies']
        if prefix['task_id'] != task_id or len(replies) != len(prefix['transitions']):
            raise ValueError('V0 G3 prefix identity mismatch')
    reference = execute_actions(env_factory, task_id, replies, 40, ledger=ledger,
                                interaction_kind='student_environment_steps')
    write_json(directory / 'prefix_replay.json', reference.to_json())
    if reference.termination_reason == 'ENV_ERROR' or reference.success is True or len(reference.steps) != len(replies):
        raise ValueError('V0 prefix replay failed or terminal; no student called')
    if prefix is not None and (reference.initial['visible'] != prefix['initial_observation'] or any(
            s['action_parsed'] != p['action'] or s['visible'] != p['observation']
            for s, p in zip(reference.steps, prefix['transitions']))):
        raise ValueError('V0 G3 prefix replay differs; no student called')
    student = student_factory()
    try:
        execution = execute_actions(env_factory, task_id, replies, 40, prefix_reference=reference,
            continuation=lambda h, v: student.generate(h, {'task_id': task_id}, v).raw_text,
            ledger=ledger, interaction_kind='evaluation_steps' if probes else 'student_environment_steps')
    finally:
        if hasattr(student, 'close'):
            student.close()
    row = dict(attempt_id=ident, episode_id='student-' + ident, task_id=task_id, decoding=decoding,
        sampling_seed=sampling_seed, prefix_length=len(replies), heldout_probe=probes,
        synthetic_manifest_sha256=manifest.sha256, success=execution.success,
        execution=execution.to_json(), executors=['student'] * len(execution.steps),
        prefix_turns_context_only=True, environment_budget=ledger.summary(),
        environment_steps=sum(ledger.summary()['interaction'].values()), teacher_calls=0, training_exposure=0)
    write_json(path, row, exclusive=True)
    return row


def selected_failures(attempts):
    by_task = {}
    for row in attempts:
        if row['heldout_probe']:
            continue
        if row['success'] is False and row['execution']['termination_reason'] != 'ENV_ERROR':
            by_task.setdefault(row['task_id'], []).append(row)
    return [r for tid in sorted(by_task) for r in sorted(by_task[tid],
        key=lambda r: (DECODINGS.index((r['decoding'], r['sampling_seed'])), r['attempt_id']))[:2]]


def clopper_pearson_upper(failures, total):
    """Upper end of the equal-tailed exact 95% binomial interval (alpha/2=.025)."""
    if type(total) is not int or type(failures) is not int or not 0 <= failures <= total:
        raise ValueError('Invalid binomial counts')
    if not total:
        return None
    if failures == total:
        return 1.0
    # Invert P_p[X <= failures] = alpha/2. Log-binomial terms avoid requiring
    # scipy in the environment-only venv; this is the same exact beta quantile.
    lower, upper = 0.0, 1.0
    coefficients = [math.lgamma(total + 1) - math.lgamma(i + 1) - math.lgamma(total - i + 1)
                    for i in range(failures + 1)]
    for _ in range(80):
        p = (lower + upper) / 2
        if p in (0., 1.):
            break
        cdf = sum(math.exp(c + i * math.log(p) + (total - i) * math.log1p(-p))
                  for i, c in enumerate(coefficients))
        if cdf > .025:
            lower = p
        else:
            upper = p
    return (lower + upper) / 2


def probe_report(tasks, attempts):
    from repair.v0_common import BINS
    index = {(r['task_id'], r['decoding'], r['sampling_seed']): r for r in attempts}
    bins = {}
    for bin_name in BINS:
        ids = [t['task_id'] for t in tasks if t['heldout_probe'] and t['bin'] == bin_name]
        decoding_stats = {}
        for decoding, seed in DECODINGS:
            values = [index.get((tid, decoding, seed), {}).get('success') for tid in ids]
            known = [v for v in values if type(v) is bool]
            failed = sum(v is False for v in known)
            decoding_stats[decoding if seed is None else f'sampled-s{seed}'] = dict(
                tasks=len(ids), known=len(known), unknown=len(ids) - len(known), failures=failed,
                failure_rate=failed / len(known) if known else None,
                cp95_upper=clopper_pearson_upper(failed, len(known)))
        # Each task contributes once; correlated sampled attempts are not n=4 trials.
        values = [[index.get((tid, d, s), {}).get('success') for d, s in DECODINGS] for tid in ids]
        complete = [v for v in values if all(type(x) is bool for x in v)]
        failed = sum(any(x is False for x in v) for v in complete)
        upper = clopper_pearson_upper(failed, len(complete))
        bins[bin_name] = dict(per_decoding=decoding_stats, tasks=len(ids), known=len(complete),
            unknown=len(ids) - len(complete), failures=failed,
            failure_rate=failed / len(complete) if complete else None, cp95_upper=upper,
            stop_eligible=bool(upper is not None and upper < .05 and len(complete) == len(ids)))
    return dict(by_bin=bins, unit='one heldout task; failure means any of four attempts fails',
        interval='equal-tailed Clopper-Pearson 95%; separate decoding marginals; no pooling correlated trials',
        teacher_calls=0, training_exposure=0)
