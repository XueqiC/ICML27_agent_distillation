"""V0 immutable phase receipts and manifest-only task authorization.

Audit records are deliberately not policy inputs. All paths published by a
phase are sealed before a dependent phase can consume them.
"""
import json
from pathlib import Path
import subprocess

from sed.provenance import safe_config, utc_now, versions
from sed.storage import digest, file_hash, write_json

ROOT = Path(__file__).resolve().parents[1]
BINS = ('near', 'mid', 'far', 'very_far')
QUOTAS = dict(near=6, mid=4, far=2, very_far=1)
CAPS = dict(input_tokens=60_000_000, output_tokens=6_000_000)
DECODINGS = [('greedy', None), ('sampled', 0), ('sampled', 1), ('sampled', 2)]
THETA0_CELLS = {10: 'e4b-tsft-k10-e32-lr5e-5', 20: 'e4b-tsft-k20-e32-lr5e-5',
                30: 'e4b-tsft-k30-e64-lr5e-5'}


def checked_k(value):
    if type(value) is not int or value not in THETA0_CELLS:
        raise ValueError('V0 requires K in {10, 20, 30} verified seed trajectories')
    return value


def seed_manifest_k(path):
    value = read(path)
    k = checked_k(value.get('K'))
    if not isinstance(value.get('verified'), list) or len(value['verified']) != k:
        raise ValueError('V0 K differs from verified seed trajectory count')
    return k


def round_identity(k):
    return dict(schema='v0-round/1', seed=0, K=checked_k(k), caps=CAPS, training_exposure=0)


def check_theta0(k, name, path):
    expected = THETA0_CELLS[checked_k(k)]
    path = Path(path).resolve()
    cells = set(THETA0_CELLS.values()).intersection(path.parts)
    if name != expected or cells != {expected}:
        raise ValueError(f'V0 K{k} requires MODEL_NAME and ADAPTER from {expected}')
    state = read(path / 'training_state.json')
    if (state.get('N_examples') != k or state.get('final') is not True or state.get('seed') != 0
            or state.get('epochs') != (64 if k == 30 else 32)):
        raise ValueError(f'V0 K{k} requires the configured final seed-0 theta0 training state')


def read(path):
    return json.loads(Path(path).read_text())


def bind(paths):
    return {str(Path(p).resolve()): file_hash(p) for p in paths}


def check_inputs(inputs):
    for path, expected in inputs.items():
        if file_hash(path) != expected:
            raise ValueError('V0 sealed input changed')


def inside(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError('V0 path escapes output/artifact root')
    return path


def code_identity():
    paths = sorted((ROOT / 'repair').glob('*.py'))
    paths += sorted((ROOT / 'tools').glob('v0_*.py'))
    paths += [ROOT / 'tools/build_v2_dataset.py']  # Shared ordered label workers/tokenizer lock.
    paths += [ROOT / p for p in ('src/sed/react.py', 'src/sed/visibility.py',
        'students/sampled_student.py', 'students/vllm_student.py',
        'harness/alfworld_env/environment.py', 'teachers/openai_teacher.py', 'teachers/capped.py')]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


def code_revision():
    """Bind V0 phase and interchange provenance to the running checkout."""
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def phase(directory, identity, *, resume=False):
    """Reserve a phase, or require the identical frozen plan on explicit resume."""
    directory = Path(directory)
    identity = dict(identity, code_sha256=code_identity(), dependencies=versions())
    identity = json.loads(json.dumps(identity))
    path = directory / 'plan.json'
    if directory.exists():
        if not resume or not path.is_file() or read(path)['identity'] != identity:
            raise ValueError('V0 phase exists; resume requires identical inputs, config and code')
    else:
        directory.mkdir(parents=True, exist_ok=False)
        write_json(path, dict(identity=identity, run_sha256=digest(identity), created=utc_now(),
            code_revision=code_revision(),
            training_exposure=0), exclusive=True)
    check_inputs(identity.get('inputs', {}))
    return read(path)


def artifacts(directory):
    directory = Path(directory)
    return {str(p.relative_to(directory)): file_hash(p) for p in sorted(directory.rglob('*'))
            if p.is_file() and p.name != 'completion.json' and not p.name.endswith('.lock')}


def completed(directory):
    directory = Path(directory)
    path = directory / 'completion.json'
    if not path.exists():
        return False
    value = read(path)
    if value['artifacts'] != artifacts(directory):
        raise ValueError('V0 completed phase artifacts changed')
    check_inputs(read(directory / 'plan.json')['identity'].get('inputs', {}))
    return True


def finish(directory, **summary):
    directory = Path(directory)
    check_inputs(read(directory / 'plan.json')['identity'].get('inputs', {}))
    write_json(directory / 'completion.json', dict(summary, artifacts=artifacts(directory)), exclusive=True)


class SyntheticManifest:
    """Only these exact IDs can reset a game; probes have a separate permission."""
    def __init__(self, path, seed_ids=None):
        self.path = Path(path).resolve()
        self.value, self.sha256 = read(path), file_hash(path)
        if self.value.get('schema') != 'v0-synthetic-tasks/1':
            raise ValueError('V0 synthetic manifest schema')
        ids = self.value['seed_task_ids']
        self.k = checked_k(self.value.get('K', len(ids)))
        if len(ids) != self.k or len(set(ids)) != len(ids):
            raise ValueError('V0 K differs from frozen seed trajectory identities')
        self.seeds = set(ids)
        if seed_ids is not None and set(seed_ids) != self.seeds:
            raise ValueError('V0 synthetic manifest seeds differ from the original seeds')
        self.tasks, fingerprints, groups = {}, set(), {'train': set(), 'holdout': set()}
        for row in self.value['tasks']:
            tid = row['task_id']
            parts = tid.split('/')
            if (len(parts) != 3 or parts[:2] != ['train', 'synth'] or parts[2] in ('', '.', '..')
                    or '\\' in tid or tid in self.tasks or row['status'] != 'valid'
                    or row['parent_seed_task_id'] not in self.seeds
                    or row['parent_seed_task_id'].startswith('train/synth/')
                    or row['bin'] not in BINS or row['generator'] not in ('G1', 'G2', 'G3', 'G5')):
                raise ValueError('V0 unauthorized synthetic identity/parent')
            if row['generator'] not in dict(near=('G1', 'G3'), mid=('G2',), far=('G2',), very_far=('G5',))[row['bin']]:
                raise ValueError('V0 generator does not match distance bin')
            probe = row.get('heldout_probe')
            if type(probe) is not bool or row['split'] != ('holdout' if probe else 'train') or row['sealed'] != probe:
                raise ValueError('V0 probe split mismatch')
            fp = row['item_fingerprint']
            if fp in fingerprints:
                raise ValueError('V0 duplicate training/probe fingerprint')
            fingerprints.add(fp)
            groups[row['split']].add(row['group_id'])
            self.tasks[tid] = row
        if groups['train'] & groups['holdout']:
            raise ValueError('V0 training/probe group overlap')

    def authorize(self, task_id, *, probes=False):
        if file_hash(self.path) != self.sha256 or task_id not in self.tasks:
            raise ValueError('V0 task not in frozen synthetic manifest')
        task = self.tasks[task_id]
        if task['heldout_probe'] != probes:
            raise ValueError('V0 held-out probe/training permission mismatch')
        return task

    def asset(self, task, key):
        root = Path(task.get('asset_root', self.path.parent)).resolve()
        path = inside(root, task[key])
        if file_hash(path) != task[key.replace('_path', '_sha256')]:
            raise ValueError('V0 synthetic asset hash changed')
        return path

    def environment(self, task_id, *, probes=False, seed=0):
        from harness.alfworld_env import Environment
        task = self.authorize(task_id, probes=probes)
        path = self.asset(task, 'game_path')
        # game_path is exactly data_root/train/synth/<id>/game.tw-pddl.
        root = path.parents[3]
        if (root / task_id / 'game.tw-pddl').resolve() != path:
            raise ValueError('V0 manifest game path does not match task ID')
        env = Environment(root, seed=seed, max_episode_steps=40, split='train')
        native_reset = env.reset
        def authorized_reset(candidate_id):
            if candidate_id != task_id:
                raise ValueError('V0 environment cannot reset an unauthorized task ID')
            self.authorize(candidate_id, probes=probes)
            self.asset(task, 'game_path')
            return native_reset(candidate_id)
        env.reset = authorized_reset
        return env


def safe_effective(config):
    return safe_config(config)
