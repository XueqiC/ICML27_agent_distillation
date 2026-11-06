"""Diagnosis-only capability wrapper over the unchanged P0 ToolSession."""
from copy import deepcopy
from pathlib import Path
import time

from repair.p0_introspect import ToolSession, write_receipt, tensor_hash
from sed.storage import digest
from surgery.common import checked, read, seal, provenance
from surgery.edits import validate_edits


class Session(ToolSession):
    NAMES = ToolSession.NAMES + ('gradient_attribution', 'neuron_attribution', 'local_finetune')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.deltas = {}
        self.identity['p1_code'] = provenance()['code_sha256']
        self.validated = {}
        self.rollout_edits = set()
        # Only completed rollouts at the original decision state qualify for
        # submission verification. Action scores/counterfactuals never do.
        self.rollout_results = {}
        self.integration_structures = None

    def examples(self, pair_ids):
        if (self.scope != 'diagnosis' or not isinstance(pair_ids, list) or not pair_ids
                or len(set(pair_ids)) != len(pair_ids)):
            raise ValueError('Choose distinct diagnosis pair IDs')
        result = []
        for ident in pair_ids:
            pair = self.pairs[ident]
            if pair['split'] != 'diagnosis':
                raise ValueError('Test pair access forbidden')
            result.append((self.state(ident), pair['a_f'], pair['a_r']))
        return result

    def resolve_edit(self, edits):
        if edits is None:
            return []
        if not isinstance(edits, list):
            raise ValueError('Edit must be an ordered list')
        result = deepcopy(edits)
        for edit in result:
            if edit['type'] == 'local_finetune':
                if set(edit) != {'type', 'delta_id', 'alpha'}:
                    raise ValueError('Use a local_finetune delta_id returned in this session')
                edit['delta'] = self.deltas[edit.pop('delta_id')]
            for ident in edit.get('direction_ids', []):
                if self.scope == 'diagnosis' and ident not in self.direction_specs:
                    raise ValueError('Direction outside diagnosis capabilities')
            for field in ('a', 'b'):
                if field in edit and edit[field]['direction_id'] not in self.direction_specs:
                    raise ValueError('Vector outside diagnosis capabilities')
        validate_edits(self.student, result)
        return result

    def call(self, tool, args):
        if tool not in self.NAMES or not isinstance(args, dict):
            raise ValueError('Unknown P1 tool/schema')
        if self.integration_structures is not None:
            if tool == 'local_finetune':
                raise ValueError('Integration cannot train new edits')
            if tool in ('apply_edit', 'rollout'):
                for edit in args.get('edit') or []:
                    structure = {k: v for k, v in edit.items() if k not in ('alpha', 'c')}
                    if structure not in self.integration_structures:
                        raise ValueError('Integration may only rescale/drop/reorder/combine category edits')
        if tool not in ('gradient_attribution', 'neuron_attribution', 'local_finetune', 'apply_edit', 'rollout'):
            return super().call(tool, args)
        args = deepcopy(args)
        framing = args.pop('framing', 'original_thought')
        key = digest(dict(identity=self.identity, tool=tool, args=args, framing=framing))
        path = self.cache / (key + '.json')
        usage_before = deepcopy(self.student.usage)
        started = time.perf_counter()
        self.ledger.append(dict(event='started', tool=tool, args=args, framing=framing, cache_key=key))
        try:
            if tool in ('apply_edit', 'rollout'):
                state = self.state(args['state'])
                edits = self.resolve_edit(args.get('edit'))
            else:
                examples = self.examples(args['pair_ids'])
            if path.exists():
                result = read(path)
                if result['sha256'] != digest(result['data']) or result['cache_key'] != key:
                    raise ValueError('Corrupt tool cache')
                result['cache_hit'] = True
                data = result['data']
                if tool == 'neuron_attribution':
                    self.install_vectors(read(self.directory / 'vectors' / (key + '.json')))
                    self.direction_specs[data['direction_id']] = dict(tool=tool, args=args)
                if tool == 'local_finetune':
                    from surgery.edits import load_delta
                    load_delta(data['delta'], self.student.relocation)
                    self.deltas[data['delta_id']] = data['delta']
            else:
                if tool == 'gradient_attribution':
                    data = self.student.gradient_attribution(examples, framing=framing)
                elif tool == 'neuron_attribution':
                    data = self.student.neuron_attribution(examples, args['layer'], args['top_n'], framing=framing)
                    ident = data['direction_id']
                    self.direction_specs[ident] = dict(tool=tool, args=args)
                    write_receipt(self.directory / 'vectors' / (key + '.json'),
                                  {ident: {str(i): v.tolist() for i, v in self.student.directions[ident].items()}})
                elif tool == 'local_finetune':
                    if set(args) != {'pair_ids', 'modules', 'rank', 'lr', 'steps'}:
                        raise ValueError('local_finetune schema; targets are selected pairs\' a_r')
                    data = self.student.local_finetune(examples, args['modules'], rank=args['rank'], lr=args['lr'],
                        steps=args['steps'], directory=self.directory / 'deltas', pair_ids=args['pair_ids'], framing=framing)
                    ref = data.pop('edit')['delta']
                    ident = ref['sha256']
                    self.deltas[ident] = ref
                    data.update(delta_id=ident, delta=ref)
                elif tool == 'apply_edit':
                    data = self.student.apply_edit(state, args['candidates'], edits, framing=framing)
                else:
                    data = self.rollout(state, edits)
                summary = (self._summary(tool, data) if tool in ToolSession.NAMES else
                           'created delta ' + data['delta_id'] if tool == 'local_finetune' else
                           'created neuron direction ' + data['direction_id'] if tool == 'neuron_attribution' else
                           'gradient ranking: ' + ', '.join(r['tensor'] for r in data['ranking'][:10]))
                result = dict(tool=tool, data=data, summary=tool + ': ' + summary,
                    cache_key=key, cache_hit=False, sha256=digest(data), framing=framing)
                write_receipt(path, result)
            if tool in ('apply_edit', 'rollout') and not state['counterfactual']:
                self.validated.setdefault(digest(args.get('edit') or []), set()).add(state['origin_pair_id'])
                if tool == 'rollout' and data['verification'] == 'real_environment':
                    self.rollout_edits.add(digest(args.get('edit') or []))
                    if (state['sha256'] == self.pair_states[state['origin_pair_id']]
                            and data['termination'] != 'ENV_ERROR'):
                        self.rollout_results.setdefault(digest(edits), {})[state['origin_pair_id']] = deepcopy(result)
            self.ledger.append(dict(event='settled', tool=tool, args=args, result=result,
                environment_steps=0 if result['cache_hit'] else data.get('environment_steps', 0),
                student_usage={k: self.student.usage[k] - v for k, v in usage_before.items()},
                wall_seconds=time.perf_counter() - started))
            return result
        except Exception as exc:
            self.ledger.append(dict(event='failed', tool=tool, args=args, error_type=type(exc).__name__,
                student_usage={k: self.student.usage[k] - v for k, v in usage_before.items()}))
            raise

    def install_vectors(self, vectors):
        import torch
        for ident, layers in vectors.items():
            converted = {int(i): torch.tensor(v, dtype=torch.float32) for i, v in layers.items()}
            if ident != digest({str(i): tensor_hash(v) for i, v in converted.items()}):
                raise ValueError('Vector hash mismatch')
            self.student.directions[ident] = converted

    def install_submission(self, frozen):
        self.install_vectors(frozen['vectors'])
        self.direction_specs.update(frozen['direction_specs'])
        edits = deepcopy(frozen['submission']['edit'])
        for edit in edits:
            if edit['type'] == 'local_finetune':
                ref = edit.pop('delta')
                self.deltas[ref['sha256']] = ref
                edit['delta_id'] = ref['sha256']
        self.resolve_edit(edits)
        return dict(frozen['submission'], edit=edits)

    def bundle(self, submission):
        edits = self.resolve_edit(submission['edit'])
        ids = set()
        for edit in edits:
            ids.update(edit.get('direction_ids', []))
            ids.update(edit[k]['direction_id'] for k in ('a', 'b') if k in edit)
        return dict(submission=dict(submission, edit=edits),
            vectors={ident: {str(i): v.tolist() for i, v in self.student.directions[ident].items()} for ident in sorted(ids)},
            direction_specs={ident: self.direction_specs[ident] for ident in sorted(ids)})
