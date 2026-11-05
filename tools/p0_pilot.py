#!/usr/bin/env python
"""P0 entrypoints. Default extraction is CPU-only and makes no API requests."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]

from repair.p0_pairs import (DEFAULT_ROUND, DEFAULT_BASE, DEFAULT_ADAPTER, CATEGORIES,
    extract_pairs, load_pairs, provenance, teacher_pair)
from sed.storage import write_json, digest, file_hash


def model_runtime(args, pairs):
    expected = dict(CUDA_DEVICE_ORDER='PCI_BUS_ID',
                    TORCH_COMPILE_DISABLE='1', TORCHDYNAMO_DISABLE='1')
    physical = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if any(os.environ.get(k) != v for k, v in expected.items()) or physical not in tuple('01234567'):
        raise RuntimeError('GPU operations require CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<one index 0-7> '
                           'TORCH_COMPILE_DISABLE=1 TORCHDYNAMO_DISABLE=1')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Torch must see exactly one CUDA device')
    runtime_device = dict(physical_gpu_index=int(physical), device_name=torch.cuda.get_device_name(0))
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    from repair.p0_introspect import load_theta0, StudentTools
    model, tokenizer, identity = load_theta0(args.base, args.adapter, pairs['student'])
    identity = dict(identity, runtime_device=runtime_device)
    return StudentTools(model, tokenizer, identity=identity,
        max_new_tokens=pairs['serving']['max_tokens'], max_model_len=pairs['serving']['max_model_len'])


class ScriptedTeacherClient:
    """Fake Responses endpoint, routed through the REAL round usage transport.

    This has no SDK/client/network construction. Usage is marked synthetic and
    must never be reported as a measured bill or real teacher performance.
    """
    max_retries = 0

    def __init__(self, script):
        self.script, self.calls = script, 0
        self.requests = []
        self.responses = self
        self.with_raw_response = self

    def create(self, **kwargs):
        self.requests.append(kwargs)
        payload = json.loads(kwargs['input'][1]['content'])
        history, budget = kwargs['input'][2]['content'].rsplit('\nBUDGET ', 1)
        payload.update(json.loads(history))
        payload['budget'] = json.loads(budget)
        turn = self.script[self.calls]
        if callable(turn):
            turn = turn(payload)
        self.calls += 1
        raw = turn if isinstance(turn, str) else json.dumps(turn)
        response = SimpleNamespace(output_text=raw, model='FAKE_NO_MODEL', id=f'fake-{self.calls}',
            status='completed', usage={'input_tokens': 100, 'output_tokens': 40,
            'output_tokens_details': {'reasoning_tokens': 0}})
        return SimpleNamespace(status_code=200, headers={}, parse=lambda: response)


def smoke_script(pair_a, pair_b, layers):
    a, b = pair_a['pair_id'], pair_b['pair_id']
    candidates = [pair_a['a_f'], pair_a['a_r']]
    def direction_id(payload):
        return next(t['result']['data']['direction_id'] for t in payload['transcript']
                    if t['request']['tool'] == 'direction')
    def edit(payload):
        return dict(direction_id=direction_id(payload), layers=layers,
                    targets=['mlp.down_proj', 'self_attn.o_proj'], alpha=0.1)
    return [dict(tool='action_scores', args=dict(state=a, candidates=list(dict.fromkeys(candidates)))),
        dict(tool='action_scores', args=dict(state=b, candidates=list(dict.fromkeys([pair_b['a_f'], pair_b['a_r']])))),
        dict(tool='logit_lens', args=dict(state=a, a_f=pair_a['a_f'], a_r=pair_a['a_r'])),
        dict(tool='attention_summary', args=dict(state=a, layers=layers)),
        lambda p: dict(tool='counterfactual_context', args=dict(state=a, edit=dict(op='insert',
            span=next(s['id'] for s in p['pairs'][0]['spans'] if s['label'] == 'latest_observation'),
            side='after', text='\nP0 scripted context intervention.'))),
        dict(tool='patch', args=dict(target_state=a, source_state=a, layers=layers,
            positions=[dict(target=-1, source=-1)], candidates=list(dict.fromkeys(candidates)))),
        dict(tool='direction', args=dict(positive=[dict(state=a, action=pair_a['a_f']), dict(state=b, action=pair_b['a_f'])],
            negative=[dict(state=a, action=pair_a['a_r']), dict(state=b, action=pair_b['a_r'])], layers=layers)),
        lambda p: dict(tool='projection', args=dict(state=a, action=pair_a['a_f'], direction_id=direction_id(p))),
        lambda p: dict(tool='apply_edit', args=dict(state=a, candidates=list(dict.fromkeys(candidates)), edit=edit(p))),
        dict(tool='rollout', args=dict(state=a)),
    ]


def run_smoke(args, pairs, student=None):
    from repair.p0_agent import AgentPrompt, make_transport, parse_turn, AgentBudget
    from repair.p0_introspect import ToolSession, tensor_hash
    from repair.v0_common import SyntheticManifest
    from sed.config import load_config
    directory = Path(args.output)
    write_json(directory / 'smoke_started.json', dict(schema='p0-gpu-smoke/1', pairs_sha256=pairs['sha256'],
        provenance=provenance(dict(cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                                   API=False, scripted_teacher=True))), exclusive=True)
    timings, start = {}, time.perf_counter()
    try:
        student = student or model_runtime(args, pairs)
    except Exception as exc:
        write_json(directory / 'smoke_report.json', dict(status='blocked_before_model_load',
            error_type=type(exc).__name__, real_gpu_smoke=False, teacher_api_calls=0,
            timings=dict(model_preflight_seconds=time.perf_counter() - start)), exclusive=True)
        raise
    timings['model_load'] = time.perf_counter() - start
    # Predefined selection: first two aligned diagnosis IDs; never select using
    # reproduction outcomes, success, or test pairs. The full split is unchanged.
    selected = sorted([p for p in pairs['pairs'] if p['split'] == 'diagnosis' and p['a_r'] is not None],
                      key=lambda p: p['pair_id'])[:2]
    if len(selected) != 2:
        raise ValueError('Need two aligned diagnosis pairs for real smoke')
    manifest = SyntheticManifest(pairs['manifest_path'])
    session = ToolSession(student, selected, directory / 'tools', env_factory=lambda p: manifest.environment(
        p['task_id'], seed=p['audit']['env_seed']))
    layers = [0, 20, 23, 24, 41]
    script = smoke_script(*selected, layers)
    client = ScriptedTeacherClient(script)
    transport = make_transport(load_config('configs/base_e4b.yaml'), 'smoke', directory / 'teacher', client=client)
    transcript, budget = [], AgentBudget()
    prompt_pairs = [teacher_pair(p, student.decision_prefix(session.state(p['pair_id']))) for p in selected]
    checks, reproduction = {}, []
    originals = {(i, target): tensor_hash(getattr(getattr(student.layers[i], parent), name).weight)
                 for i in layers for target, parent, name in [('mlp.down_proj', 'mlp', 'down_proj'),
                    ('self_attn.o_proj', 'self_attn', 'o_proj')]}
    for index in range(len(script)):
        raw = transport(AgentPrompt(dict(category='scripted_smoke', pairs=prompt_pairs,
            transcript=transcript, budget=budget.status())))
        turn, = parse_turn(raw)  # The smoke script deliberately uses singleton replies.
        budget.reserve(turn['tool'], turn['args'])
        started = time.perf_counter()
        before_steps = session.environment_steps
        result = session.call(turn['tool'], turn['args'])
        budget.steps += session.environment_steps - before_steps
        timings[f"{index:02d}_{turn['tool']}"] = time.perf_counter() - started
        transcript.append(dict(request=turn, result=result))
        if turn['tool'] == 'action_scores':
            p = selected[len(reproduction)]
            reproduction.append(dict(pair_id=p['pair_id'], a_f=p['a_f'], decoding=p['decoding'],
                greedy=result['data']['greedy']['action'], matches=result['data']['greedy']['action'] == p['a_f']))
        if turn['tool'] == 'patch':
            checks['patch_identity'] = all(r['delta_log_probability'] == 0 for r in result['data']['changes'])
    checks['edit_reverted_exactly'] = all(tensor_hash(getattr(getattr(student.layers[i], target.split('.')[0]),
        target.split('.')[1]).weight) == h for (i, target), h in originals.items())
    checks['reproduces_original_actions'] = all(r['matches'] for r in reproduction)
    checks['single_real_rollout'] = budget.rollouts == 1
    report = dict(status='passed' if all(checks.values()) else 'completed_with_mismatches',
        real_gpu_smoke=True, teacher_api_calls=0, fake_teacher_calls=client.calls,
        fake_usage_note='Synthetic fixed usage counts for transport tests only',
        timings=timings, checks=checks, reproduction=reproduction, budget=budget.status(),
        selected_pairs=[p['pair_id'] for p in selected], model_identity=student.identity,
        pairs_sha256=pairs['sha256'])
    write_json(directory / 'smoke_report.json', report, exclusive=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['extract', 'smoke', 'diagnose', 'evaluate'])
    parser.add_argument('--round-dir', default=DEFAULT_ROUND)
    parser.add_argument('--pairs', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--base', default=DEFAULT_BASE)
    parser.add_argument('--adapter', default=DEFAULT_ADAPTER)
    parser.add_argument('--category', choices=CATEGORIES)
    parser.add_argument('--submission', action='append', default=[], type=Path)
    parser.add_argument('--allow-teacher-api', action='store_true', help='Explicit future-run gate; never needed for smoke')
    parser.add_argument('--input-token-cap', type=int, default=2_000_000)
    parser.add_argument('--output-token-cap', type=int, default=500_000)
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument('--shard', help='evaluate only: zero-based i/n stable-hash shard (0 <= i < n)')
    execution.add_argument('--merge', action='store_true', help='evaluate only: verify and merge shards on CPU')
    args = parser.parse_args(argv)
    if (args.shard is not None or args.merge) and args.command != 'evaluate':
        parser.error('--shard/--merge require evaluate')
    if args.shard is not None:
        from repair.p0_eval import parse_shard
        try:
            args.shard = parse_shard(args.shard)
        except ValueError as exc:
            parser.error(str(exc))
    if args.command == 'diagnose' and args.output.exists():
        parser.error('diagnose output directory already exists; use a fresh reports/p0_runs/diagnose_v4/<category> directory')
    if args.command == 'extract':
        value = extract_pairs(args.round_dir)
        write_json(args.output / 'pairs.json', value, exclusive=True)
        write_json(args.output / 'extraction_report.json', dict(schema=value['schema'], counts=value['counts'],
            categories={c: dict(total=value['counts']['categories'].get(c, 0),
                **{s: value['counts']['splits'][s]['categories'][c] for s in ('diagnosis', 'test')}) for c in CATEGORIES},
            excluded=value['excluded'], split_rule=value['split_rule'], split_sha256=value['split_sha256'],
            pairs_sha256=value['sha256'], reproduction='not_run',
            note='First divergence of final verified execution; teacher executor and visible prefix checked. '
                 'Exclusion reasons are ordered: divergence eligibility/state mismatch, then category_other. '
                 'All applicable reasons are preserved; parent-seed overlap is reported only.'), exclusive=True)
        print(json.dumps(value['counts'], indent=2))
        return
    if args.pairs is None:
        parser.error('--pairs is required')
    pairs = load_pairs(args.pairs)
    if args.command == 'smoke':
        print(json.dumps(run_smoke(args, pairs), indent=2))
        return
    if args.command == 'diagnose':
        if not args.allow_teacher_api or not args.category:
            parser.error('diagnose requires --category and explicit --allow-teacher-api')
        selected = [p for p in pairs['pairs'] if p['split'] == 'diagnosis' and p['category'] == args.category]
        from repair.p0_agent import P0Agent, make_transport, validate_diagnosis
        validate_diagnosis(selected, args.category, pairs['schema'])
        student = model_runtime(args, pairs)
        from repair.p0_introspect import ToolSession
        from repair.v0_common import SyntheticManifest
        from sed.config import load_config
        manifest = SyntheticManifest(pairs['manifest_path'])
        session = ToolSession(student, selected, args.output / 'tools', env_factory=lambda p: manifest.environment(
            p['task_id'], seed=p['audit']['env_seed']))
        # Reproduction checks are diagnostic only and never filter the split.
        checks = []
        for p in selected:
            greedy = student.greedy(session.state(p['pair_id']))
            checks.append(dict(pair_id=p['pair_id'], original=p['a_f'], greedy=greedy,
                               matches=greedy['action'] == p['a_f'], kept=True))
        write_json(args.output / 'reproduction.json', checks, exclusive=True)
        transport = make_transport(load_config('configs/base_e4b.yaml'), args.category, args.output,
            input_cap=args.input_token_cap, output_cap=args.output_token_cap)
        result = P0Agent(session, transport, args.output, category=args.category,
                         pairs_sha256=pairs['sha256'], pairs_schema=pairs['schema']).run()
        print(json.dumps(dict(submission_sha256=result['sha256'])))
        return
    # No teacher object or client is constructed in evaluation.
    from repair.p0_eval import evaluate, merge_evaluation
    from repair.p0_agent import load_submission
    if len(args.submission) != 5:
        parser.error('evaluate needs all five --submission paths')
    for p in args.submission:
        load_submission(p, pairs_sha256=pairs['sha256'])
    if args.merge:
        report = merge_evaluation(pairs, args.submission, args.output)
    else:
        student = model_runtime(args, pairs)
        report = evaluate(student, pairs, args.submission, args.output, shard=args.shard)
    print(json.dumps(report if args.shard is not None else
                     dict(report=str(args.output / 'report.json'), complete=report['complete'])))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # No raw SDK/model/environment exception text can reach logs.
        print('P0 halted: ' + type(exc).__name__ + '; inspect sanitized receipts and P0_PILOT.md', file=sys.stderr)
        sys.exit(1)
