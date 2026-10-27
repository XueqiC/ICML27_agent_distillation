"""Evidence-backed ALFWorld facts. Only observation/action/outcome are read.

Step ids are public policy turns (initial = -1). Locations are last observed,
not omniscient state. No admissible commands, task ids, rewards or audit fields
are consulted. Unrecognized prose remains unknown.
"""
from copy import deepcopy
import json
import re

INSTANCE = r'[a-z][a-z0-9_]* \d+'


def fact(value, *steps):
    return {'value': value, 'evidence_steps': sorted(set(steps))}


def observation(value):
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return ''
    # Deliberate whitelist; never recursively scan arbitrary event fields.
    visible = value.get('visible')
    return (visible.get('observation', '') if isinstance(visible, dict)
            else value.get('observation', '')) or ''


def _noun(text, plural=False):
    text = re.sub(r'^(?:a|an|some|the) ', '', text.strip().lower())
    if plural:
        irregular = {'knives': 'knife', 'loaves': 'loaf', 'potatoes': 'potato', 'tomatoes': 'tomato'}
        text = irregular.get(text, text[:-3] + 'y' if text.endswith('ies') else
                             text[:-2] if text.endswith(('sses', 'shes', 'ches', 'xes')) else
                             text[:-1] if text.endswith('s') and not text.endswith('ss') else text)
    return text


def parse_goal(text):
    match = re.search(r'Your task is to:\s*([^\n.!?]+)[.!?]?', text, re.I)
    if not match:
        return fact(None)
    goal = match[1].strip().lower()
    noun = r'([a-z][a-z _-]*?)'
    article = r'(?:(?:a|an|some|the) )?'
    state, count, obj, target = None, 1, None, None
    m = re.fullmatch(r'(?:examine|look at) ' + article + noun +
                     r' (?:with|under) ' + article + r'(desklamp)', goal)
    if m:
        obj, target, state = m[1], m[2], 'lit'
    else:
        m = (re.fullmatch(r'put two ' + noun + r' (?:in|on) ' + article + noun, goal) or
             re.fullmatch(r'find two ' + noun + r' and put them (?:in|on) ' + article + noun, goal))
        if m:
            obj, target, count = _noun(m[1], plural=True), m[2], 2
        else:
            m = re.fullmatch(r'(clean|heat|cool) ' + article + noun +
                             r' and put it (?:in|on) ' + article + noun, goal)
            if m:
                state, obj, target = {'clean': 'clean', 'heat': 'hot', 'cool': 'cool'}[m[1]], m[2], m[3]
            else:
                m = re.fullmatch(r'put ' + article + r'(?:(clean|hot|cool) )?' + noun +
                                 r' (?:in|on) ' + article + noun, goal)
                if m:
                    state, obj, target = m[1], m[2], m[3]
    if obj is None:
        return fact(None, -1)
    return fact({'object_type': _noun(obj), 'required_state': state,
                 'target_receptacle_type': _noun(target), 'count': count}, -1)


def _type(instance):
    return instance.rsplit(' ', 1)[0]


class FactTable:
    def __init__(self, initial=''):
        self.goal = parse_goal(observation(initial))
        self.holding = fact(None)
        self.locations, self.sightings, self.receptacles = {}, [], {}
        self.refused_actions, self.state_changes, self.lamps = {}, {}, {}
        self.inventory_checks, self.refused_takes_while_holding = [], []
        self.current = None
        self.next_step = 0
        self._observe(observation(initial), -1, '')

    @classmethod
    def from_record(cls, initial, steps):
        table = cls(initial)
        for step in steps:
            table.update(step)
        return table

    def _receptacle(self, name):
        return self.receptacles.setdefault(name, {
            'state': fact(None), 'visited': fact(None), 'opened': fact(None),
            'contents_observed': fact(None), 'contents': fact(None)})

    def _set_location(self, obj, location, step):
        old = self.locations.get(obj, {}).get('value')
        if old and old != 'held' and old != location:
            rec = self._receptacle(old)
            if rec['contents']['value'] is not None:
                rec['contents'] = fact([x for x in rec['contents']['value'] if x != obj],
                                       *rec['contents']['evidence_steps'], step)
        self.locations[obj] = fact(location, step)
        if location and location != 'held':
            rec = self._receptacle(location)
            if rec['contents']['value'] is not None:
                rec['contents'] = fact(sorted(set(rec['contents']['value']) | {obj}),
                                       *rec['contents']['evidence_steps'], step)

    def _contents(self, place, contents, step):
        rec = self._receptacle(place)
        if rec['state']['value'] == 'closed':
            return  # A view obstructed by a closed container cannot exclude objects.
        objects = sorted(set(re.findall(r'\b(' + INSTANCE + r')\b', contents.lower())))
        if not objects and not re.search(r'\bnothing\b', contents, re.I):
            return  # An unfamiliar list is not evidence of absence.
        rec['contents_observed'] = fact(True, step)
        rec['contents'] = fact(objects, step)
        for obj, loc in list(self.locations.items()):
            if loc['value'] == place and obj not in objects:
                self.locations[obj] = fact(None, step)
        for obj in objects:
            self.locations[obj] = fact(place, step)
            self.sightings.append({'instance': obj, 'receptacle': place, 'evidence_steps': [step]})

    def _observe(self, text, step, action):
        text = text.lower()
        # Only explicit carrying statements reconcile inventory.
        carrying = re.search(r'you are carrying:\s*([^\n.]+)', text)
        empty = 'you are not carrying anything' in text
        if carrying or empty:
            objects = [] if empty else sorted(set(re.findall(INSTANCE, carrying[1])))
            if empty or objects:
                before = self.holding
                agreed = None if before['value'] is None else before['value'] == objects
                self.inventory_checks.append({'step': step, 'expected': deepcopy(before),
                    'observed': fact(objects, step), 'agreement': agreed, 'disagreement': agreed is False})
                for obj in before['value'] or []:
                    if obj not in objects:
                        self._set_location(obj, None, step)
                self.holding = fact(objects, step)
                for obj in objects:
                    self._set_location(obj, 'held', step)
        # Arrival establishes a public referent for "In it" and "you see nothing".
        arrival = re.search(r'you (?:arrive at|are (?:at|on|in)) (?:the )?(' + INSTANCE + ')', text)
        if arrival:
            self.current = arrival[1]
            self._receptacle(self.current)['visited'] = fact(True, step)
        for m in re.finditer(r'(?:the (' + INSTANCE + r') is (closed|open)|you (open|close) the (' + INSTANCE + r'))', text):
            place, state = (m[1], m[2]) if m[1] else (m[4], 'open' if m[3] == 'open' else 'closed')
            rec = self._receptacle(place)
            rec['state'] = fact(state, step)
            if m[3] == 'open':
                rec['opened'] = fact(True, step)
                rec['contents_observed'] = fact(True, step)
            self.current = place
        for m in re.finditer(r'(?:on|in) the (' + INSTANCE + r'),? you see ([^.\n]+)', text):
            self.current = m[1]
            self._receptacle(m[1])['contents_observed'] = fact(True, step)
            self._contents(m[1], m[2], step)
        for m in re.finditer(r'in it,? you see ([^.\n]+)', text):
            if self.current:
                self._contents(self.current, m[1], step)
        if self.current and re.search(r'\byou see nothing\b', text):
            self._contents(self.current, 'nothing', step)
        # "You see ..." after a successful examine/open/arrival is local. The
        # initial room inventory without a location is deliberately not a sighting.
        if self.current and (arrival or action.startswith(('go to ', 'open ', 'examine ', 'look'))):
            m = re.search(r'(?:^|[.!]\s*)you see ([^.\n]+)', text)
            if m and self._receptacle(self.current)['state']['value'] != 'closed':
                self._contents(self.current, m[1], step)

    def update(self, step):
        index = self.next_step
        self.next_step += 1
        raw_action = step.get('action_parsed') or ''
        action = raw_action.strip().lower()
        outcome = step.get('action_outcome')
        text = observation(step)
        refused = outcome == 'inadmissible' or bool(re.search(r'\bnothing happens\.', text, re.I))
        if refused and action:
            item = self.refused_actions.setdefault(raw_action, {'count': 0, 'evidence_steps': []})
            item['count'] += 1
            item['evidence_steps'].append(index)
            if action.startswith('take ') and self.holding['value']:
                self.refused_takes_while_holding.append({'action': raw_action, 'step': index,
                                                        'holding': deepcopy(self.holding)})
        legal = outcome == 'legal' and not refused
        if legal:
            m = re.fullmatch(r'(?:go to|open|close|examine) (' + INSTANCE + ')', action)
            if m:
                self.current = m[1]
                self._receptacle(m[1])['visited'] = fact(True, index)
            m = re.fullmatch(r'take (' + INSTANCE + r') from (' + INSTANCE + ')', action)
            if m:
                self.holding = fact(sorted(set(self.holding['value'] or []) | {m[1]}), index)
                self._set_location(m[1], 'held', index)
            m = re.fullmatch(r'(?:move|put) (' + INSTANCE + r') (?:to|in|on) (' + INSTANCE + ')', action)
            if m:
                if self.holding['value'] is not None:
                    self.holding = fact([x for x in self.holding['value'] if x != m[1]],
                                        *self.holding['evidence_steps'], index)
                self._set_location(m[1], m[2], index)
            m = re.fullmatch(r'(heat|cool|clean) (' + INSTANCE + r') with (' + INSTANCE + ')', action)
            if m:
                state = {'heat': 'hot', 'cool': 'cool', 'clean': 'clean'}[m[1]]
                states = self.state_changes.setdefault(m[2], {})
                if state in ('hot', 'cool'):
                    states.pop('cool' if state == 'hot' else 'hot', None)
                states[state] = fact(True, index)
            m = re.fullmatch(r'use (desklamp \d+)', action)
            if m:
                self.lamps[m[1]] = fact(True, index)
                for obj in self.holding['value'] or []:
                    self.state_changes.setdefault(obj, {})['lit'] = fact(
                        True, *self.holding['evidence_steps'], index)
        self._observe(text, index, action if legal else '')
        return self

    def cleared_receptacles(self):
        goal = self.goal['value']
        if not goal:
            return {}
        return {name: fact(True, *rec['contents']['evidence_steps'], -1)
                for name, rec in sorted(self.receptacles.items())
                if rec['contents']['value'] is not None and
                not any(_type(obj) == goal['object_type'] for obj in rec['contents']['value'])}

    def conditions(self):
        def condition(status='undetermined', evidence=(), **extra):
            return dict(status=status, evidence_steps=sorted(set(evidence)), **extra)
        result = {k: condition() for k in ('target_location_known', 'target_held', 'hands_free',
                                           'required_state_done', 'second_instance', 'placed_count')}
        held = self.holding['value']
        if held is not None:
            result['hands_free'] = condition('satisfied' if not held else 'known_unsatisfied',
                                             self.holding['evidence_steps'])
        goal = self.goal['value']
        if not goal:
            return result
        objects = {k: v for k, v in self.locations.items() if _type(k) == goal['object_type']}
        located = {k: v for k, v in objects.items() if v['value'] is not None}
        if located:
            result['target_location_known'] = condition('satisfied', [s for v in located.values() for s in v['evidence_steps']])
        elif objects:
            result['target_location_known'] = condition('refuted', [s for v in objects.values() for s in v['evidence_steps']])
        if held is not None:
            target_held = any(_type(x) == goal['object_type'] for x in held)
            contradicted = any(c['disagreement'] and c['step'] in self.holding['evidence_steps'] and
                              any(_type(x) == goal['object_type'] for x in c['expected']['value'] or [])
                              for c in self.inventory_checks)
            result['target_held'] = condition('satisfied' if target_held else
                'refuted' if contradicted else 'known_unsatisfied', self.holding['evidence_steps'])
        state = goal['required_state']
        evidence = [s for obj, states in self.state_changes.items() if _type(obj) == goal['object_type']
                    for s in states.get(state, {}).get('evidence_steps', [])]
        if state is None or evidence:
            result['required_state_done'] = condition('satisfied', evidence or [-1], required=state)
        if goal['count'] == 1 or len(objects) >= 2:
            result['second_instance'] = condition('satisfied', [-1] if goal['count'] == 1 else
                                                  [s for v in objects.values() for s in v['evidence_steps']])
        placed = [v for v in located.values() if v['value'] != 'held' and
                  _type(v['value']) == goal['target_receptacle_type']]
        if placed:
            result['placed_count'] = condition('satisfied' if len(placed) >= goal['count'] else 'known_unsatisfied',
                [s for v in placed for s in v['evidence_steps']], value=len(placed))
        else:
            result['placed_count'] = condition(value=None)  # No invented zero before observation.
        return result

    def to_json(self):
        return deepcopy({'schema_version': 1, 'steps_processed': self.next_step,
            'goal': self.goal, 'holding': self.holding, 'object_locations': self.locations,
            'object_sightings': self.sightings, 'receptacles': self.receptacles,
            'cleared_receptacles': self.cleared_receptacles(), 'refused_actions': self.refused_actions,
            'state_changes': self.state_changes, 'lamps_on': self.lamps,
            'inventory_checks': self.inventory_checks,
            'refused_takes_while_holding': self.refused_takes_while_holding, 'conditions': self.conditions()})

    def prompt_text(self):
        data = self.to_json()
        # No raw event text or metadata can leak via this serialization.
        keys = ('goal', 'holding', 'object_locations', 'receptacles', 'cleared_receptacles',
                'refused_actions', 'state_changes', 'lamps_on', 'conditions')
        def unknown(value):
            if value is None:
                return 'unknown'
            if isinstance(value, dict):
                return {k: 'not_required' if k in ('required_state', 'required') and v is None else unknown(v)
                        for k, v in value.items()}
            if isinstance(value, list):
                return [unknown(v) for v in value]
            return value
        return json.dumps(unknown({k: data[k] for k in keys}), sort_keys=True, separators=(',', ':'))
