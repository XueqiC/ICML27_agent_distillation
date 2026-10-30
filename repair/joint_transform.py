"""Opt-in transform preconditions for joint-review/2, from public history only.

Like FactTable, input is a complete initial observation plus executed prefix.
transform_done describes a witnessed execution in that prefix, not an inferred
initial temperature/cleanliness. required_state_done retains V1's unknowns.
"""
from copy import deepcopy
import re

from repair.facts import FactTable, INSTANCE, fact, observation, _type

TEMPLATE_VERSION = 'joint-review/2'
ENVIRONMENT_RULE = """
Environment rule: Transform with `heat <obj> with <microwave>`, `cool <obj> with <fridge>`,
or `clean <obj> with <sinkbasin>` (use instance names). Hold the object and be at the
appliance. Putting it inside or opening/closing the appliance does not transform it;
if the target is inside and not held, take it first. When transform_ready is satisfied
and the transform command is admissible, that command is the next step.
"""
TRANSFORMS = {'hot': ('heat', 'microwave'), 'cool': ('cool', 'fridge'),
              'clean': ('clean', 'sinkbasin')}


def _condition(status='undetermined', evidence=(), **extra):
    return dict(status=status, evidence_steps=sorted(set(evidence)), **extra)


class TransformFactTable(FactTable):
    def __init__(self, initial=''):
        super().__init__(initial)
        self.current_location = fact(None)
        self.placements = {}
        self.uncertain_transforms = {}
        self._arrival(observation(initial), -1)

    def _arrival(self, text, index):
        match = re.search(r'you (?:arrive at|are (?:at|on|in)) (?:the )?('
                          + INSTANCE + ')', text.lower())
        if match:
            self.current_location = fact(match[1], index)

    def update(self, step):
        index = self.next_step
        action = (step.get('action_parsed') or '').strip().lower()
        text, outcome = observation(step), step.get('action_outcome')
        refused = outcome == 'inadmissible' or bool(re.search(r'\bnothing happens\.', text, re.I))
        legal = outcome == 'legal' and not refused
        super().update(step)
        if legal:
            move = re.fullmatch(r'go to (' + INSTANCE + ')', action)
            if move:
                self.current_location = fact(move[1], index)
            put = re.fullmatch(r'(?:move|put) (' + INSTANCE + r') (?:to|in|on) (' + INSTANCE + ')', action)
            if put:
                self.placements[put[1]] = fact(put[2], index)
        self._arrival(text, index)
        self.placements = {obj: place for obj, place in self.placements.items()
                           if self.locations.get(obj, {}).get('value') == place['value']}
        # An attempted transform with no known outcome is not evidence of either
        # completion or non-completion. A subsequent witnessed state supersedes it.
        transform = re.fullmatch(r'(heat|cool|clean) (' + INSTANCE + r') with (' + INSTANCE + ')', action)
        if transform and not refused and outcome != 'parser_error':
            state = {'heat': 'hot', 'cool': 'cool', 'clean': 'clean'}[transform[1]]
            affected = ('hot', 'cool') if state in ('hot', 'cool') else ('clean',)
            for value in affected:
                key = (transform[2], value)
                if legal:
                    self.uncertain_transforms.pop(key, None)
                else:
                    self.uncertain_transforms[key] = index
        return self

    def conditions(self):
        result = super().conditions()
        result.update({key: _condition() for key in ('required_transform', 'transform_done',
                       'at_required_appliance', 'target_inside_appliance', 'transform_ready')})
        goal = self.goal['value']
        if not goal or goal['required_state'] not in TRANSFORMS:
            return result
        state = goal['required_state']
        verb, appliance = TRANSFORMS[state]
        goal_steps = self.goal['evidence_steps']
        result['required_transform'] = _condition('satisfied', goal_steps,
            value={'command': verb, 'appliance_type': appliance, 'required_state': state})
        uncertain = [index for (obj, value), index in self.uncertain_transforms.items()
                     if _type(obj) == goal['object_type'] and value == state]
        if uncertain:
            result['transform_done'] = _condition(evidence=[*goal_steps, *uncertain])
        elif result['required_state_done']['status'] == 'satisfied':
            result['transform_done'] = deepcopy(result['required_state_done'])
        else:
            # Absence is over the complete public execution prefix, not hidden
            # world state. Cite that entire prefix (including the initial goal).
            result['transform_done'] = _condition('known_unsatisfied', range(-1, self.next_step))
        location = self.current_location
        if location['value'] is not None:
            result['at_required_appliance'] = _condition(
                'satisfied' if _type(location['value']) == appliance else 'known_unsatisfied',
                [*goal_steps, *location['evidence_steps']], value=location['value'])
        held = result['target_held']
        objects = {obj: loc for obj, loc in self.locations.items() if _type(obj) == goal['object_type']}
        # Last known public locations include sightings as well as placements.
        inside = [obj for obj, loc in objects.items() if loc['value'] not in (None, 'held')
                  and _type(loc['value']) == appliance]
        if held['status'] == 'satisfied':
            result['target_inside_appliance'] = _condition('known_unsatisfied', [*goal_steps, *held['evidence_steps']])
        elif held['status'] in ('known_unsatisfied', 'refuted'):
            if inside:
                evidence = [s for obj in inside for entry in (objects[obj], self.placements.get(obj, {}))
                            for s in entry.get('evidence_steps', [])]
                result['target_inside_appliance'] = _condition('satisfied',
                    [*goal_steps, *held['evidence_steps'], *evidence], instances=sorted(inside))
            elif objects and all(loc['value'] not in (None, 'held') and _type(loc['value']) != appliance
                                 for loc in objects.values()):
                result['target_inside_appliance'] = _condition('known_unsatisfied',
                    [*goal_steps, *held['evidence_steps'],
                     *(s for loc in objects.values() for s in loc['evidence_steps'])])
        at, done = result['at_required_appliance'], result['transform_done']
        blockers = [c for c in (held, at) if c['status'] in ('known_unsatisfied', 'refuted')]
        if done['status'] == 'satisfied':
            blockers.append(done)
        if blockers:
            result['transform_ready'] = _condition('known_unsatisfied',
                [*goal_steps, *(s for c in blockers for s in c['evidence_steps'])])
        elif held['status'] == at['status'] == 'satisfied' and done['status'] == 'known_unsatisfied':
            result['transform_ready'] = _condition('satisfied',
                [*goal_steps, *(s for c in (held, at, done) for s in c['evidence_steps'])])
        return result
