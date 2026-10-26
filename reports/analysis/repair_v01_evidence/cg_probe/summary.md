# Condition-guidance probe

Report first; a teacher closed-loop follow-up requires user authorization and is outside this probe.

Seven TRAIN search failures; C/B length confound; one injection; greedy is separate; no significance claim.

## sampled

| branch | decoding | valid | expected | found_rate | median_steps_to_find | picked_rate | success_rate | follow_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | sampled | 56 | 56 | 0.3393 | 12 | 0.2143 | 0.1429 | unknown |
| B | sampled | 56 | 56 | 0.3393 | 12 | 0.2143 | 0.1607 | unknown |
| C | sampled | 56 | 56 | 0.4286 | 12.0000 | 0.2857 | 0.1429 | unknown |
| D | sampled | 56 | 56 | 0.4643 | 5.5000 | 0.3036 | 0.2321 | 0.9286 |

## greedy

| branch | decoding | valid | expected | found_rate | median_steps_to_find | picked_rate | success_rate | follow_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | greedy | 7 | 7 | 0.5714 | 20.0000 | 0.2857 | 0.1429 | unknown |
| B | greedy | 7 | 7 | 0.2857 | 9.0000 | 0.1429 | 0.1429 | unknown |
| C | greedy | 7 | 7 | 0.4286 | 12 | 0.2857 | 0.1429 | unknown |
| D | greedy | 7 | 7 | 0.5714 | 3.5000 | 0.4286 | 0.4286 | 1.0000 |

## Sampled paired comparisons

Scope: full_preregistered_probe. Case cluster bootstrap: 10,000 resamples, numpy seed 0.

| comparison | metric | difference | ci95 | cases_x_gt_y | cases_x_lt_y | decision |
| --- | --- | --- | --- | --- | --- | --- |
| C-B | found_rate | 0.0893 | [0.0, 0.23214285714285715] | 2 | 0 | no clear difference |
| D-C | found_rate | 0.0357 | [0.0, 0.10714285714285714] | 1 | 0 | no clear difference |
| B-A | found_rate | 0.0000 | [-0.05357142857142857, 0.05357142857142857] | 1 | 1 | no clear difference |
| C-A | found_rate | 0.0893 | [-0.03571428571428571, 0.25] | 2 | 1 | no clear difference |
| D-A | found_rate | 0.1250 | [0.017857142857142856, 0.26785714285714285] | 3 | 0 | no clear difference |
| D-B | found_rate | 0.1250 | [0.017857142857142856, 0.26785714285714285] | 3 | 0 | no clear difference |
| C-B | picked_rate | 0.0714 | [0.0, 0.21428571428571427] | 1 | 0 | no clear difference |
| D-C | picked_rate | 0.0179 | [0.0, 0.05357142857142857] | 1 | 0 | no clear difference |
| B-A | picked_rate | 0.0000 | [0.0, 0.0] | 0 | 0 | no clear difference |
| C-A | picked_rate | 0.0714 | [0.0, 0.21428571428571427] | 1 | 0 | no clear difference |
| D-A | picked_rate | 0.0893 | [0.0, 0.23214285714285715] | 2 | 0 | no clear difference |
| D-B | picked_rate | 0.0893 | [0.0, 0.23214285714285715] | 2 | 0 | no clear difference |
| C-B | success_rate | -0.0179 | [-0.05357142857142857, 0.0] | 0 | 1 | no clear difference |
| D-C | success_rate | 0.0893 | [0.0, 0.23214285714285715] | 2 | 0 | no clear difference |
| B-A | success_rate | 0.0179 | [0.0, 0.05357142857142857] | 1 | 0 | no clear difference |
| C-A | success_rate | 0.0000 | [0.0, 0.0] | 0 | 0 | no clear difference |
| D-A | success_rate | 0.0893 | [0.0, 0.23214285714285715] | 2 | 0 | no clear difference |
| D-B | success_rate | 0.0714 | [0.0, 0.17857142857142858] | 2 | 0 | no clear difference |

## Per case

| task_id | branch | decoding | cd_status | scheduled | expected | recorded | valid | void | missing | found_rate | median_steps_to_find | picked_rate | success_rate | follow_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train/look_at_obj_in_light-Newspaper-None-DeskLamp-224/trial_T20190909_091628_986448 | A | greedy | approved | True | 1 | 1 | 1 | 0 | 0 | 1.0000 | 12 | 0.0000 | 0.0000 | unknown |
| train/look_at_obj_in_light-Newspaper-None-DeskLamp-224/trial_T20190909_091628_986448 | A | sampled | approved | True | 8 | 8 | 8 | 0 | 0 | 0.7500 | 12.0000 | 0.0000 | 0.0000 | unknown |
| train/look_at_obj_in_light-Newspaper-None-DeskLamp-224/trial_T20190909_091628_986448 | B | greedy | approved | True | 1 | 1 | 1 | 0 | 0 | 1.0000 | 12 | 0.0000 | 0.0000 | unknown |
| train/look_at_obj_in_light-Newspaper-None-DeskLamp-224/trial_T20190909_091628_986448 | B | sampled | approved | True | 8 | 8 | 8 | 0 | 0 | 0.8750 | 12 | 0.0000 | 0.0000 | unknown |
| train/look_at_obj_in_light-Newspaper-None-DeskLamp-224/trial_T20190909_091628_986448 | C | greedy | approved | True | 1 | 1 | 1 | 0 | 0 | 1.0000 | 12 | 0.0000 | 0.0000 | unknown |
| train/look_at_obj_in_light-Newspaper-None-DeskLamp-224/trial_T20190909_091628_986448 | C | sampled | approved | True | 8 | 8 | 8 | 0 | 0 | 1.0000 | 12.0000 | 0.0000 | 0.0000 | unknown |
| train/look_at_obj_in_light-Newspaper-None-DeskLamp-224/trial_T20190909_091628_986448 | D | greedy | approved | True | 1 | 1 | 1 | 0 | 0 | 1.0000 | 1 | 0.0000 | 0.0000 | 1.0000 |
| train/look_at_obj_in_light-Newspaper-None-DeskLamp-224/trial_T20190909_091628_986448 | D | sampled | approved | True | 8 | 8 | 8 | 0 | 0 | 1.0000 | 1.0000 | 0.1250 | 0.1250 | 1.0000 |
| train/pick_and_place_simple-Kettle-None-Shelf-1/trial_T20190907_011745_444370 | A | greedy | approved | True | 1 | 1 | 1 | 0 | 0 | 1.0000 | 30 | 0.0000 | 0.0000 | unknown |
| train/pick_and_place_simple-Kettle-None-Shelf-1/trial_T20190907_011745_444370 | A | sampled | approved | True | 8 | 8 | 8 | 0 | 0 | 0.1250 | 30 | 0.0000 | 0.0000 | unknown |
| train/pick_and_place_simple-Kettle-None-Shelf-1/trial_T20190907_011745_444370 | B | greedy | approved | True | 1 | 1 | 1 | 0 | 0 | 0.0000 | unknown | 0.0000 | 0.0000 | unknown |
| train/pick_and_place_simple-Kettle-None-Shelf-1/trial_T20190907_011745_444370 | B | sampled | approved | True | 8 | 8 | 8 | 0 | 0 | 0.0000 | unknown | 0.0000 | 0.0000 | unknown |
| train/pick_and_place_simple-Kettle-None-Shelf-1/trial_T20190907_011745_444370 | C | greedy | approved | True | 1 | 1 | 1 | 0 | 0 | 0.0000 | unknown | 0.0000 | 0.0000 | unknown |
| train/pick_and_place_simple-Kettle-None-Shelf-1/trial_T20190907_011745_444370 | C | sampled | approved | True | 8 | 8 | 8 | 0 | 0 | 0.0000 | unknown | 0.0000 | 0.0000 | unknown |
| train/pick_and_place_simple-Kettle-None-Shelf-1/trial_T20190907_011745_444370 | D | greedy | approved | True | 1 | 1 | 1 | 0 | 0 | 1.0000 | 26 | 1.0000 | 1.0000 | 1.0000 |
| train/pick_and_place_simple-Kettle-None-Shelf-1/trial_T20190907_011745_444370 | D | sampled | approved | True | 8 | 8 | 8 | 0 | 0 | 0.2500 | 30.0000 | 0.0000 | 0.0000 | 1.0000 |
| train/pick_clean_then_place_in_recep-Kettle-None-DiningTable-7/trial_T20190907_144338_014196 | A | greedy | approved | True | 1 | 1 | 1 | 0 | 0 | 0.0000 | unknown | 0.0000 | 0.0000 | unknown |
| train/pick_clean_then_place_in_recep-Kettle-None-DiningTable-7/trial_T20190907_144338_014196 | A | sampled | approved | True | 8 | 8 | 8 | 0 | 0 | 0.0000 | unknown | 0.0000 | 0.0000 | unknown |
| train/pick_clean_then_place_in_recep-Kettle-None-DiningTable-7/trial_T20190907_144338_014196 | B | greedy | approved | True | 1 | 1 | 1 | 0 | 0 | 0.0000 | unknown | 0.0000 | 0.0000 | unknown |
| train/pick_clean_then_place_in_recep-Kettle-None-DiningTable-7/trial_T20190907_144338_014196 | B | sampled | approved | True | 8 | 8 | 8 | 0 | 0 | 0.0000 | unknown | 0.0000 | 0.0000 | unknown |
| train/pick_clean_then_place_in_recep-Kettle-None-DiningTable-7/trial_T20190907_144338_014196 | C | greedy | approved | True | 1 | 1 | 1 | 0 | 0 | 0.0000 | unknown | 0.0000 | 0.0000 | unknown |
| train/pick_clean_then_place_in_recep-Kettle-None-DiningTable-7/trial_T20190907_144338_014196 | C | sampled | approved | True | 8 | 8 | 8 | 0 | 0 | 0.0000 | unknown | 0.0000 | 0.0000 | unknown |
| train/pick_clean_then_place_in_recep-Kettle-None-DiningTable-7/trial_T20190907_144338_014196 | D | greedy | approved | True | 1 | 1 | 1 | 0 | 0 | 0.0000 | unknown | 0.0000 | 0.0000 | 1.0000 |
| train/pick_clean_then_place_in_recep-Kettle-None-DiningTable-7/trial_T20190907_144338_014196 | D | sampled | approved | True | 8 | 8 | 8 | 0 | 0 | 0.0000 | unknown | 0.0000 | 0.0000 | 1.0000 |
| train/pick_clean_then_place_in_recep-Lettuce-None-CounterTop-18/trial_T20190909_071155_178558 | A | greedy | approved | True | 1 | 1 | 1 | 0 | 0 | 0.0000 | unknown | 0.0000 | 0.0000 | unknown |
| train/pick_clean_then_place_in_recep-Lettuce-None-CounterTop-18/trial_T20190909_071155_178558 | A | sampled | approved | True | 8 | 8 | 8 | 0 | 0 | 0.0000 | unknown | 0.0000 | 0.0000 | unknown |
| train/pick_clean_then_place_in_recep-Lettuce-None-CounterTop-18/trial_T20190909_071155_178558 | B | greedy | approved | True | 1 | 1 | 1 | 0 | 0 | 0.0000 | unknown | 0.0000 | 0.0000 | unknown |
| train/pick_clean_then_place_in_recep-Lettuce-None-CounterTop-18/trial_T20190909_071155_178558 | B | sampled | approved | True | 8 | 8 | 8 | 0 | 0 | 0.0000 | unknown | 0.0000 | 0.0000 | unknown |
| train/pick_clean_then_place_in_recep-Lettuce-None-CounterTop-18/trial_T20190909_071155_178558 | C | greedy | approved | True | 1 | 1 | 1 | 0 | 0 | 0.0000 | unknown | 0.0000 | 0.0000 | unknown |
| train/pick_clean_then_place_in_recep-Lettuce-None-CounterTop-18/trial_T20190909_071155_178558 | C | sampled | approved | True | 8 | 8 | 8 | 0 | 0 | 0.0000 | unknown | 0.0000 | 0.0000 | unknown |
| train/pick_clean_then_place_in_recep-Lettuce-None-CounterTop-18/trial_T20190909_071155_178558 | D | greedy | approved | True | 1 | 1 | 1 | 0 | 0 | 0.0000 | unknown | 0.0000 | 0.0000 | 1.0000 |
| train/pick_clean_then_place_in_recep-Lettuce-None-CounterTop-18/trial_T20190909_071155_178558 | D | sampled | approved | True | 8 | 8 | 8 | 0 | 0 | 0.0000 | unknown | 0.0000 | 0.0000 | 1.0000 |
| train/pick_cool_then_place_in_recep-Potato-None-SideTable-21/trial_T20190907_040257_213566 | A | greedy | approved | True | 1 | 1 | 1 | 0 | 0 | 1.0000 | 6 | 1.0000 | 1.0000 | unknown |
| train/pick_cool_then_place_in_recep-Potato-None-SideTable-21/trial_T20190907_040257_213566 | A | sampled | approved | True | 8 | 8 | 8 | 0 | 0 | 1.0000 | 9.0000 | 1.0000 | 1.0000 | unknown |
| train/pick_cool_then_place_in_recep-Potato-None-SideTable-21/trial_T20190907_040257_213566 | B | greedy | approved | True | 1 | 1 | 1 | 0 | 0 | 1.0000 | 6 | 1.0000 | 1.0000 | unknown |
| train/pick_cool_then_place_in_recep-Potato-None-SideTable-21/trial_T20190907_040257_213566 | B | sampled | approved | True | 8 | 8 | 8 | 0 | 0 | 1.0000 | 6.0000 | 1.0000 | 1.0000 | unknown |
| train/pick_cool_then_place_in_recep-Potato-None-SideTable-21/trial_T20190907_040257_213566 | C | greedy | approved | True | 1 | 1 | 1 | 0 | 0 | 1.0000 | 5 | 1.0000 | 1.0000 | unknown |
| train/pick_cool_then_place_in_recep-Potato-None-SideTable-21/trial_T20190907_040257_213566 | C | sampled | approved | True | 8 | 8 | 8 | 0 | 0 | 1.0000 | 5.0000 | 1.0000 | 1.0000 | unknown |
| train/pick_cool_then_place_in_recep-Potato-None-SideTable-21/trial_T20190907_040257_213566 | D | greedy | approved | True | 1 | 1 | 1 | 0 | 0 | 1.0000 | 5 | 1.0000 | 1.0000 | 1.0000 |
| train/pick_cool_then_place_in_recep-Potato-None-SideTable-21/trial_T20190907_040257_213566 | D | sampled | approved | True | 8 | 8 | 8 | 0 | 0 | 1.0000 | 6.0000 | 1.0000 | 1.0000 | 1.0000 |
| train/pick_cool_then_place_in_recep-Apple-None-Microwave-19/trial_T20190906_210908_217201 | A | greedy | approved | True | 1 | 1 | 1 | 0 | 0 | 1.0000 | 28 | 1.0000 | 0.0000 | unknown |
| train/pick_cool_then_place_in_recep-Apple-None-Microwave-19/trial_T20190906_210908_217201 | A | sampled | approved | True | 8 | 8 | 8 | 0 | 0 | 0.5000 | 28.0000 | 0.5000 | 0.0000 | unknown |
| train/pick_cool_then_place_in_recep-Apple-None-Microwave-19/trial_T20190906_210908_217201 | B | greedy | approved | True | 1 | 1 | 1 | 0 | 0 | 0.0000 | unknown | 0.0000 | 0.0000 | unknown |
| train/pick_cool_then_place_in_recep-Apple-None-Microwave-19/trial_T20190906_210908_217201 | B | sampled | approved | True | 8 | 8 | 8 | 0 | 0 | 0.5000 | 28.0000 | 0.5000 | 0.1250 | unknown |
| train/pick_cool_then_place_in_recep-Apple-None-Microwave-19/trial_T20190906_210908_217201 | C | greedy | approved | True | 1 | 1 | 1 | 0 | 0 | 1.0000 | 28 | 1.0000 | 0.0000 | unknown |
| train/pick_cool_then_place_in_recep-Apple-None-Microwave-19/trial_T20190906_210908_217201 | C | sampled | approved | True | 8 | 8 | 8 | 0 | 0 | 1.0000 | 28.0000 | 1.0000 | 0.0000 | unknown |
| train/pick_cool_then_place_in_recep-Apple-None-Microwave-19/trial_T20190906_210908_217201 | D | greedy | approved | True | 1 | 1 | 1 | 0 | 0 | 1.0000 | 2 | 1.0000 | 1.0000 | 1.0000 |
| train/pick_cool_then_place_in_recep-Apple-None-Microwave-19/trial_T20190906_210908_217201 | D | sampled | approved | True | 8 | 8 | 8 | 0 | 0 | 1.0000 | 5.5000 | 1.0000 | 0.5000 | 0.5000 |
| train/pick_cool_then_place_in_recep-Egg-None-Microwave-5/trial_T20190908_005812_214691 | A | greedy | approved | True | 1 | 1 | 1 | 0 | 0 | 0.0000 | unknown | 0.0000 | 0.0000 | unknown |
| train/pick_cool_then_place_in_recep-Egg-None-Microwave-5/trial_T20190908_005812_214691 | A | sampled | approved | True | 8 | 8 | 8 | 0 | 0 | 0.0000 | unknown | 0.0000 | 0.0000 | unknown |
| train/pick_cool_then_place_in_recep-Egg-None-Microwave-5/trial_T20190908_005812_214691 | B | greedy | approved | True | 1 | 1 | 1 | 0 | 0 | 0.0000 | unknown | 0.0000 | 0.0000 | unknown |
| train/pick_cool_then_place_in_recep-Egg-None-Microwave-5/trial_T20190908_005812_214691 | B | sampled | approved | True | 8 | 8 | 8 | 0 | 0 | 0.0000 | unknown | 0.0000 | 0.0000 | unknown |
| train/pick_cool_then_place_in_recep-Egg-None-Microwave-5/trial_T20190908_005812_214691 | C | greedy | approved | True | 1 | 1 | 1 | 0 | 0 | 0.0000 | unknown | 0.0000 | 0.0000 | unknown |
| train/pick_cool_then_place_in_recep-Egg-None-Microwave-5/trial_T20190908_005812_214691 | C | sampled | approved | True | 8 | 8 | 8 | 0 | 0 | 0.0000 | unknown | 0.0000 | 0.0000 | unknown |
| train/pick_cool_then_place_in_recep-Egg-None-Microwave-5/trial_T20190908_005812_214691 | D | greedy | approved | True | 1 | 1 | 1 | 0 | 0 | 0.0000 | unknown | 0.0000 | 0.0000 | 1.0000 |
| train/pick_cool_then_place_in_recep-Egg-None-Microwave-5/trial_T20190908_005812_214691 | D | sampled | approved | True | 8 | 8 | 8 | 0 | 0 | 0.0000 | unknown | 0.0000 | 0.0000 | 1.0000 |

Unknown rates indicate missing/void rollouts or unavailable C/D; they are never scored as failures.
