Scripts behind ../teacher_only_tasks_ZH.md (10-26). hk_extract.py runs on hpg (`python3 - want.json < hk_extract.py > hk_episodes_all.json`,
want.json = sorted dev140 task ids) and writes compact episodes (actions, observations, first 300 chars of replies) of every E4B
grid cell + teacher; the 29 MB output is not committed. hk_an3.py = failure-mode classifier (A..F) and per-K shares; hk_an4.py <phrasing>
= per-task modes; hk_tail.py <type> <mode> <Ks> <n> = what students do after taking the object. Cells exclude SAD K10 (collapsed, 8/140).
dev_goals.json = dev140 goal texts; dev_phr.json = phrasing class; hk_sets_33.json = rescued (R) / persistent (P) task lists.
