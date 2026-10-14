#!/bin/bash
# Lower pending ICML27 job requests to measured usage (sacct 10-01..10-14). Each line: jobid cpus mem_MB
set -u
while read -r j c m; do
  [ -z "$j" ] && continue
  st=$(squeue -h -j "$j" -o '%T|%u' 2>/dev/null)
  [[ "$st" == PENDING\|xc25.fsu ]] || { echo "SKIP $j ($st)"; continue; }
  args=(JobId="$j")
  [ "$c" != - ] && args+=(MinCPUsNode="$c" NumCPUs="$c" CPUsPerTask="$c")
  [ "$m" != - ] && args+=(MinMemoryNode="$m")
  scontrol update "${args[@]}"; rc=$?
  scontrol show job "$j" | tr ' ' '\n' | grep -E '^(JobName|Reason|NumCPUs|CPUs/Task|MinCPUsNode|MinMemoryNode)=' | tr '\n' ' '; echo " rc=$rc"
done <<'LIST'
45729774 3 49152
45729775 3 49152
45729776 3 49152
45729777 3 49152
45754199 3 49152
45733065 3 49152
45733059 3 49152
45733061 3 49152
45733063 3 49152
45733067 3 49152
45867088 3 49152
45867089 3 49152
45867090 - 49152
45754200 - 49152
45733060 - 49152
45733062 - 49152
45733064 - 49152
45733066 - 49152
45733068 - 49152
45470256 - 49152
45470257 4 -
45470259 4 -
45470261 4 -
45470263 4 -
45470265 4 -
45470267 4 -
45058904 4 -
LIST
