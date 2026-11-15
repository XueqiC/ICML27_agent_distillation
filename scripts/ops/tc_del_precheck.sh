#!/bin/bash
# tc_del_precheck.sh — READ-ONLY checks before deleting the 11-15 approved items (A, B). Runs ON hpg.
# 1. shows the top level of every directory that is a delete target (look before deleting);
# 2. for every environment that is KEPT, resolves all its symlinks, .pth lines and editable-install records and
#    reports any that point into a path that is about to be deleted (list B).
set -u
D=/blue/fsu-compsci-dept/xc25.fsu/hq; Y=/blue/yd24f.fsu/xc25.fsu/hq
BRE='^/blue/fsu-compsci-dept/xc25\.fsu/hq/tc-alignment/envs/(bfcl|bfcl-venv|tau2|tau-bench|appworld-data|appworld-official|appworld-repo|appworld-venv|appworld-data-downloaded|ds1000|bird|\.python)(/|$)|^/blue/yd24f\.fsu/xc25\.fsu/hq/tc-alignment/envs(/|$)'
echo "== 1. targets: top level"
for t in "$D/ICML27_agent_distillation/_trash" "$D/ICML27_v0_runs/_failed" "$Y/_trash/ICML27_agent_distillation/superseded_1007" \
         "$D/ICML27_agent_distillation/reports/smartad-selection" "$D/ICML27_agent_distillation/reports/runs/smartad-collections" \
         "$D/ICML27_agent_distillation/reports/adftp-collections" "$D/tc-alignment-vllm/envs" "$D/tc-alignment/envs/vllm-serve" \
         "$D/tc-alignment/envs/.python" "$D/tc-alignment/envs/.uv-python" "$Y/tc-alignment/envs/.uv-python"; do
  echo "-- $t"; ls -la "$t" 2>&1 | tail -n +2 | awk '{printf "   %s %s %s %s %s %s %s %s\n", $1,$5,$6,$7,$8,$9,$10,$11}' | head -n 40
done
echo "== 2. kept envs: links into paths that will be deleted"
KEPT=( "$Y/tc-alignment/.venv" "$D/tc-alignment/envs/vllm" "$D/tc-alignment/envs/alfworld" "$D/tc-alignment/envs/vllm-serve"
       "$D/envs/p1-venv" "$D/ICML27_agent_distillation/.venv" "$D/tc-alignment-vllm" "$D/tc-alignment/envs/.uv-python" )
for k in "${KEPT[@]}"; do
  [ -e "$k" ] || { echo "-- $k MISSING"; continue; }
  nl=$(lfs find "$k" -type l 2>/dev/null | wc -l)
  bad=$(lfs find "$k" -type l -print0 2>/dev/null | xargs -0 -r -n 200 realpath -m -- 2>/dev/null | grep -c -E "$BRE")
  # .pth lines and editable-install records (text): any absolute path into a B directory
  pth=$( { find "$k" -maxdepth 8 \( -name '*.pth' -o -name 'direct_url.json' -o -name '__editable__*' -o -name 'pyvenv.cfg' \) -type f -size -1M -print0 2>/dev/null \
           | xargs -0 -r grep -l -E '/tc-alignment/envs/(bfcl|bfcl-venv|tau2|tau-bench|appworld-data|appworld-official|appworld-repo|appworld-venv|appworld-data-downloaded|ds1000|bird|\.python)(/|"|$)|/blue/yd24f\.fsu/xc25\.fsu/hq/tc-alignment/envs(/|"|$)' 2>/dev/null; } | wc -l)
  echo "-- $k symlinks=$nl into_B=$bad text_refs_into_B=$pth"
  if [ "$bad" != 0 ]; then lfs find "$k" -type l -print0 | xargs -0 -r -n 200 realpath -m -- 2>/dev/null | grep -E "$BRE" | sed -E 's#^((/[^/]+){9}).*#\1#' | sort | uniq -c | head -n 20; fi
  if [ "$pth" != 0 ]; then find "$k" -maxdepth 8 \( -name '*.pth' -o -name 'direct_url.json' -o -name '__editable__*' -o -name 'pyvenv.cfg' \) -type f -size -1M -print0 2>/dev/null \
      | xargs -0 -r grep -H -o -E '/blue/[^"[:space:]]*tc-alignment/envs/[^"[:space:]]*' 2>/dev/null | cut -c1-300 | head -n 20; fi
done
echo "== 3. who else points at the B python (.python) or the old uv python"
for v in "$D/tc-alignment/envs"/*/.venv "$D/tc-alignment/envs"/*/pyvenv.cfg "$Y/tc-alignment/envs"/*/pyvenv.cfg; do
  [ -e "$v" ] || continue
  c="$v"; [ -d "$v" ] && c="$v/pyvenv.cfg"
  printf '   %s | %s\n' "$c" "$(grep -E '^home' "$c" 2>/dev/null | head -n 1)"
done
echo PRECHECK_DONE
