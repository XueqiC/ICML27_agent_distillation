#!/bin/bash
# tc_refscan.sh — READ-ONLY. Before deleting the "D" directories (12B / E2B SmartAD selection + K100 collection
# records), find every reference to them from OUTSIDE them in tc's trees on hpg:
#   (1) file contents: grep -r over every non-binary file (venvs, .git, __pycache__ skipped);
#   (2) symlinks: every symlink, resolved; flagged when it resolves into a D directory.
# usage (ON hpg): bash tc_refscan.sh OUTDIR NWORKERS(1-4)
# OUTDIR/matches.tsv : M <file>             a file whose text names a D directory
# OUTDIR/links.tsv   : L <link> <resolved>  a symlink that resolves into a D directory
# OUTDIR/units.tsv   : U <kind R|S> <unit> <grep rc> <n symlinks>   one line per finished unit; E lines = errors
# OUTDIR/log         : ends with REFSCAN_DONE units=<n>
set -u -o pipefail
OUT="${1:?outdir}"; NW="${2:?workers}"
case "$OUT" in /orange/yd24f.fsu/xc25.fsu/tc-cold/_ops/*) ;; *) echo "REFUSE: outdir must be under tc-cold/_ops"; exit 2;; esac
case "$NW" in 1|2|3|4) ;; *) echo "REFUSE: 1-4 workers"; exit 2;; esac
mkdir -p "$OUT" || exit 1
[ ! -e "$OUT/log" ] || { echo "REFUSE: $OUT/log exists"; exit 2; }
D=/blue/fsu-compsci-dept/xc25.fsu/hq; Y=/blue/yd24f.fsu/xc25.fsu/hq; A=$D/ICML27_agent_distillation
# the D directories (never scanned: they are the delete candidates themselves)
DDIRS=(
  "$A/reports/smartad-selection/chain_k500_s0_r2_e2b"
  "$A/reports/smartad-selection/chain_k500_s0_r2"
  "$A/reports/smartad-selection/k100_s0" "$A/reports/smartad-selection/k100_s1" "$A/reports/smartad-selection/k100_s2"
  "$A/reports/runs/smartad-collections/k100_s0" "$A/reports/runs/smartad-collections/k100_s1" "$A/reports/runs/smartad-collections/k100_s2"
  "$A/reports/adftp-collections/k100_s0" "$A/reports/adftp-collections/k100_s1" "$A/reports/adftp-collections/k100_s2"
)
# also not scanned: the approved-for-deletion A / B directories
SKIP=( "${DDIRS[@]}" "$A/_trash" "$D/ICML27_v0_runs/_failed"
  "$D/tc-alignment/envs/bfcl" "$D/tc-alignment/envs/bfcl-venv" "$D/tc-alignment/envs/tau2" "$D/tc-alignment/envs/tau-bench"
  "$D/tc-alignment/envs/appworld-data" "$D/tc-alignment/envs/appworld-official" "$D/tc-alignment/envs/appworld-repo"
  "$D/tc-alignment/envs/appworld-venv" "$D/tc-alignment/envs/appworld-data-downloaded" "$D/tc-alignment/envs/ds1000"
  "$D/tc-alignment/envs/bird" "$D/tc-alignment/envs/.python" "$Y/tc-alignment/envs" )
# directories known to be huge: split one more level so the workers stay balanced
BIG=( "$A/reports/runs/eval" "$A/reports/smartad-selection/chain_k500_s0_r2_e4b" "$A/reports/smartad-selection/chain_k500_s0_r2_e4b/collection"
      "$A/data/verified/seeds_full_train_r2" "$A/reports/runs/smartad-chain" "$A/reports/runs/smartad-chain/chain_k500_s0_r2"
      "$A/reports/runs/adftp-chain" "$A/reports/runs/adftp-chain/chain_k500_s0_r2" "$A/reports/runs/baselines" )
ROOTS=( "$A" "$D/ICML27_v0_runs" "$D/ICML27_runs" "$D/ICML27_p1_backup" "$D/ICML27_eval_46b8527" "$D/tc-alignment-vllm"
        "$D/tc-alignment" "$D/.sed-hash-receipts" "$Y/ICML27_archive" "$Y/tc-alignment" )
MIND=3
export OUT
export PAT='smartad-selection/(chain_k500_s0_r2_e2b|chain_k500_s0_r2([^_[:alnum:]]|$)|k100_s[0-9])|smartad-collections/k100_s[0-9]|adftp-collections/k100_s[0-9]'
export DRE='^/blue/fsu-compsci-dept/xc25\.fsu/hq/ICML27_agent_distillation/reports/(smartad-selection/(chain_k500_s0_r2_e2b|chain_k500_s0_r2|k100_s[0-9])|runs/smartad-collections/k100_s[0-9]|adftp-collections/k100_s[0-9])(/|$)'
inlist() { local x="$1"; shift; local e; for e in "$@"; do [ "$e" = "$x" ] && return 0; done; return 1; }
isanc() { local x="$1"; shift; local e; for e in "$@"; do case "$e" in "$x"/*) return 0;; esac; done; return 1; }
: > "$OUT/units.list0"
expand() {  # $1 dir, $2 depth below its root
  local d="$1" dep="$2" c
  if inlist "$d" "${SKIP[@]}"; then echo "SKIP $d" >> "$OUT/log"; return; fi
  if [ "$dep" -lt "$MIND" ] || isanc "$d" "${SKIP[@]}" || inlist "$d" "${BIG[@]}"; then
    printf 'S\0%s\0' "$d" >> "$OUT/units.list0"
    while IFS= read -r -d '' c; do
      case "${c##*/}" in .venv|.git|__pycache__|node_modules|site-packages) continue;; esac
      expand "$c" $((dep + 1))
    done < <(find "$d" -mindepth 1 -maxdepth 1 -type d -print0)
  else
    printf 'R\0%s\0' "$d" >> "$OUT/units.list0"
  fi
}
unit() {  # $1 kind (R recursive | S only the entries directly in the dir), $2 dir
  local kind="$1" u="$2" m rc nl bad
  if [ "$kind" = R ]; then
    m=$(grep -r -I -l -E "$PAT" --exclude-dir=.venv --exclude-dir=site-packages --exclude-dir=node_modules --exclude-dir=.git --exclude-dir=__pycache__ -- "$u" 2>>"$OUT/grep.err"); rc=$?
    nl=$(find "$u" -type l -printf x 2>/dev/null | wc -c)
    bad=$(find "$u" -type l -print0 2>/dev/null | xargs -0 -r -n 100 realpath -m -- 2>/dev/null | grep -E "$DRE")
    [ -z "$bad" ] || bad=$(find "$u" -type l -print0 2>/dev/null | while IFS= read -r -d '' l; do t=$(realpath -m -- "$l" 2>/dev/null); printf '%s\n' "$t" | grep -q -E "$DRE" && printf 'L\t%s\t%s\n' "$l" "$t"; done)
  else
    m=$(find "$u" -mindepth 1 -maxdepth 1 -type f -print0 2>/dev/null | xargs -0 -r grep -I -l -E "$PAT" -- 2>>"$OUT/grep.err"); rc=$?
    nl=$(find "$u" -mindepth 1 -maxdepth 1 -type l -printf x 2>/dev/null | wc -c)
    bad=$(find "$u" -mindepth 1 -maxdepth 1 -type l -print0 2>/dev/null | while IFS= read -r -d '' l; do t=$(realpath -m -- "$l" 2>/dev/null); printf '%s\n' "$t" | grep -q -E "$DRE" && printf 'L\t%s\t%s\n' "$l" "$t"; done)
  fi
  { flock 8
    [ -z "$m" ] || printf '%s\n' "$m" | sed 's/^/M\t/' >> "$OUT/matches.tsv"
    [ -z "$bad" ] || printf '%s\n' "$bad" >> "$OUT/links.tsv"
    printf 'U\t%s\t%s\t%s\t%s\n' "$kind" "$u" "$rc" "$nl" >> "$OUT/units.tsv"
    # grep: 0 = match, 1 = no match; with xargs 123 = some invocation returned 1 (no match). Anything else is an error.
    case "$rc" in 0|1|123) ;; *) printf 'E\t%s\t%s\tgrep rc=%s\n' "$kind" "$u" "$rc" >> "$OUT/units.tsv";; esac
  } 8>>"$OUT/.append.lock"
}
export -f unit
echo "$(date '+%T') start host=$(hostname) workers=$NW" >> "$OUT/log"
: > "$OUT/matches.tsv"; : > "$OUT/links.tsv"; : > "$OUT/units.tsv"; : > "$OUT/grep.err"
for r in "${ROOTS[@]}"; do
  [ -d "$r" ] && [ ! -L "$r" ] || { echo "SKIP not a dir: $r" >> "$OUT/log"; continue; }
  expand "$r" 0
  echo "$(date '+%T') listed $r" >> "$OUT/log"
done
N=$(( $(tr -cd '\0' < "$OUT/units.list0" | wc -c) / 2 ))
echo "$(date '+%T') UNITS $N" >> "$OUT/log"
tr '\0\n' '\n\0' < "$OUT/units.list0" | paste -d '\t' - - | shuf | tr '\t' '\n' | tr '\n\0' '\0\n' \
  | xargs -0 -r -n 2 -P "$NW" bash -c 'unit "$1" "$2"' _ ; rc=$?
echo "$(date '+%T') xargs rc=$rc" >> "$OUT/log"
echo "$(date '+%T') REFSCAN_DONE units=$(grep -c '^U' "$OUT/units.tsv") errors=$(grep -c '^E' "$OUT/units.tsv") matches=$(wc -l < "$OUT/matches.tsv") links=$(wc -l < "$OUT/links.tsv")" >> "$OUT/log"
