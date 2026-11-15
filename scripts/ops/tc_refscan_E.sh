#!/bin/bash
# tc_refscan_E.sh — READ-ONLY. Before deleting item E (the LEAP early-ladder weights of
# YD/ICML27_archive/baselines_evaluated_1007, 38 big files now on /orange), find every reference to them from
# OUTSIDE that archive directory in tc's trees on hpg. Same engine as tc_refscan.sh, different constants:
#   (1) file contents: grep -r over every non-binary file (venvs, .git, __pycache__ skipped) for the archive
#       name or for a path into a LEAP cell's stages/ directory (catches the pre-archive paths too);
#   (2) symlinks: every symlink, resolved; flagged when it resolves into the archive (/blue or /orange copy).
# usage (ON hpg): bash tc_refscan_E.sh OUTDIR NWORKERS(1-4)
# OUTDIR/matches.tsv : M <file>             a file whose text names the archive or a LEAP stages path
# OUTDIR/links.tsv   : L <link> <resolved>  a symlink that resolves into the archive
# OUTDIR/units.tsv   : U <kind R|S> <unit> <grep rc> <n symlinks>   one line per finished unit; E lines = errors
# OUTDIR/log         : ends with REFSCAN_DONE units=<n>
set -u -o pipefail
OUT="${1:?outdir}"; NW="${2:?workers}"
case "$OUT" in /orange/yd24f.fsu/xc25.fsu/tc-cold/_ops/*) ;; *) echo "REFUSE: outdir must be under tc-cold/_ops"; exit 2;; esac
case "$NW" in 1|2|3|4) ;; *) echo "REFUSE: 1-4 workers"; exit 2;; esac
mkdir -p "$OUT" || exit 1
[ ! -e "$OUT/log" ] || { echo "REFUSE: $OUT/log exists"; exit 2; }
D=/blue/fsu-compsci-dept/xc25.fsu/hq; Y=/blue/yd24f.fsu/xc25.fsu/hq; A=$D/ICML27_agent_distillation
# never scanned: the archive directory itself (its own small files stay on /blue and name their own paths)
SKIP=( "$Y/ICML27_archive/baselines_evaluated_1007" )
# directories known to be huge: split one more level so the workers stay balanced
BIG=( "$A/reports/runs/eval" "$A/reports/smartad-selection/chain_k500_s0_r2_e4b" "$A/reports/smartad-selection/chain_k500_s0_r2_e4b/collection"
      "$A/data/verified/seeds_full_train_r2" "$A/reports/runs/smartad-chain" "$A/reports/runs/smartad-chain/chain_k500_s0_r2"
      "$A/reports/runs/adftp-chain" "$A/reports/runs/adftp-chain/chain_k500_s0_r2" "$A/reports/runs/baselines" )
ROOTS=( "$A" "$D/ICML27_v0_runs" "$D/ICML27_runs" "$D/ICML27_p1_backup" "$D/ICML27_eval_46b8527" "$D/tc-alignment-vllm"
        "$D/tc-alignment" "$D/.sed-hash-receipts" "$Y/ICML27_archive" "$Y/tc-alignment" )
MIND=3
export OUT
export PAT='baselines_evaluated_1007|leap-(privileged|zero)-k[0-9]+s[0-9](-m[0-9]+)?/stages'
export DRE='^/blue/yd24f\.fsu/xc25\.fsu/hq/ICML27_archive/baselines_evaluated_1007(/|$)|^/orange/yd24f\.fsu/xc25\.fsu/icml27-cold/blue-yd24f/ICML27_archive(/|$)'
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
