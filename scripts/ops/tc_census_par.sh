#!/bin/bash
# tc_census_par.sh — READ-ONLY parallel census of trees on hpg Lustre. The only thing it writes is OUTDIR.
# usage (ON hpg): bash tc_census_par.sh OUTDIR NWORKERS DEPTH ROOT [ROOT...]
# Work units = every directory at depth DEPTH under each root (one `lfs find --lazy` walk each, NWORKERS at a
# time, pooled over all roots and shuffled so partial results are representative) + one "(shallow)" unit per
# root for the regular files at depth <= DEPTH. Every regular file is counted exactly once.
# OUTDIR/units.tsv (tab-separated, appended as units finish):
#   U <root> <unit relpath> <files> <bytes> <c1..c6> <b1..b6>   size bins: <4K <64K <1M <64M <1G >=1G
#   B <root> <size> <relpath>                                    every file >= 64 MiB
#   E <root> <unit> <RC ...>                                     a unit whose walk did not exit 0
# OUTDIR/log: "UNITS n" once the unit list exists, last line CENSUS_PAR_DONE.
# Sizes are the MDS lazy sizes (no OST glimpse per file); validate on a sample before relying on them.
set -u -o pipefail
OUT="${1:?outdir}"; NW="${2:?workers}"; DEPTH="${3:?depth}"; shift 3
case "$OUT" in /orange/yd24f.fsu/xc25.fsu/tc-cold/_ops/*) ;; *) echo "REFUSE: outdir must be under tc-cold/_ops"; exit 2;; esac
case "$NW" in 1|2|3|4) ;; *) echo "REFUSE: 1-4 workers"; exit 2;; esac
mkdir -p "$OUT" || exit 1
[ ! -e "$OUT/units.tsv" ] || { echo "REFUSE: $OUT/units.tsv exists"; exit 2; }
export OUT
export AWKP='{ s=$1+0; n++; b+=s; h=(s<4096?1:(s<65536?2:(s<1048576?3:(s<67108864?4:(s<1073741824?5:6))))); hc[h]++; hb[h]+=s;
  if (s>=67108864) printf "B\t%s\t%d\t%s\n", root, s, substr($2, length(root)+2) }
END { printf "U\t%s\t%s\t%d\t%d", root, (unit==root ? "(shallow)" : substr(unit, length(root)+2)), n+0, b+0;
  for (i=1;i<=6;i++) printf "\t%d", hc[i]+0; for (i=1;i<=6;i++) printf "\t%d", hb[i]+0; printf "\n" }'
emit() {  # $1 root, $2 unit, $3 result text ending in an "RC ..." line; append under a lock
  local rcl; rcl=$(printf '%s\n' "$3" | grep '^RC ')
  { flock 8
    printf '%s\n' "$3" | grep -v '^RC ' >> "$OUT/units.tsv"
    [ "$rcl" = "RC 0 0" ] || printf 'E\t%s\t%s\t%s\n' "$1" "$2" "$rcl" >> "$OUT/units.tsv"
  } 8>>"$OUT/.append.lock"
}
unit() {  # $1 root, $2 unit dir
  local res
  res=$(lfs find "$2" -type f --lazy -printf '%s\t%p\n' | awk -F'\t' -v root="$1" -v unit="$2" "$AWKP"; echo "RC ${PIPESTATUS[*]}")
  emit "$1" "$2" "$res"
}
export -f unit emit
echo "$(date '+%T') start host=$(hostname) workers=$NW depth=$DEPTH roots=$#" >> "$OUT/log"
: > "$OUT/units.list0"
for ROOT in "$@"; do
  ROOT="${ROOT%/}"
  [ -d "$ROOT" ] && [ ! -L "$ROOT" ] || { echo "SKIP not a dir: $ROOT" >> "$OUT/log"; continue; }
  res=$(find "$ROOT" -maxdepth "$DEPTH" -type f -printf '%s\t%p\n' | awk -F'\t' -v root="$ROOT" -v unit="$ROOT" "$AWKP"; echo "RC ${PIPESTATUS[*]}")
  emit "$ROOT" "$ROOT" "$res"
  find "$ROOT" -mindepth "$DEPTH" -maxdepth "$DEPTH" -type d -printf "$ROOT\\0%p\\0" >> "$OUT/units.list0" || echo "WARN unit listing failed for $ROOT" >> "$OUT/log"
  echo "$(date '+%T') listed $ROOT" >> "$OUT/log"
done
N=$(( $(tr -cd '\0' < "$OUT/units.list0" | wc -c) / 2 ))
echo "$(date '+%T') UNITS $N" >> "$OUT/log"
# shuffle the (root, unit) pairs, keeping each pair together
tr '\0\n' '\n\0' < "$OUT/units.list0" | paste -d '\t' - - | shuf | tr '\t' '\n' | tr '\n\0' '\0\n' \
  | xargs -0 -r -n 2 -P "$NW" bash -c 'unit "$1" "$2"' _ ; rc=$?
echo "$(date '+%T') xargs rc=$rc" >> "$OUT/log"
echo "$(date '+%T') CENSUS_PAR_DONE" >> "$OUT/log"
