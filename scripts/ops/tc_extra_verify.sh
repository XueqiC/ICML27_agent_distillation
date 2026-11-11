#!/bin/bash
# tc_extra_verify.sh — READ-ONLY deep check of a finished copy, run between orange_move.sh's PASS and its
# --delete-source re-run. orange_move.sh hashes only the largest file DIRECTLY inside each immediate
# subdirectory (find -maxdepth 1), which never reaches nested checkpoints. This adds:
#   1. sha256 of the NTOP largest regular files anywhere in the tree,
#   2. sha256 of NSAMPLE regular files drawn with a fixed seed,
#   3. identical symlink manifest (relative path + target) and identical directory count.
# Writes nothing anywhere. Runs ON hpg:  ssh hpg 'bash -s -- SRC DST [NTOP] [NSAMPLE]' < tc_extra_verify.sh
# The last line is the verdict: EXTRA_PASS or EXTRA_FAIL.
set -u -o pipefail
SRC="${1:?src dir}"; DST="${2:?dst dir}"; NTOP="${3:-20}"; NSAMPLE="${4:-200}"
SRC="${SRC%/}"; DST="${DST%/}"
[ -d "$SRC" ] && [ -d "$DST" ] || { echo "EXTRA_FAIL: src or dst is not a directory"; exit 2; }
ts(){ date '+%H:%M:%S'; }
echo "$(ts) SRC=$SRC"; echo "$(ts) DST=$DST  NTOP=$NTOP NSAMPLE=$NSAMPLE"

# One walk of SRC: pick the files to hash (top by size + seeded reservoir sample), count dirs, hash symlinks.
sel=$(find "$SRC" \( -type f -o -type l -o -type d \) -printf '%y\t%s\t%P\t%l\n' | awk -F'\t' -v ntop="$NTOP" -v ns="$NSAMPLE" '
  BEGIN { srand(20261111) }
  $1=="d" { nd++; next }
  $1=="l" { nl++; next }
  $1=="f" {
    nf++; bytes+=$2
    # top-ntop by size: keep a small array, replace the current minimum
    if (nt < ntop) { nt++; ts[nt]=$2; tp[nt]=$3 }
    else { m=1; for (i=2;i<=nt;i++) if (ts[i]<ts[m]) m=i; if ($2>ts[m]) { ts[m]=$2; tp[m]=$3 } }
    # reservoir sample of ns paths
    if (nf <= ns) { sp[nf]=$3 } else { j=int(rand()*nf)+1; if (j<=ns) sp[j]=$3 }
  }
  END {
    printf "META\t%d\t%d\t%d\t%d\n", nf, nd, nl, bytes
    for (i=1;i<=nt;i++) printf "TOP\t%s\n", tp[i]
    n=(nf<ns?nf:ns); for (i=1;i<=n;i++) printf "SMP\t%s\n", sp[i]
  }') || { echo "EXTRA_FAIL: walk of SRC failed"; exit 1; }
meta=$(printf '%s\n' "$sel" | awk -F'\t' '$1=="META"')
echo "$(ts) src: files=$(echo "$meta" | cut -f2) dirs=$(echo "$meta" | cut -f3) symlinks=$(echo "$meta" | cut -f4) file-bytes=$(echo "$meta" | cut -f5)"

bad=0; n=0; hashed=0
while IFS=$'\t' read -r kind rel; do
  [ "$kind" = TOP ] || [ "$kind" = SMP ] || continue
  n=$((n+1))
  s1=$(stat -c %s "$SRC/$rel" 2>/dev/null); s2=$(stat -c %s "$DST/$rel" 2>/dev/null)
  h1=$(sha256sum "$SRC/$rel" 2>/dev/null | cut -d' ' -f1); h2=$(sha256sum "$DST/$rel" 2>/dev/null | cut -d' ' -f1)
  hashed=$((hashed + ${s1:-0}))
  if [ -z "$h1" ] || [ "$h1" != "$h2" ] || [ "$s1" != "$s2" ]; then echo "   MISMATCH [$kind] $rel (size $s1/$s2)"; bad=$((bad+1)); fi
done <<< "$sel"
echo "$(ts) hashed $n files ($hashed bytes) on both sides, mismatches=$bad"

# Symlink manifest and directory count: one walk per side.
l1=$(find "$SRC" -type l -printf '%P\t%l\n' | LC_ALL=C sort | sha256sum | cut -d' ' -f1) || { echo "EXTRA_FAIL: symlink walk of SRC failed"; exit 1; }
w2=$(find "$DST" \( -type l -o -type d \) -printf '%y\t%P\t%l\n') || { echo "EXTRA_FAIL: walk of DST failed"; exit 1; }
l2=$(printf '%s\n' "$w2" | awk -F'\t' '$1=="l"{print $2"\t"$3}' | LC_ALL=C sort | sha256sum | cut -d' ' -f1)
d2=$(printf '%s\n' "$w2" | awk -F'\t' '$1=="d"{n++} END{print n+0}')
d1=$(echo "$meta" | cut -f3)
echo "$(ts) symlink manifest sha: src=${l1:0:16} dst=${l2:0:16}; dirs: src=$d1 dst=$d2"
[ "$l1" = "$l2" ] || { echo "   MISMATCH symlink manifest"; bad=$((bad+1)); }
[ "$d1" = "$d2" ] || { echo "   MISMATCH directory count"; bad=$((bad+1)); }

if [ $bad -eq 0 ] && [ $n -gt 0 ]; then echo "$(ts) EXTRA_PASS"; else echo "$(ts) EXTRA_FAIL (mismatches=$bad, hashed=$n)"; exit 1; fi
