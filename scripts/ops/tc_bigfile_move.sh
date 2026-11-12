#!/bin/bash
# tc_bigfile_move.sh — move a LIST of large regular files from one of our /blue trees to its /orange mirror,
# keeping relative paths; with --delete-source remove them from /blue, but only after EVERY file has been
# verified. Purpose: free bytes on /blue when a tree is millions of small files plus a few thousand weights
# (a per-file copy of the whole tree runs at ~16 files/s on /orange).
# Runs ON hpg, detached. Holds the global move lock for the whole run. One rsync stream; hashing uses at
# most 4 readers and never touches both filesystems at once.
#
# usage: bash tc_bigfile_move.sh NAME ROOT DSTROOT LIST [--delete-source|--verify-only]
#   NAME     label; state dir = /orange/yd24f.fsu/xc25.fsu/tc-cold/_ops/bigmove/NAME/
#   LIST     text file, one path per line, relative to ROOT; regular files only, no leading '/', no '..'
# Checks (all files, not a sample): same number of files, same size per file, same sha256 per file, and the
# source size+mtime unchanged between the first stat and the end of hashing.
# Verdict lines: "HH:MM:SS PASS: …", "HH:MM:SS DELETED …", or "FAIL: …" / "REFUSE: …" / "BUSY: …".
# Restore later: rsync -a --files-from=<manifest paths> DSTROOT/ ROOT/   (manifest is written to DSTROOT).
set -u -o pipefail
NAME="${1:?name}"; ROOT="${2:?root}"; DSTROOT="${3:?dstroot}"; LIST="${4:?list}"; MODE="${5:-}"
ROOT="${ROOT%/}"; DSTROOT="${DSTROOT%/}"
case "$NAME" in *[!A-Za-z0-9._-]*|"") echo "REFUSE: bad name"; exit 2;; esac
case "$ROOT" in
  /blue/fsu-compsci-dept/xc25.fsu/hq/*|/blue/yd24f.fsu/xc25.fsu/hq/*) ;;
  /orange/yd24f.fsu/xc25.fsu/tc-cold/_ops/bigmove_test/*) echo "TEST MODE: root is under bigmove_test";;
  *) echo "REFUSE: root must be under one of our two /blue trees: $ROOT"; exit 2;;
esac
case "$DSTROOT" in /orange/yd24f.fsu/xc25.fsu/*) ;; *) echo "REFUSE: dstroot must be under /orange/yd24f.fsu/xc25.fsu/: $DSTROOT"; exit 2;; esac
case "$ROOT$DSTROOT" in *..*) echo "REFUSE: no '..' in paths"; exit 2;; esac
case "$MODE" in ""|--delete-source|--verify-only) ;; *) echo "REFUSE: fifth arg must be --delete-source, --verify-only, or absent"; exit 2;; esac
[ -d "$ROOT" ] || { echo "REFUSE: root not a directory: $ROOT"; exit 2; }
[ -f "$LIST" ] && [ -s "$LIST" ] || { echo "REFUSE: list missing or empty: $LIST"; exit 2; }
WORK=/orange/yd24f.fsu/xc25.fsu/tc-cold/_ops/bigmove/$NAME
LOCK=/orange/yd24f.fsu/xc25.fsu/.orange_move.lock
exec 9>"$LOCK"; flock -n 9 || { echo "BUSY: another move is running (lock $LOCK)"; exit 3; }
ts(){ date '+%H:%M:%S'; }
mkdir -p "$WORK" "$DSTROOT" || { echo "FAIL: cannot create $WORK or $DSTROOT"; exit 1; }
RUN="$WORK/run.$(date -u +%Y%m%dT%H%M%SZ).$$"; mkdir "$RUN" || { echo "FAIL: cannot create $RUN"; exit 1; }
L="$RUN/list"; LC_ALL=C sort -u "$LIST" > "$L" || { echo "FAIL: cannot copy the list"; exit 1; }
N=$(wc -l < "$L")
echo "$(ts) NAME=$NAME MODE=${MODE:-copy} host=$(hostname)"; echo "$(ts) ROOT=$ROOT"; echo "$(ts) DSTROOT=$DSTROOT"; echo "$(ts) files in list=$N  run dir=$RUN"

echo "$(ts) [0/5] validate the list (regular files, one link, inside root)"
if grep -n -E '^/|(^|/)\.\.(/|$)|^$' "$L" | head -n 5 | grep -q .; then echo "REFUSE: list has an absolute, empty or '..' path"; exit 2; fi
( cd "$ROOT" && xargs -d '\n' -a "$L" stat --printf='%F\t%h\t%s\t%Y\t%n\n' ) > "$RUN/src.stat0" 2> "$RUN/src.stat0.err"; rc=$?
[ $rc -eq 0 ] || { echo "REFUSE: stat failed for some listed path (rc=$rc): $(head -n 3 "$RUN/src.stat0.err")"; exit 2; }
[ "$(wc -l < "$RUN/src.stat0")" -eq "$N" ] || { echo "REFUSE: stat returned a different number of lines than the list"; exit 2; }
if awk -F'\t' '$1!="regular file" || $2!=1' "$RUN/src.stat0" | head -n 5 | grep -q .; then
  echo "REFUSE: not a regular single-link file: $(awk -F'\t' '$1!="regular file" || $2!=1 {print $5}' "$RUN/src.stat0" | head -n 3 | tr '\n' ' ')"; exit 2; fi
RROOT=$(realpath "$ROOT") || { echo "REFUSE: realpath of root failed"; exit 2; }
( cd "$ROOT" && xargs -d '\n' -a "$L" realpath ) > "$RUN/src.real" 2>/dev/null; rc=$?
[ $rc -eq 0 ] && sed "s|^|$RROOT/|" "$L" | cmp -s - "$RUN/src.real" || { echo "REFUSE: a listed path goes through a symlink (realpath differs from root/path)"; exit 2; }
cut -f3- "$RUN/src.stat0" | LC_ALL=C sort -t$'\t' -k3 > "$RUN/src.stat"     # size, mtime, path
BYTES=$(awk -F'\t' '{s+=$1} END{printf "%d", s}' "$RUN/src.stat")
echo "   $N regular files, $BYTES bytes"

if [ "$MODE" = "--verify-only" ]; then
  echo "$(ts) [1/5] rsync skipped (--verify-only)"
else
  echo "$(ts) [1/5] rsync --files-from (one stream; resumable, re-run is safe)"
  rsync -a --partial --inplace --files-from="$L" --info=stats1 "$ROOT"/ "$DSTROOT"/; rc=$?
  [ $rc -eq 0 ] || { echo "FAIL: rsync rc=$rc — nothing deleted"; exit 1; }
fi

echo "$(ts) [2/5] destination: every file present with the same size"
( cd "$DSTROOT" && xargs -d '\n' -a "$L" stat --printf='%F\t%s\t%n\n' ) > "$RUN/dst.stat0" 2> "$RUN/dst.stat0.err"; rc=$?
[ $rc -eq 0 ] || { echo "FAIL: a listed file is missing on the destination (rc=$rc): $(head -n 2 "$RUN/dst.stat0.err") — nothing deleted"; exit 1; }
if awk -F'\t' '$1!="regular file"' "$RUN/dst.stat0" | grep -q .; then echo "FAIL: a destination entry is not a regular file — nothing deleted"; exit 1; fi
cut -f2- "$RUN/dst.stat0" | LC_ALL=C sort -t$'\t' -k2 > "$RUN/dst.size"
cut -f1,3 "$RUN/src.stat" > "$RUN/src.size"
cmp -s "$RUN/src.size" "$RUN/dst.size" || { echo "FAIL: sizes differ: $(diff "$RUN/src.size" "$RUN/dst.size" | head -n 4 | tr '\n' ' ') — nothing deleted"; exit 1; }
echo "   $(wc -l < "$RUN/dst.size") files, sizes identical"

echo "$(ts) [3/5] sha256 of every source file (4 readers)"
( cd "$ROOT" && xargs -d '\n' -a "$L" -P 4 -n 1 sha256sum ) > "$RUN/src.sha0" 2> "$RUN/src.sha0.err"; rc=$?
[ $rc -eq 0 ] || { echo "FAIL: hashing the source failed (rc=$rc): $(head -n 2 "$RUN/src.sha0.err") — nothing deleted"; exit 1; }
LC_ALL=C sort -k2 "$RUN/src.sha0" > "$RUN/src.sha256"
echo "$(ts) [4/5] sha256 of every destination file (4 readers)"
( cd "$DSTROOT" && xargs -d '\n' -a "$L" -P 4 -n 1 sha256sum ) > "$RUN/dst.sha0" 2> "$RUN/dst.sha0.err"; rc=$?
[ $rc -eq 0 ] || { echo "FAIL: hashing the destination failed (rc=$rc): $(head -n 2 "$RUN/dst.sha0.err") — nothing deleted"; exit 1; }
LC_ALL=C sort -k2 "$RUN/dst.sha0" > "$RUN/dst.sha256"
ns=$(wc -l < "$RUN/src.sha256"); nd=$(wc -l < "$RUN/dst.sha256")
[ "$ns" -eq "$N" ] && [ "$nd" -eq "$N" ] || { echo "FAIL: hashed $ns source and $nd destination files, expected $N — nothing deleted"; exit 1; }
cmp -s "$RUN/src.sha256" "$RUN/dst.sha256" || { echo "FAIL: sha256 differs: $(diff "$RUN/src.sha256" "$RUN/dst.sha256" | head -n 4 | cut -c1-200 | tr '\n' ' ') — nothing deleted"; exit 1; }

echo "$(ts) [5/5] source unchanged since the first stat (size + mtime)"
( cd "$ROOT" && xargs -d '\n' -a "$L" stat --printf='%s\t%Y\t%n\n' ) 2> "$RUN/src.stat1.err" | LC_ALL=C sort -t$'\t' -k3 > "$RUN/src.stat1"; rc=("${PIPESTATUS[@]}")
[ "${rc[0]}" -eq 0 ] || { echo "FAIL: re-stat of the source failed — nothing deleted"; exit 1; }
cmp -s "$RUN/src.stat" "$RUN/src.stat1" || { echo "FAIL: a source file changed while it was being copied or hashed — nothing deleted"; exit 1; }

MAN="$DSTROOT/_tc_moved_big_files.$NAME.tsv"
{ echo "# files moved from $ROOT to $DSTROOT by tc_bigfile_move.sh ($NAME). columns: sha256, bytes, relative path"
  echo "# restore: cut -f3 THIS_FILE | grep -v '^#' > /some/list; rsync -a --files-from=/some/list $DSTROOT/ $ROOT/"
  LC_ALL=C join -t$'\t' -1 2 -2 2 <(sed 's/  /\t/' "$RUN/src.sha256" | LC_ALL=C sort -t$'\t' -k2) <(LC_ALL=C sort -t$'\t' -k2 "$RUN/src.size") | awk -F'\t' '{print $2 "\t" $3 "\t" $1}'
} > "$MAN.tmp" && mv "$MAN.tmp" "$MAN" || { echo "FAIL: cannot write the manifest $MAN — nothing deleted"; exit 1; }
[ "$(grep -vc '^#' "$MAN")" -eq "$N" ] || { echo "FAIL: manifest has the wrong number of rows — nothing deleted"; exit 1; }
echo "$(ts) PASS: $N files, $BYTES bytes, every sha256 equal: $ROOT -> $DSTROOT  (manifest $MAN)"

if [ "$MODE" = "--delete-source" ]; then
  echo "$(ts) deleting $N verified source files under $ROOT"
  del=0; fail=0
  while IFS= read -r rel; do
    if rm -f -- "$ROOT/$rel"; then del=$((del+1)); else fail=$((fail+1)); echo "   rm failed: $rel"; fi
  done < "$L"
  left=$( ( cd "$ROOT" && xargs -d '\n' -a "$L" ls -d 2>/dev/null ) | wc -l )
  [ $fail -eq 0 ] && [ "$left" -eq 0 ] || { echo "FAIL: rm failed for $fail files, $left still present; check by hand"; exit 1; }
  cp "$MAN" "$ROOT/_MOVED_TO_ORANGE.$NAME.tsv" 2>/dev/null || echo "   note: could not leave a pointer file in $ROOT (quota?); the manifest is on /orange"
  echo "$(ts) DELETED $del files, $BYTES bytes from $ROOT"
  echo "$(ts) remember: record $DSTROOT and $MAN in PROJECT_STATE.md"
else
  echo "$(ts) source kept. Re-run with --delete-source to remove the listed files (re-verifies first)."
fi
