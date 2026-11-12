#!/bin/bash
# tc_census.sh — READ-ONLY census of one or more trees on hpg Lustre. One MDS-only walk per tree
# (lfs find --lazy: sizes come from the MDS lazy-size attribute, so there is no OST glimpse per file;
# a size can be stale for a file written very recently). Per tree it prints regular files and bytes by
# depth-2 prefix (P lines), a size histogram (H lines) and the total (T line). Writes nothing.
# usage (ON hpg): bash tc_census.sh DIR [DIR...]      last line: CENSUS_DONE
set -u -o pipefail
for DIR in "$@"; do
  DIR="${DIR%/}"
  [ -d "$DIR" ] || { echo "SKIP not a dir: $DIR"; continue; }
  echo "$(date '+%T') BEGIN $DIR host=$(hostname)"
  lfs find "$DIR" -type f --lazy -printf '%s\t%p\n' | awk -F'\t' -v root="$DIR/" '
    { s=$1; p=substr($2, length(root)+1); n=split(p, a, "/");
      k=(n>=3 ? a[1]"/"a[2]"/" : (n==2 ? a[1]"/" : "(top-level files)"));
      c[k]++; b[k]+=s; tc++; tb+=s;
      h=(s<4096?1:(s<65536?2:(s<1048576?3:(s<67108864?4:(s<1073741824?5:6))))); hc[h]++; hb[h]+=s }
    END {
      for (k in c) printf "P\t%d\t%d\t%s\n", c[k], b[k], k;
      split("<4K,4K-64K,64K-1M,1M-64M,64M-1G,>=1G", L, ",");
      for (i=1;i<=6;i++) printf "H\t%d\t%d\t%s\n", hc[i]+0, hb[i]+0, L[i];
      printf "T\t%d\t%d\tTOTAL\n", tc+0, tb+0 }' | sort -t$'\t' -k1,1 -k2,2nr; rc=("${PIPESTATUS[@]}")
  echo "$(date '+%T') END $DIR rc=${rc[*]}"
done
echo "$(date '+%T') CENSUS_DONE"
