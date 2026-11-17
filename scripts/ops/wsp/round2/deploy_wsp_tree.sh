#!/bin/bash
# deploy_wsp_tree.sh HOST SHA7 [PREFIX] — deploy one committed tree as <runs>/<PREFIX>_<SHA7> on hpg or loni:
# git archive -> scp -> extract -> GIT_COMMIT (= SHA7) -> logs/ (+ .venv link on hpg) -> sha256 of every tracked file compared.
set -u
host=${1:?hpg|loni}; t=${2:?sha7}; pre=${3:-wsp}
S=${SCRATCH:-/tmp/claude-1010/-home-xueqi-hq-projects-tc-alignment/035c9823-fbb5-436e-b37c-069b2060bb62/scratchpad/r2}
G=/home/xueqi/hq/projects/ICML27_agent_distillation
case "$host" in
  hpg) R=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_runs; extra="ln -sfn /blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_agent_distillation/.venv $R/${pre}_$t/.venv";;
  loni) R=/work/xueqic/hq/ICML27_runs; extra=true;;
  *) echo "bad host"; exit 2;;
esac
[ "${#t}" = 7 ] || { echo "need 7-char sha"; exit 2; }
D=$R/${pre}_$t
ssh -o BatchMode=yes "$host" "test ! -e $D" < /dev/null 2>/dev/null || { echo "$D exists on $host or ssh failed"; exit 1; }
[ -s $S/tree_$t.tar ] || git -C $G archive -o $S/tree_$t.tar $t || exit 1
rm -rf $S/ref_$t && mkdir -p $S/ref_$t && tar -xf $S/tree_$t.tar -C $S/ref_$t
(cd $S/ref_$t && find . -type f | sed 's#^\./##' | sort | xargs -d '\n' sha256sum) > $S/ref_$t.sha
scp -q -o BatchMode=yes $S/tree_$t.tar "$host:$R/tree_$t.tar" 2>/dev/null || { echo "$t scp failed"; exit 1; }
ssh -o BatchMode=yes "$host" "mkdir $D && tar -xf $R/tree_$t.tar -C $D && echo $t > $D/GIT_COMMIT && mkdir -p $D/logs && $extra && rm $R/tree_$t.tar" < /dev/null 2>/dev/null || { echo "$t extract failed"; exit 1; }
ssh -o BatchMode=yes "$host" "cd $D && xargs -d '\n' sha256sum" < <(cut -c67- $S/ref_$t.sha) 2>/dev/null | grep -E '^[0-9a-f]{64}  ' > $S/${host}_$t.sha
if diff -q <(sort -k2 $S/ref_$t.sha) <(sort -k2 $S/${host}_$t.sha) >/dev/null; then echo "$D $host OK $(wc -l < $S/ref_$t.sha) tracked files sha-identical"; else echo "$t $host MISMATCH"; exit 1; fi
