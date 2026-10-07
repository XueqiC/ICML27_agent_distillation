#!/bin/bash
# 2026-10-07: shrink CPU/memory of our pending paper-1 jobs to measured peaks + margin (dept colleague request).
# Measured (sacct MaxRSS of .batch, TotalCPU/Elapsed): 1-GPU explore/MC 7-15G, ~2.6 cores; eval 8-22G, ~2 cores;
# 2-GPU FSDP sft/DPO 11-54G, ~2 cores; 4-GPU sft 25G; LEAP GPU 19-29G, ~1 core; LEAP CPU-only correct 0.3G.
set -u
upd() { local j=$1 c=$2 m=$3 n; n=$(squeue -h -j $j -o "%j %t" 2>/dev/null)
  case "$n" in eto-tp6-*" PD"|ipr-r4-*" PD"|leap-k*" PD") scontrol update JobId=$j CpusPerTask=$c MinMemoryNode=$((${m%G}*1024)) && echo "$j $n -> ${c}cpu ${m}";;
    *) echo "skip $j '$n'";; esac; }
for j in 45058903 45058905 45082399 45082401 45082403 45082405 45082407 45082409; do upd $j 6 32G; done   # 1-GPU explore/MC/eval
upd 45058904 8 96G                                                                                    # 2-GPU DPO
for j in 45082398 45082400 45082402 45082404 45082406 45082408; do upd $j 8 160G; done                 # 4-GPU DPO
for j in 45082131 45082136; do upd $j 2 8G; done                                                     # LEAP CPU-only correct
for j in 45082132 45082133 45082137 45082138; do upd $j 4 48G; done                                    # LEAP 1-GPU

squeue -u xc25.fsu -h -o "%i|%j|%t|%C|%m" | grep -v wm- | sort
