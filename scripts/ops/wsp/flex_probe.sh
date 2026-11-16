#!/bin/bash
# flex_probe.sh (hpg, ONE RTX PRO 6000 GPU) — which Flex attention tiles compile on this card (see flex_probe.py).
set -u
R=/blue/fsu-compsci-dept/xc25.fsu/hq/ICML27_wsp; Q=$R/queue; PY=/blue/fsu-compsci-dept/xc25.fsu/hq/tc-alignment/.venv/bin/python
set -a; . "$Q/jobs/prep.sc-k20.env"; set +a
source "$PROJECT_ROOT/configs/hpg.env"
cd "$PROJECT_ROOT" || exit 2
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" "$Q/derive/flex_probe.py" "$R/cells/sc-k20"
