# WKM port registration — 2026-10-19

Official pin: `f1fe28fc189c59801a98cabda01188bc9b64eae6`
([source](https://github.com/zjunlp/WKM/tree/f1fe28fc189c59801a98cabda01188bc9b64eae6)).
Paper: [Agent Planning with World Knowledge Model](https://arxiv.org/abs/2405.14205).
The frozen `docs/baselines/NEW_METHOD_SPECS_1019.md` WKM section governs this port.
This is an engineering implementation, with no experimental reproduction or
learning finding claimed. No GPU run, teacher call or cluster job was performed.
Upstream files and hash-pinned protocol documents remain unchanged.

| Official file:line / paper | Replacement/change | Reason class |
| --- | --- | --- |
| `README.md`, Model Training; paper §3.3 | Both adapters use the configured Gemma backbone, r8/alpha16, q/v text projections only, lr1e-4, cosine, 3 epochs, batch32→min(32,max(1,floor(N/8))). Experienced and final agent and world LoRAs independently initialize from the base. Zero dropout, zero warmup, wd0, AdamW and clip1; seed0. | fixed model / X5; unspecified optimizer details registered |
| `src/world_knowledge_build.py:20-89`; paper §3.1 | Missing sampler uses shared `baselines.common.explore`: one greedy H0 rollout per verified support task, actual success and replay checked. Only known failures are rejected; unknown/incomplete execution halts the stage. All outcomes and replay transitions remain in artifacts. | requested missing sampler; evidence integrity |
| `src/prompts/knowledge_template.py`; `src/world_knowledge_build.py` | Selectively port the ALFWorld synthesis prompt constants, with attribution and Apache-2.0 license. Task and state knowledge are generated exclusively by the experienced **student adapter**, temperature .5, seed0, top_p1/top_k−1, 512/128 tokens. No teacher/embedding API calls. The state prompt's unexpanded `{task}` is filled with `alfworld`; no newline stop is imposed, and incomplete/empty outputs halt with raw evidence retained. | unchanged self-synthesis; deterministic sampling / released placeholder fix |
| `src/world_knowledge_build.py:25-40` | Use one fixed upstream pick-and-place workflow example for every task; do not select prompts from goal-revealing game filenames/task types. Released ALFWorld state example remains a fixed external prompt resource, disclosed here; no released trajectory dataset is loaded. Preserve whole visible observations rather than the upstream 1000-character cutoff. | hidden-field isolation / no truncation |
| Missing successful-exploration handling | If the only exploration succeeds, its rejected entry is absent. The same synthesis prompt explicitly states there is no rejected trajectory, preserving the task's expert and state examples. No successful run is falsely labelled rejected, no retry or additional task is introduced. | unspecified edge case |
| Missing merge; `src/train_data_process.py:43-112` | Merge task and per-step state knowledge by checked task/parent/execution hashes. Agent format appends task knowledge only to the first observation's text **inside H0 JSON**. H0 system/parser/admissible commands remain intact. No state knowledge is inserted into agent observations. | requested merge; H0 mapping |
| `src/train_data_process.py:43-75` | World-model user turns contain the previous agent reply plus current visible observation; world assistant history contains its task/state knowledge, as in `process_wm`. The first target is task knowledge. Add a separate initial-state request after the task knowledge to cover the spec's state knowledge at **every** expert step (the repo replaces the first state target with task knowledge). This extra initial state is not prepended to later world histories. | explicit spec / initial-state edge case |
| `src/state_base_build.py:16-31` | Keep `(pre_action,state,action)` triplets, initial previous action `look`, and all verified expert steps. Preserve complete commands rather than discarding arguments. Build the KB immediately after merge, before world/agent training, so each newly published checkpoint already has an evaluation bundle; content is independent of training order. | H0 action mapping / staging |
| `src/state_base_build.py:44`; `utils/documentsearch.py` | Replace paid ada-002/Chroma with local `sentence-transformers/all-MiniLM-L6-v2`, masked mean pooling, normalized vectors, cosine ranking and stable row-index ties. Require `--embedding-snapshot`, check the MiniLM architecture, bind **all snapshot bytes** and pooling configuration in provenance. No downloads or alternate checkpoints. Encoder input truncation at its native 256 tokens affects retrieval only, never supervised student text. | required non-Chinese local embedding choice |
| `utils/retrieve_action.py:87-155`; paper §3.4 | Top min(3000,KB size), then the released previous-action **verb** filter. Count exact next commands that are currently admissible and normalize. If none survive, explicitly fall back to the agent distribution. | same retrieval/filter ordering; H0 frequency mapping and empty evidence policy |
| `eto_multi_main_probs.py:56-79`; `utils/retrieve_action.py:21-58` | Generate the agent's visible thought under task knowledge, then score each admissible full `Thought/Action` reply plus native turn end with the existing vLLM student prompt-logprob scorer. Softmax summed log likelihoods (common thought contribution cancels); argmax `.4*p_agent+.6*p_know`. The repo mixes first-verb-token probabilities then generates arguments. The spec requires full-command admissible-action mixing; this adaptation is explicit. Scoring is of the local student, never the teacher. All scoring requests/results/usage retained. | user-required complete-action adaptation |
| Agent/world data serialization | Shared `train.dataset.encode_turn` Gemma OFF prefill, assistant targets/turn ends only; exact full histories, no training truncation; logical batches count original trajectories, world knowledge targets grouped by parent trajectory. Reuse `baselines.common.fsdp_train` LoRA path and `train.modeling.attach_lora`, with an opt-in q/v recipe; existing defaults unchanged. | X3/X6/X7 / trainer reuse |
| Evaluation scripts | `models.json`: experienced agent alone, experienced+new world at the world checkpoint, final agent+world. Two named LoRAs on one vLLM backbone. **Both** existing HF/vLLM gates must pass; every bundle uses greedy dev140, seed0 and batch invariance, with no selection by score. Shared suite gains opt-in `wkm` decoding. World checkpoints are evaluated in their intended paired role, not as action policies. | dual-model evaluation integration |
| All stages | Shared hash-bound stage transactions, retained failed attempts, durable student raw calls and exposures, frozen `--verified` source only. Accept any positive size that the shared verified-collection loader validates; no method-specific K≤100 restriction, no invented full-collection schema or evidence bypass. K remains executed trusted seeds, separate from instances/groups. | engineering / nested and forthcoming full-support collections |

Usage (local snapshots must already exist; run no downloads here):

```bash
python -m baselines.wkm.pipeline --snapshot "$MODEL_PATH" --verified "$VERIFIED" \
  --embedding-snapshot "$MINILM_SNAPSHOT" --output "$WKM_OUT" --through experienced
```

Stages: `experienced`, `explore`, `synthesize`, `merge`, `knowledge-base`, `world`,
`agent`. Each `--through` validates/resumes earlier stages. `merge` and
`knowledge-base` require completed GPU prerequisites. The SLURM wrapper has no
implicit GPU allocation: explicitly submit GPU stages with `--gpus=1` and CPU
stages with `--gpus=0` on the appropriate partition. Keep `NGPUS=1`, paths and
interpreters fixed across jobs. `WKM_THROUGH` selects the boundary;
`WKM_MODE=evaluate` evaluates all published bundles. Direct
`scripts/hpg_eval_adapters.slurm` also accepts `DECODING=wkm`, exactly two
`ADAPTERS`, `WKM_AGENT_NAME`, `WKM_KNOWLEDGE_NAME`, `WKM_KNOWLEDGE_BASE`, and
`INCLUDE_BASE=0`. The environment-worker interpreter needs local transformers/
PyTorch for CPU embedding inference; training uses `TRAIN_PYTHON`.

CPU tests establish protocol and implementation behavior only. Production
throughput, memory fit, live vLLM prompt scoring and numerical gates still require
an authorized GPU run.
