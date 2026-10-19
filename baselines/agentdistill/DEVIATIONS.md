# AgentDistill implementation choices (registered before experiments)

Authority: `docs/baselines/NEW_METHOD_SPECS_1019.md`, AgentDistill section,
SHA256 `d6cc35f3222ee234cc9c970fe2392174365bf48e23ed51ba7e6c1ed91cee4f76`.
Method: [Qiu et al., arXiv:2506.14728v1](https://arxiv.org/html/2506.14728v1),
§3.2–3.5. Reference checkout: `third_party/AgentDistillation`,
`jtk16/AgentDistillation@83da61021941ae717f86ddf61e8b8e1d45ae0206`.

## Source availability and prompts

The supplied checkout has a CartPole mentor with local neural models and a
`LLMPromptManager` in `llm_mentor.py`, not the paper's Manager Agent/MCP module.
Its prompt separates system instructions, current state, and context. That
organization informs this port's manager prompt; CartPole physics, numerical
state descriptions, confidence heads, local checkpoint loading, and activation
distillation cannot be used in the black-box H0 protocol. No reference modules
are imported or copied wholesale. The inspected file's SHA256 is
`e9bc2aedbbfb7bc7f949fffe50b9c6571b37b2e6253a5c45e988cdc6b0e7d575`.

The paper specifies transformations and code-level clustering but does not
publish literal prompt templates. `prompts.py` therefore reconstructs the
manager, creation, abstraction, clustering, and consolidation prompts from those
sections; these are not claimed to be official verbatim prompts. All prompt code
and this register are bound by the shared provenance code hashes.

## User-specified environment and models

Both teacher roles use exactly `gpt-6-luna`; the frozen student is
`google/gemma-4-12B-it@707f0a3b`, OFF, 1,024 output tokens, greedy, seed 0.
There is no student training, optimizer, LoRA, merged model, or new checkpoint.
`models.json` publishes the original snapshot plus the box and their hashes.
Results explicitly say **different interface**.

Only the verified collection passed with `--verified` supplies task identities
and parent seed IDs. Shared `registered_support` and verified-evidence checks
admit registered nested K views and full-TRAIN collections, without a magic
full-TRAIN K. A box is independently built for each cell; no larger-cell tools
are shared into smaller cells. No demonstration text is added to manager
prompts. K remains trusted executed seed trajectories, not the number of runs.

The paper's external search, image captioning, external Python dependencies,
and networked FastMCP server are replaced by the user-required text H0 handle
and local callable interface. Creation planning and script generation share
one creator call, using the H0 API documentation as the resource description;
the sandbox validates and executes the result on the current real environment.
No open-source search call, external resource retrieval, or simulated transition
is performed. This is the registered ALFWorld adaptation.

## Execution and termination

Extended ReAct retains the shared H0 prompt and primitive action parser, adding
`Action: mcp {"name":...,"arguments":...}` and, for the manager only,
`Action: create_mcp {"request":...}`. The entire box's descriptions and parameter
names are visible every model turn, without retrieval or ranking. Tool code is
executed by the runtime, not inserted as arbitrary prompt metadata. All
parameters are strings, with up to three besides the environment handle.

An allowlisted Python AST interpreter implements local assignment, bounded
strings/lists/dicts, indexing, conditionals, loops, comparisons and interpolation.
Generated code is never passed to Python `exec`, `eval`, imports, host callables,
or a shell. `env.observe()` returns only whitelisted current visible fields;
`env.step()` yields a currently admissible H0 command and returns whitelisted
observations. Tool returns must be actual step observations, individually or
as a list. No files, network, system access, arbitrary attributes, exceptions,
recursion, comprehensions, arithmetic, annotations or decorators are supported.
Limits: 32,768 source characters, 3,000 AST nodes, 10,000 interpreter operations,
131,072 serialized characters per value/local store, 4,096 characters per
parameter/description, 64 characters per identifier. These are implementation
sandbox limits, not paper hyperparameters.

Coroutine yields are flattened into individual `sed.runner.run_episode` turns;
every tool action receives its own real step, event, observation and audit blob.
Primitive actions retain H0 parsing and inadmissible-action behavior. Tool
inadmissibility is rejected before execution as required by the restricted
handle. No tool action is executed after termination or the 40-turn runner cap.
Manager/student model calls separately stop at 40, including malformed replies
and tool requests that consume no environment steps. Exhausted model calls emit
parser rejections until the shared runner terminates; there are no further model
or environment calls. This avoids unbounded zero-action tool loops.

Raw extended replies and tool returns are logged separately from flattened H0
action events. Model token totals count actual returned model calls, including
malformed replies, rather than synthetic coroutine continuations. The teacher
ledger is authoritative for both teacher roles' costs. A pending tool is resumed
with its final actual observation on episode completion, but any newly requested
action is stopped. Only completed, nonempty executions from an environment-
verified winning run enter the pool. A procedure stopped midway through at
termination is retained as rejected execution evidence, not admitted to the pool.

## Budgets, retry and construction

Stop at the first verified successful run, otherwise reserve at most three runs
per task. Each run makes at most 40 manager calls and at most one creator call
per manager call. Interrupted reservations consume a run and are never reset or
treated as restored state. Complete run receipts are reused; unknown interrupted
runs cannot supply tools. There are no automatic transport retries or fallback
teacher/student substitutions.

Every call goes through one `teachers.capped.CappedTeacher` counter and usage
ledger per cell. With N support instances, the conservative hard transport cap
is `3 * 40 * (2 + 1 + 1) * N + 1 = 480*N + 1`: manager/creator calls,
one abstraction and at most one consolidation per possible tool, and one global
clustering call. Logical role counters additionally bound each category, and
stable request slots prevent extra sampling on restart. All teacher requests
use a 32,768-output-token ceiling to accommodate code and full-cell clustering;
other sampling settings are the configured teacher settings. No logits or
hidden reasoning are requested. Provider reasoning-token counts are accounting
metadata only.

Abstraction uses one teacher call per distinct executed tool per successful run.
Exact repeated uses within that run are represented once. Invalid abstractions
are retained as rejected records and excluded from clustering. Clustering is
one global code-level call; membership must be a complete disjoint partition.
Consolidation makes one call per cluster; invalid code, duplicate final names,
or invalid clustering block publication. There is no heuristic clustering,
fixed action macro fallback, empty-result replacement, or automatic redraw.
An empty successful pool produces an explicitly empty box with zero construction
calls. Large cells that exceed the provider context fail with retained evidence;
there is no undocumented truncation, retrieval, or hierarchical clustering.

Generated tools are dynamically executed during collection. Abstraction and
consolidation rewrites receive static sandbox validation, without additional
environment trajectories beyond the permitted runs. Their changed behavior is
explicitly **unknown**, not labeled verified success. The original successful
run lineage remains available for every final tool.

## Stages and evaluation

Each CLI invocation performs just its selected stage. Missing prerequisites
fail before creating a teacher client or starting a server. Shared stage
transactions preserve failed attempts; stage artifacts bind external run and
teacher request evidence as well. Publish freezes the final teacher ledger.
Teacher and publication jobs reject GPU allocations even if CUDA is hidden.

The original snapshot needs a base-model HF/vLLM gate, because the existing full
and adapter gates require trained checkpoint metadata. `gate.py` reuses their
eight-TRAIN-turn selection, native OFF encoding, scoring and full-checkpoint
error tolerances. Prompts contain the entire final tool catalog. HF runs first;
vLLM then starts through the existing harness launcher. The gate binds weights,
box, prompts, server provenance, tokenizer template and verified collection.
No extra special tokens or weight updates are introduced.

Separate evaluate mode consumes `models.json` and the frozen dev140 task list,
using the shared suite with a method-specific worker. It never invokes
collection, construction, training, or writes to verified data. There is no
dev-based tool selection. All reported outcomes come from environment execution.
CPU tests and infrastructure checks establish no empirical finding or faithful
numerical reproduction of the paper's original benchmarks.
