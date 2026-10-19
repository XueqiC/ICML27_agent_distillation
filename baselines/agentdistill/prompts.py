"""Paper §3.2–3.5 prompt reconstructions; see DEVIATIONS.md for attribution.

The reference's LLMPromptManager separates system, CURRENT STATE, and context.
We reuse that organization, replacing CartPole physics/privileged inputs with H0.
Neither the paper nor 83da610 supplies literal MCP construction templates.
"""
import json

from sed.react import PROMPT_TEMPLATE, build_messages

LANGUAGE = """Write exactly one self-contained Python function with signature
def name(env, param1, ...): and at most three string parameters besides env.
The only effects allowed are env.step(command), returning a visible observation
dictionary, and env.observe(), reading the current visible observation dictionary.
Each step command must belong to the CURRENT admissible_commands. Every command
costs one of the episode's 40 environment actions. Return one actual step observation
or a list of actual step observations. Never invent observations or success.
Supported Python: local assignment, strings/integers/booleans, lists/dicts,
indexing, string/list +, simple f-strings, if, for, while, break, continue,
comparisons == != in not in, and/or/not, local list.append. No other calls,
imports, decorators, builtins, exceptions, attributes, files, network or system
access. No recursion, comprehensions, arithmetic or type annotations.
Tool JSON: {"name":"name","description":"What it does and parameter meanings",
"parameters":["param1"],"code":"def name(env, param1):\\n    return env.step(param1)"}.
Use only the visible task/history. Do not encode instance IDs or hidden fields.
"""

EXTENDED = """
AgentDistill extended ReAct: optionally give a brief visible Thought, then one
Action line. A primitive H0 Action remains available. To call a tool use
Action: mcp {"name":"tool_name","arguments":{"parameter":"value"}}
All tools below are available; choose the tool and its arguments yourself.
Tool observations are actual environment observations. The runtime stops at
environment termination or 40 actions. No online learning or new tools at test time.
"""
MANAGER = """You are the AgentDistill Manager Agent. Decompose the visible task
into subtasks, delegate reusable action procedures to the MCP Creation Module,
use the returned observations, and coordinate completion. Use existing tools
when appropriate. To delegate creation AND execution at the current state use:
Action: create_mcp {"request":"A concise subtask and desired action procedure"}
The creator will supply a tool and arguments. You may also use primitive H0 actions.
"""
CONSTRUCTION = {
    "abstract": """Abstract this executable MCP from a successful trajectory into
a concise task-agnostic reusable Python tool. Preserve its core action logic;
remove example-specific phrases and make up to THREE critical values configurable.
Return only one tool JSON object. Do not solve new tasks or use hidden facts.""",
    "cluster": """Cluster ALL supplied abstracted Python tools by function using
code-level application semantics, not task identity or lexical similarity alone.
Return only {"clusters":[{"name":"functional_label","members":["tool_id",...]}]}.
Every input tool ID must occur exactly once; clusters must be nonempty with
unique names. Do not generate, drop, rank or retrieve tools.""",
    "consolidate": """Consolidate ALL implementations in this functional cluster
into ONE general-purpose Python tool. Unify parameters (at most three), validate
action preconditions using visible admissible_commands, document parameter
meanings, and preserve the cluster's functional behaviors. Return one tool JSON
object, with no imports or dependencies on other tools.""",
}
CREATE = """You are the AgentDistill MCP Creation Module. Plan a procedure for the
manager's subtask, then implement it using the supplied H0 API documentation.
Return only {"tool": TOOL_JSON, "arguments": {"parameter":"value"}}.
The runtime will validate and execute the script in the actual environment.
""" + LANGUAGE


def catalog(tools):
    # Explicit whitelist: lineage, hashes, audit and success labels never render.
    return [{key: tool[key] for key in ("name", "description", "parameters")} for tool in tools]


def policy_template(tools, *, manager=False):
    return PROMPT_TEMPLATE + EXTENDED + (MANAGER if manager else "") + "\nMCP box:\n" + json.dumps(catalog(tools))


def messages(history, task, visible, tools):
    return build_messages(history, task, visible, prompt_template=policy_template(tools))
