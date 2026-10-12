# Theory Markdown validation

Run from the repository root with Python 3.11 or later:

```sh
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theoretical_framework/tools/check_markdown.py
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python -m unittest discover -s theoretical_framework/tools -p 'test_check_markdown.py'
```

The default check and tests run offline, without credentials, network access or
OpenAI API calls. The scope is every Markdown file under `theoretical_framework/`,
`docs/direction/2026-10-11_v015_progress_ZH.md`, and every Markdown file directly
under `docs/direction/v015_extension/`. The revision and handoff directories are
excluded. The v0.16 statement and label policy (S1) covers the framework's
`README.md`, `README_ZH.md`, all Markdown files directly under `proofs/` and
`experiments/`, and `figures/README.md`. Historical direction documents, figure
specs, tool documentation and other pages receive rendering and link checks only.

Pass optional positional `PATH...` arguments to check exactly those files instead
of the default scope. Paths are relative to the repository root (regardless of
the current directory), or absolute paths inside the repository:

```sh
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theoretical_framework/tools/check_markdown.py theoretical_framework/README.md theoretical_framework/tools/README.md
```

The same selection applies to the offline check, `--online` and `--blob`, including
when both flags are combined. Selected files in S1 keep its statement and label
policy. Every other selected file receives
rendering and link checks only, including files outside the default scope.
Link targets are still checked even when they are not selected; this does not
add those targets to the validation or rendering scope.
The declaration index reads all S1 pages to resolve references and duplicates.
Selecting a new proof or experiment page also compares its declaration set with
its language counterpart, without validating that counterpart's rendering.

Paths are resolved through symlinks and repeated resolved paths are checked once.
Paths outside the repository (including symlink escapes), missing files,
directories and files without the `.md` extension are command-line errors,
detected before validation or network operations. With no paths, the default
scope is used. Diagnostics include per-rule hit counts before the final summary.

The v0.16 policy uses one statement counter per section, shared by Definition,
Assumption, Theorem, Proposition, Lemma, Corollary, Remark, Example and Algorithm:

- Declare statements in a bold paragraph at the start of a line:
  `**Theorem 6.3 (Title).**` or `**Theorem 6.3（标题）.**`. Both languages use
  English type names. Section 1 declarations belong in the framework READMEs;
  sections 2–9 belong in `proofs/0K_name.md` and `proofs/0K_name_ZH.md`.
- Each declaration needs its explicit HTML anchor (an `a` element with `id="sK-J"`) in the preceding
  three non-blank lines. Numbering starts at `K.1`, increases in source order,
  and has no gaps. Each number can be declared only once per language across
  the READMEs and proof pages, regardless of statement type.
- Mentions such as `Theorem 6.3`, `Theorems 6.3 and 6.5`,
  `Propositions 5.5–5.9`, `Corollaries 6.7–6.8` and `Definitions 1.3–1.4`
  must resolve to declared statements of the stated type. Both range endpoints
  must exist and belong to the same section; intermediate numbers are not
  checked as mentions. Wrapped prose references are supported. References resolve
  across the framework, including
  unselected pages and either language. The labels `Theorem A`, `Theorem B`
  and `Theorem C` are also valid.
- Declare checks in experiment pages with `## Check K.J · Title` and an explicit
  HTML anchor (`id="cK-J"`) within the preceding three non-blank lines. Their numbering
  also starts at 1 and has no gaps or duplicates. `Check K.J` mentions must
  resolve. Section numbers must match `experiments/0K_name.md` or its ZH version.
- Proof pairs must declare the same set of `(type, K.J)` pairs; experiment pairs
  must declare the same checks. Missing counterparts are reported. Links ending
  in `#sK-J` or `#cK-J` must resolve to the declaring page and its anchor; a stray
  anchor on another page does not suffice.
- Section references `§1` through `§9`, including `§2–§9`, are valid. Bare source
  labels such as `§10`, `§5.2`, `A5` and `eqs. (A11)`, old result labels such as
  `Theorem 7`, `Thm. C4`, `Prop. A5`, `Corollary 9`, `Lemma W0` and `定理 3`, and
  node labels `N01`–`N20` (including parts) and `W0`–`W3` are rejected. Node labels
  are checked inside inline code too; link labels count as prose, while link
  destinations and HTML attributes do not.
- Only the exact README sections
  `## Appendix · Correspondence with earlier labels and sources` and
  `## 附录 · 与早期编号及出处的对照` exempt earlier labels and statement references.
  The exemption includes subordinate headings and ends at the next level 1 or
  level 2 heading. It never exempts rendering or links. Old `Source concordance`
  headings and proof-page source lines do not create exemptions.
- README check codes such as `C5`, `V7`, `E12` and `T03` must be in code spans,
  including in the appendix. Experiment pages require both a code span and a
  suite prefix, for example `v014:C5`, `v015:E6`, `pkg:T12` or
  `v016:C4.amplitude`. Bare prose IDs and code spans without a prefix produce
  an `unprefixed check code` error. Prefixes use `v` followed by digits, or `pkg`.

Old `proofs/N??_*.md` and `experiments/N??_*.md` pages can coexist with new pages
during the rewrite. They remain in scope for rendering, links and label policy,
so retired-label hits are expected until removal. Only the new `0K_` page pairs
require the new bilingual structure. Historical direction documents are exempt
from all statement, source-label, node-label and check-code policy.
Old pages' local declarations do not enter the v0.16 index or its duplicate,
anchor, numbering and parity checks.

Display math uses `math` fences at column 0, with a blank line before and after.
The `math-fence-in-list` error rejects fenced math inside any bullet or numbered
list, including nested lists and indented continuations. It also rejects indented
math fences outside lists. Inline math uses the dollar/backtick form, such as
`` $`x`$ ``. When moving a display out of a list, convert sibling items into
paragraphs too, retaining their labels and bolding any explicit item numbers.

`--blob` is the authoritative pre-push rendering check:

```sh
python theoretical_framework/tools/check_markdown.py --blob --cache /tmp/theory-markdown-html
```

GitHub's `/markdown` API renders math fences in lists that GitHub's real file
view leaves as raw code when the same list contains inline math. Consequently,
passing `--online` alone does not establish that a pushed file will render.

`--blob` hashes the scoped working-tree files at their repository paths, builds
the directory trees with `git mktree`, creates a parentless commit with
`git commit-tree`, and force-pushes it to `refs/render-check/tmp`. This is a
temporary hidden ref, not a branch. It does not stage files or change HEAD, the
index, or local branches. Run one blob check at a time because the ref is shared.
This mode creates local Git objects and performs remote writes; use it when those
operations are authorized.

The check fetches every file through the GitHub contents endpoint at that commit,
requesting `application/vnd.github.html`. It rejects raw math code blocks,
unrendered math in visible prose, broken emphasis and markup inside math
renderers. It checks heading and explicit-anchor IDs from the returned HTML,
and local fragment links between scoped files. Links outside the snapshot retain
the offline checks against local files.

A `finally` block always attempts to delete the hidden ref, including when the
push, HTTP request or HTML processing fails. A cleanup failure exits with an
explicit diagnostic; the ref can then be removed with
`git push origin --delete refs/render-check/tmp`. Temporary objects remain in
Git's object database until normal garbage collection. An uncatchable process
termination can prevent cleanup.

Both network modes obtain a token through `git credential fill`, keep it only in
memory, and share rate-limit retries. Tokens and raw Git/network errors are never
printed or written. HTML caches must be outside the repository; omitting `--cache`
uses a temporary directory. Blob cache keys include the repository, file path and
commit ID, separately from the Markdown API cache.

The supplementary API check remains available with `--online`; both flags may be
combined. Unit tests inject Git, credentials and HTTP responses to exercise tree
building, HTML parsing, caching, retries and ref cleanup entirely offline.
