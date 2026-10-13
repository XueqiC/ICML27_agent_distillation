# Theory Markdown validation

Run from the repository root with Python 3.11 or later:

```sh
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theoretical_framework/tools/check_markdown.py
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python -m unittest discover -s theoretical_framework/tools -p 'test_*.py'
```

The default check and tests run offline, without credentials, network access or
OpenAI API calls. The scope is every Markdown file under `theoretical_framework/`,
`docs/direction/2026-10-11_v015_progress_ZH.md`, and every Markdown file directly
under `docs/direction/v015_extension/`. The revision and handoff directories are
excluded. The statement and label policy (S1) covers the framework's
`README.md`, `README_ZH.md`, `references.md`, `references_ZH.md`, all Markdown
files directly under `statements/`, `proofs/` and `experiments/`, and
`figures/README.md`. Historical direction documents, figure specs, tool
documentation and other pages receive rendering and link checks. All framework
pages except the two global reference catalogs also receive the page bibliography
rules below. Citation links anywhere in the selected scope receive the bilingual
reference-key check; the direction documents retain their legacy citation syntax.

Pass optional positional `PATH...` arguments to check exactly those files instead
of the default scope. Paths are relative to the repository root (regardless of
the current directory), or absolute paths inside the repository:

```sh
CUDA_VISIBLE_DEVICES='' theory/.venv/bin/python theoretical_framework/tools/check_markdown.py theoretical_framework/README.md theoretical_framework/tools/README.md
```

The same selection applies to the offline check, `--online` and `--blob`, including
when both flags are combined. Selected files in S1 keep its statement and label
policy. Every other selected file receives rendering and link checks,
including files outside the default scope, plus bilingual citation-key checks.
Selected framework pages also receive page bibliography checks.
Link targets are still checked even when unselected; this does not add those
targets to the rendering scope. The declaration index reads all S1 pages to
resolve references and duplicates. A selected result page brings its section's
statement and check numbering into scope across all pages in the same language.
Selecting a README or result page also checks its EN/ZH counterpart's declarations
and anchors, without validating that counterpart's rendering.

Paths are resolved through symlinks and repeated resolved paths are checked once.
Paths outside the repository (including symlink escapes), missing files,
directories and files without the `.md` extension are command-line errors,
detected before validation or network operations. With no paths, the default
scope is used. Diagnostics include per-rule hit counts before the final summary;
rules with no hits are omitted. Each diagnostic is one hit. Reference-key
availability is reported once per missing key and reference language per run.

The per-result layout uses one statement counter per section, shared by
Definition, Assumption, Theorem, Proposition, Lemma, Corollary, Remark, Example
and Algorithm. Both languages retain these English type words.

- Each result in sections 2–9 owns `statements/KK-JJ_<slug>.md` and
  `statements/KK-JJ_<slug>_ZH.md`, for example `06-03_information_bound.md`.
  Declare it at the start of a line as `**Theorem 6.3 (Title).**` or
  `**Theorem 6.3（标题）.**`. Each statement page has exactly one declaration,
  matching its numeric filename prefix. Section 1 declarations live in the
  framework READMEs (Definitions 1.1–1.5 in the current framework).
- Each declaration needs its explicit HTML anchor, such as an `a` element with `id="s6-3"`,
  within the preceding three non-blank lines. Code examples do not supply anchors
  or declarations. In each language, every number from `K.1` to `K.max` must be
  declared exactly once across the section, regardless of type.
- Optional proof and experiment pages use the identical basename in `proofs/`
  and `experiments/`, including the language suffix. A page without its matching
  statement page is an error. Restatements are not declarations: use
  `**Statement ([Theorem 6.3](../statements/06-03_information_bound.md#s6-3)).**`
  in a proof, or `**Statement under test ...**` in experiments. A numbered
  declaration in a proof or experiment page is rejected.
- Every proof and experiment page must link to its own statement page. Every
  statement page must link to `../README.md` (EN) or `../README_ZH.md` (ZH),
  and to its proof and experiment pages whenever they exist. Links may include
  fragments. Code examples and images do not count as navigation.
- Declare a check only in an experiment result page using
  `## Check K.J · Title`, with an `a` element with `id="cK-J"` within the preceding three
  non-blank lines. Its section K must match the filename's section; J need not
  match the result's item number. Each check is declared once per language,
  and each section's checks cover `1..max` without gaps. Checks on a single page
  need not be consecutive or ordered by number.
- Secondary check mentions must link to the declaring page's primary anchor;
  `### [Check 6.2](../experiments/06-03_information_bound.md#c6-2) · Title`
  is valid. Bare secondary mentions, links without the primary fragment,
  and repeated primary headings are errors.
- Each README and result page must have an EN/ZH counterpart with the same
  `(type, K.J)` declarations and the same set of explicit `sK-J` / `cK-J`
  anchors. Check declarations must also match per page; identical section totals
  with different check placement do not satisfy parity.
- Framework citation links target `#ref-key`, with the corresponding entry in
  the page's final bibliography. Each key must have an explicit
  anchor entry (an `a` element with `id="key"`) in **both** global reference files.
  The target and fragment must also pass ordinary link checks. A heading slug
  alone does not supply a reference entry. URL-encoded fragments are decoded
  before validation. Legacy global citation links in direction documents still
  receive the bilingual key check.
- Mentions such as `Theorem 6.3`, `Theorems 6.3 and 6.5`,
  `Propositions 5.5–5.9`, `Corollaries 6.7–6.8` and `Definitions 1.3–1.4`
  must resolve to declarations of the stated type. Range endpoints must exist
  and belong to the same section; intermediate numbers are not checked as
  mentions. Wrapped prose references are supported. References resolve across
  the framework, including unselected pages and either language. The labels
  `Theorem A`, `Theorem B` and `Theorem C` remain valid.
- Links ending in `#sK-J` or `#cK-J` must resolve to the declaring page and its
  anchor; a stray anchor on another page does not suffice.
- Section references `§1` through `§9`, including `§2–§9`, are valid. Bare source
  labels such as `§10`, `§5.2`, `A5` and `eqs. (A11)`, old result labels such as
  `Theorem 7`, `Thm. C4`, `Prop. A5`, `Corollary 9`, `Lemma W0` and `定理 3`,
  and node labels `N01`–`N20` (including parts) and `W0`–`W3` are rejected.
  Node labels are checked inside inline code, including the figure README.
  Link labels count as prose; link destinations and HTML attributes do not.
- Only the exact README sections `## Appendix · Sources` and `## 附录 · 出处`
  exempt source-document labels and statement/check mentions. The exemption
  includes subordinate headings and ends at the next level 1 or level 2 heading.
  Former appendix titles, `Source concordance` headings, and proof-page source
  lines do not create exemptions. Rendering and link checks always apply.
- README check codes such as `C5`, `V7`, `E12` and `T03` must be in code spans,
  including in the appendix. Experiment pages require both a code span and a
  suite prefix, for example `v014:C5`, `v015:E6`, `pkg:T12` or
  `v016:C4.amplitude`. Bare prose IDs and code spans without a prefix produce
  an `unprefixed check code` error. Prefixes use `v` followed by digits, or `pkg`.

Legacy section pages matching `0[2-9]_*.md` in `statements/`, `proofs/` or
`experiments/` remain in scope until the coordinator removes them. They retain
rendering, ordinary links and label checks, but do not supply declarations or
participate in declaration ownership, numbering, page pairing, navigation, or
secondary-check-link policy. Old `N??_*.md` pages receive the same treatment.
Thus legacy declarations never create duplicates or fill gaps in the new index.
References in legacy prose still resolve against the new index; undefined-label
hits can be expected while result pages are incomplete. Historical direction
documents remain exempt from statement, source-label, node-label and check-code
policy.

Display math uses `math` fences at column 0, with a blank line before and after.
GitHub renders the math in Markdown with MathJax ([Cervone, 2012](#ref-cervone2012mathjax)), and its
browser renderer stops rendering expressions past an undocumented per-page count.
Every page in the validation scope may contain at most 600 math expressions:
one per inline dollar/backtick expression and one per `math` block. Literal
examples inside code spans or non-math code fences do not count. Exceeding the
limit reports `too many math expressions`, including outside S1 and in the README
appendix. Split pages at statement or check boundaries, preserve each block's
anchor and proof, and retarget links to the page that owns the moved anchor.
The `math-fence-in-list` error rejects fenced math inside any bullet or numbered
list, including nested lists and indented continuations. It also rejects indented
math fences outside lists. Inline math uses the dollar/backtick form, such as
`` $`x`$ ``. When moving a display out of a list, convert sibling items into
paragraphs too, retaining their labels and bolding any explicit item numbers.

## Page bibliographies

Every `*.md` under `theoretical_framework/` except `references.md` and
`references_ZH.md` is a ref page, including figure specs and this tools README.
Direction documents are outside this policy. Names ending in `_ZH.md` use
Chinese bibliography headings and catalog entries; other pages use English.

A citation is an inline Markdown link such as `[Author, year](#ref-key)`.
The key is the id of an explicit HTML anchor that both global catalogs declare.
Links inside code spans or fenced blocks, images, and reference-link definitions
are not citations. `[References](../references.md)` is navigation, not a citation.
Body citation keys must match between an English page and its `_ZH` counterpart
when both exist. Each ref page needs at least one body citation.

The final H2 must be exactly `## References` or `## 参考文献`, appear once, and
contain exactly the generated introduction and entries. Each entry has an
explicit HTML anchor whose id is `ref-` plus the key, and the complete paragraph from the
appropriate global catalog, in English catalog order. The generator rebases
relative links, excludes global usage lines, and ends the file with one newline.
Every entry must be cited in the body; every body citation must resolve in this
section. The `ref-` anchor prefix is reserved for this section. A README's
`Appendix · Sources` / `附录 · 出处` remains an appendix when References follows it;
its label exemption ends at that next H2.

Inspect the proposed synchronization without writing:

```sh
theory/.venv/bin/python theoretical_framework/tools/sync_references.py --check
theory/.venv/bin/python theoretical_framework/tools/sync_references.py --check theoretical_framework/tools/README.md
```

Omitting `--check` writes the planned changes. Optional paths are repository-relative
or absolute ref pages or directories within the framework. A selected run rewrites
only those pages, but always recomputes both global usage indexes from all framework
body citations. No arguments selects all ref pages. `--check` exits 1 if files would
change or an error occurs, and 0 otherwise; it never writes. It also lists pages
without citations even when the files are already synchronized. A clean sync check
does not imply a clean validator run: the validator also rejects empty bibliographies
and citation parity mismatches. The summary reports pages rewritten, files that would
change, a distribution of body citation counts / distinct works, and uncited pages.

The sync tool converts legacy targets such as `../references.md#key`,
`../../references_ZH.md#key`, or any equivalent relative path to `#ref-key`.
Unknown keys and mismatched catalog key sets are errors. A final old reference
section may contain a generated bibliography or citation-led list items such as
`- [Author, year](../references.md#key). Full bibliographic text.` Other paragraphs,
bullets, headings, or math cause an error with a file and line number. The entire
plan is checked before any write. A citation present only in the old section is
excluded from body citations, and the tool warns for each key thereby removed.
Pages with no body citations get no generated section and remain validator errors.

Global `Used in:` / `引用于：` lines group statement, proof, and experiment citations
by framework item, sorted numerically and linked to the statement page using its
H1 item name. They then list citing overview pages, figure specs, tool documentation,
and other non-item pages. Bilingual citations are combined by item or page, with
links localized to available statement counterparts; non-item links always lead
to a page citing the work. Thus English-only figure specs also appear in the Chinese
index under localized labels. Entries with no citing page stay in the catalog with
`Used in: none` / `引用于：无` and produce warnings rather than errors. The catalog
header is preserved apart from its usage-description sentence.

Validator diagnostic codes:

| Code | Rule |
| --- | --- |
| `ref-section` | Exactly one correctly named final reference H2. |
| `ref-content` | Section exactly matches the shared generator. |
| `ref-key` | Every body citation key exists in the catalog. |
| `ref-target` | Every on-page body citation has its own section anchor. |
| `ref-padding` | No uncited entries. |
| `ref-legacy` | No body citations targeting the global catalog. |
| `ref-parity` | English and Chinese counterparts cite the same key set. |
| `ref-empty` | Every ref page has a body citation. |
| `ref-reserved` | No `ref-` anchors outside the section. |
| `ref-catalog` | Both catalogs parse and have identical key sets. |
| `ref-used-in` | Global usage lines match recomputed citations. |
| `ref-unused` | Warning: a catalog entry has no citing page. |

Selected-page validation reads the other framework pages for citation parity and
usage computation, but checks global usage lines only when their catalogs are
selected. All unit tests use temporary trees and fake network/credential clients.
This README's syntax examples are deliberately literal; its own substantive body
citation is still to be supplied by the documentation author.

## GitHub rendering

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

## References

Works cited on this page. The complete list, with the pages that use each work, is in [References](../references.md).

<a id="ref-cervone2012mathjax"></a>

**Davide Cervone (2012).** MathJax: A platform for mathematics on the Web. *Notices of the American Mathematical Society* 59, no. 2, 312–316. DOI [10.1090/noti794](https://doi.org/10.1090/noti794).
