# Theory Markdown validation

Run from the repository root with Python 3.11 or later:

```sh
CUDA_VISIBLE_DEVICES='' python theoretical_framework/tools/check_markdown.py
CUDA_VISIBLE_DEVICES='' python -m unittest discover -s theoretical_framework/tools -p 'test_check_markdown.py'
```

The default check and tests run offline, without credentials, network access or
OpenAI API calls. The scope is every Markdown file under `theoretical_framework/`,
`docs/direction/2026-10-11_v015_progress_ZH.md`, and every Markdown file directly
under `docs/direction/v015_extension/`. The revision and handoff directories are
excluded. Source-label and check-ID rules retain their previous scope; figure
specs, tool documentation and other newly included pages receive rendering and
link checks only.

Pass optional positional `PATH...` arguments to check exactly those files instead
of the default scope. Paths are relative to the repository root (regardless of
the current directory), or absolute paths inside the repository:

```sh
CUDA_VISIBLE_DEVICES='' python theoretical_framework/tools/check_markdown.py theoretical_framework/README.md theoretical_framework/tools/README.md
```

The same selection applies to the offline check, `--online` and `--blob`, including
when both flags are combined. Selected files in the default S1 set keep its
source-label and check-ID rules: the framework's `README.md` and `README_ZH.md`,
Markdown files directly under `proofs/` and `experiments/`, `figures/README.md`, and
`docs/direction/2026-10-11_v015_progress_ZH.md`. Every other selected file receives
rendering and link checks only, including files outside the default scope.
Link targets are still checked even when they are not selected; this does not
add those targets to the validation or rendering scope.

Paths are resolved through symlinks and repeated resolved paths are checked once.
Paths outside the repository (including symlink escapes), missing files,
directories and files without the `.md` extension are command-line errors,
detected before validation or network operations. With no paths, the default
scope and rules are unchanged.

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
