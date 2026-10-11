"""Offline regressions for GitHub math syntax; no credentials or network."""
import contextlib
import hashlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError


spec = importlib.util.spec_from_file_location(
    "check_markdown", Path(__file__).with_name("check_markdown.py"))
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


class MarkdownTests(unittest.TestCase):
    def validate(self, text):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "page.md"
            path.write_text(text, encoding="utf-8")
            (path.parent / "target.md").write_text("# Target\n", encoding="utf-8")
            report = check.Report()
            out = io.StringIO()
            with contextlib.redirect_stdout(out), \
                    patch.object(check, "credential", side_effect=AssertionError("credential access")), \
                    patch.object(check, "urlopen", side_effect=AssertionError("network access")), \
                    patch.object(check.subprocess, "run", side_effect=AssertionError("subprocess access")):
                check.offline([path], {path}, report)
            return report.errors, out.getvalue()

    def test_backtick_math_in_rendering_contexts(self):
        for text in (
            r"中文，$`\mathbb E[(M_K-n)_+]`$。",
            r"prefix$`x_i`$",
            r"*$`(\mu^\top M)_\pi`$*",
            r"**$`\max_\pi`$**",
            r"- $`S_1`$ and $`S_k`$",
            "| Formula |\n|---|\n" + r"| $`\max_\pi`$ |",
            r"`code`$`x_i`$",
        ):
            with self.subTest(text=text):
                self.assertEqual(self.validate(text)[0], 0)

    def test_plain_dollars_are_always_rejected(self):
        for text in (r"$x$", r"$\max_\pi$", r"*$x_i$*", r"S$_1$", r"$`x`$ then $y$"):
            with self.subTest(text=text):
                self.assertIn("inline math must use", self.validate(text)[1])

    def test_parenthesized_tex_is_rejected(self):
        self.assertIn("inline math must use", self.validate(r"\(x_i\)")[1])

    def test_unbalanced_delimiters(self):
        for text in ("$x", "$`x`", "$`x$", "$`x`$ then $"):
            with self.subTest(text=text):
                self.assertIn("unbalanced inline dollars", self.validate(text)[1])

    def test_atx_headings_reject_math(self):
        for text in ("# $`x_i`$", "### **$x$** ###", r"# \(x_i\)"):
            with self.subTest(text=text):
                self.assertIn("math in heading", self.validate(text)[1])

    def test_setext_headings_reject_math(self):
        for underline in ("===", "---"):
            self.assertIn("math in heading", self.validate("$`x_i`$\n" + underline)[1])

    def test_inline_link_labels_reject_math(self):
        for label in (r"$`x_i`$", r"$x_i$", r"**$`x_i`$**", r"$`\mathbb E[(M_K-n)_+]`$", r"\(x_i\)"):
            with self.subTest(label=label):
                self.assertIn("math in link text", self.validate(f"[{label}](target.md)")[1])

    def test_image_alt_rejects_math(self):
        self.assertIn("math in link text", self.validate("![$`x`$](target.md)")[1])

    def test_reference_link_labels_reject_math(self):
        for text in (
            "[$`x`$][target]\n\n[target]: target.md",
            "[$`x`$][]\n\n[$`x`$]: target.md",
            "[$`x`$]\n\n[$`x`$]: target.md",
        ):
            with self.subTest(text=text):
                self.assertIn("math in link text", self.validate(text)[1])

    def test_brackets_without_a_link_allow_math(self):
        self.assertEqual(self.validate("[$`x`$]")[0], 0)

    def test_link_and_math_can_be_adjacent(self):
        self.assertEqual(self.validate("[label](target.md)$`x_i`$")[0], 0)

    def test_code_examples_are_literal(self):
        text = "`$x$` and `` $`x_i`$ ``\n\n```python\n'$x$'\n```\n"
        self.assertEqual(self.validate(text)[0], 0)

    def test_math_fences_remain_valid(self):
        text = "```math\n\\mathbb E[(M_K-n)_+]\n```\n"
        self.assertEqual(self.validate(text)[0], 0)

    def test_measured_list_fence_failure(self):
        text = "- **(b)** text $`x`$\n\n  ```math\n  z=1\n  ```\n"
        self.assertIn("math-fence-in-list", self.validate(text)[1])

    def test_list_fences_fail_regardless_of_inline_math_location(self):
        for inline in ("$`x`$", "$x$"):
            for marker in ("```", "~~~"):
                for prefix, suffix in (
                    (f"- {inline}\n- display", ""),
                    ("- display", f"\n- {inline}\n"),
                    (f"- display {inline}", ""),
                    ("- display", f"\n  after {inline}\n"),
                    ("- display", ""),
                ):
                    with self.subTest(inline=inline, marker=marker, prefix=prefix, suffix=suffix):
                        text = f"{prefix}\n\n  {marker}math\n  z=1\n  {marker}\n{suffix}"
                        self.assertIn("math-fence-in-list", self.validate(text)[1])

    def test_numbered_nested_and_same_line_list_fences(self):
        for text in (
            "1. text\n\n   ```math\n   z=1\n   ```\n",
            "12) text\n\n    ~~~math\n    z=1\n    ~~~\n",
            "- outer\n  - inner\n\n    ```math\n    z=1\n    ```\n",
            "1. outer\n   2. inner\n      - deep\n\n        ```math\n        z=1\n        ```\n",
            "- item\n  continuation\n\n  ```math\n  z=1\n  ```\n",
            "- ```math\n  z=1\n  ```\n",
            "1. - ~~~math\n     z=1\n     ~~~\n",
            "> - inner\n>\n>   ```math\n>   z=1\n>   ```\n",
            "- outer\n\n  > ```math\n  > z=1\n  > ```\n",
            "- > ```math\n  > z=1\n  > ```\n",
        ):
            with self.subTest(text=text):
                self.assertIn("math-fence-in-list", self.validate(text)[1])

    def test_indented_math_fences_without_lists_are_rejected(self):
        for indent in (" ", "  ", "   ", "    ", "\t"):
            with self.subTest(indent=indent):
                text = f"{indent}```math\n{indent}z=1\n{indent}```\n"
                self.assertIn("math-fence-in-list", self.validate(text)[1])

    def test_measured_top_level_math_passes(self):
        for text in (
            "**(b)** text $`x`$\n\n```math\nz=1\n```\n",
            "$`x`$\n\n```math\nz=1\n```\n\n$`y`$\n",
            "- text $`x`$\n\n```math\nz=1\n```\n",
            "1. text $`x`$\n\n~~~math\nz=1\n~~~\n",
            "- outer\n  - inner $`x`$\n\n```math\nz=1\n```\n",
            "> text $`x`$\n>\n> ```math\n> z=1\n> ```\n",
            "- text $`x`$\n\n> ```math\n> z=1\n> ```\n",
        ):
            with self.subTest(text=text):
                self.assertEqual(self.validate(text), (0, ""))

    def test_list_containers_and_fence_contents(self):
        text = "- outer\n  - inner\n\n  ```math\n  z=1\n  ```\n\n```math\nz=2\n```\n"
        openings = [row for row in check.block_lines(text) if row[3]]
        self.assertEqual([row[4] for row in openings], [True, False])
        text = "````markdown\n- item\n\n  ```math\n  z=1\n  ```\n````\n"
        self.assertNotIn("math-fence-in-list", self.validate(text)[1])

    def test_converted_numbered_clauses_keep_reference_allowance(self):
        self.assertEqual(self.validate("**1.** Condition.\n\nBy (1), done.\n")[0], 0)

    def test_scope_includes_all_framework_markdown_without_expanding_label_rules(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            names = ["theoretical_framework/README.md",
                     "theoretical_framework/proofs/N01.md",
                     "theoretical_framework/experiments/N01.md",
                     "theoretical_framework/figures/README.md",
                     "theoretical_framework/figures/specs/N01.md",
                     "theoretical_framework/tools/README.md",
                     "theoretical_framework/extra/deep/page.md",
                     "docs/direction/2026-10-11_v015_progress_ZH.md",
                     "docs/direction/v015_extension/Extra.md",
                     "docs/direction/v015_revision/untouched.md",
                     "docs/direction/v014_handoff/untouched.md"]
            for name in names:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("", encoding="utf-8")
            files, labels = check.scope(root)
            self.assertEqual(set(files), {root / name for name in names[:-2]})
            self.assertIn(root / names[1], labels)
            self.assertIn(root / names[3], labels)
            for index in (4, 5, 6, 8):
                self.assertNotIn(root / names[index], labels)
            spec = root / names[4]
            spec.write_text("Thm. 7, (32), C23.\n\n  ```math\n  z=1\n  ```\n", encoding="utf-8")
            report = check.Report()
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                check.offline([spec], labels, report)
            self.assertIn("math-fence-in-list", out.getvalue())
            self.assertNotIn("nonlocal", out.getvalue())
            self.assertNotIn("unlinked", out.getvalue())

    def test_math_body_safety_checks_still_apply(self):
        cases = {
            "$` x `$": "space at an inline math boundary",
            r"| $`|x|`$ |": "raw pipe in table math",
            r"$`\{x\}`$": "unsafe inline math escape",
            r"$`\operatorname{span} X`$": "denylisted macro",
        }
        for text, message in cases.items():
            with self.subTest(text=text):
                self.assertIn(message, self.validate(text)[1])

    def test_inline_formula_body_and_columns_are_preserved(self):
        line = r"`code` 中文，$`\mathbb E[(M_K-n)_+]`$ then $x_i$"
        tokens = list(check.inline_tokens(line))
        self.assertEqual([t[0] for t in tokens], ["code", "math", "math"])
        _, start, end, body, backtick = tokens[1]
        self.assertEqual(line[start:end], "$`" + body + "`$")
        self.assertEqual(body, r"\mathbb E[(M_K-n)_+]")
        self.assertTrue(backtick)
        self.assertFalse(tokens[2][-1])

    def test_escaped_dollars_are_literal(self):
        self.assertEqual(self.validate(r"Cost: \$5; $`x_i`$.")[0], 0)

    def test_link_positions_survive_math_and_code(self):
        line = r"`code` $`x_i`$ [a $`\mathbb E[(M_K-n)_+]`$](target.md#target)"
        label, dest, start, end = list(check.markdown_links(line))[0]
        self.assertEqual(label, r"a $`\mathbb E[(M_K-n)_+]`$")
        self.assertEqual(dest, "target.md#target")
        self.assertEqual(line[start:end], f"[{label}]({dest})")

    def test_balanced_parentheses_in_link_destinations(self):
        self.assertEqual(list(check.markdown_links("[label](path_(copy).md)"))[0][1], "path_(copy).md")

    def test_missing_targets_and_anchors_still_fail(self):
        self.assertIn("missing link target", self.validate("[x](missing.md)")[1])
        self.assertIn("missing heading anchor", self.validate("[x](target.md#missing)")[1])

    def test_legacy_anchor_preserves_old_link(self):
        text = '<a id="old-title"></a>\n\n# New title\n\n[x](#old-title) [y](#new-title)\n'
        self.assertEqual(self.validate(text)[0], 0)
        self.assertNotIn("fake", check.headings('```html\n<a id="fake"></a>\n```'))

    def test_heading_slug_collisions_still_work(self):
        self.assertEqual(check.headings("# Same\n\n# Same\n"), {"same": 1, "same-1": 3})

    def test_heading_slug_drops_superscript_number(self):
        heading = '## Panel C — "Warm Adam: E[G²] enters through v"'
        self.assertEqual(check.headings(heading), {
            "panel-c--warm-adam-eg-enters-through-v": 1,
        })

    def test_heading_slug_drops_subscript_number_but_keeps_modifier_letter(self):
        heading = '## Panel E — "LoRA at B=0: only the sketch ΓA₀ᵀ matters"'
        self.assertEqual(check.headings(heading), {
            "panel-e--lora-at-b0-only-the-sketch-γaᵀ-matters": 1,
        })

    def test_heading_slug_unicode_word_categories(self):
        # Keep letters, combining marks, decimal digits and connector punctuation;
        # drop letter numbers (Ⅳ) and other numbers (½), symbols and non-space whitespace.
        self.assertEqual(check.github_slug("Ée\u0301 中ᵀ 0١ a_b‿c - Ⅳ½+\t"),
                         "ée\u0301-中ᵀ-0١-a_b‿c---")

    def test_online_html_parser_behavior_is_unchanged(self):
        parser = check.RenderedHTML()
        parser.feed('<p><math-renderer>x_i</math-renderer><code>$literal$</code>$raw$</p>')
        self.assertEqual("".join(parser.text).strip(), "$raw$")


class OnlineHTMLTests(unittest.TestCase):
    def validate(self, rendered, source=""):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "page.md"
            path.write_text(source, encoding="utf-8")
            report = check.Report()
            out = io.StringIO()
            with contextlib.redirect_stdout(out), \
                    patch.object(check, "credential", return_value="fake-token"), \
                    patch.object(check, "fetch_html", return_value=rendered), \
                    patch.object(check, "urlopen", side_effect=AssertionError("network access")), \
                    patch.object(check.subprocess, "run", side_effect=AssertionError("subprocess access")):
                check.online([path], Path(directory) / "cache", report)
            return report.errors, out.getvalue()

    def test_legitimate_emphasis_on_lines_with_math(self):
        for tag, marker in (("em", "*"), ("strong", "**")):
            for label in ("(a)", "Step 1 (reduction)"):
                with self.subTest(tag=tag, label=label):
                    source = f"{marker}{label}{marker} " + r"$`\alpha_i`$"
                    rendered = (f"<p><{tag}>{label}</{tag}> "
                                r"<math-renderer>\alpha_i</math-renderer></p>")
                    self.assertEqual(self.validate(rendered, source)[0], 0)

    def test_math_and_link_attributes_are_excluded_from_emphasis(self):
        for tag in ("em", "strong"):
            with self.subTest(tag=tag):
                rendered = (f"<p><{tag}>See "
                            r'<a href="target/$\alpha">the result</a> '
                            r"<math-renderer>\alpha + \$</math-renderer>"
                            f"</{tag}></p>")
                self.assertEqual(self.validate(rendered)[0], 0)

    def test_broken_formula_in_emphasis(self):
        for tag, marker in (("em", "*"), ("strong", "**")):
            with self.subTest(tag=tag):
                source = f"$x{marker}\\alpha{marker}y$"
                rendered = f"<p>$x<{tag}>\\alpha</{tag}>y$</p>"
                errors, output = self.validate(rendered, source)
                self.assertGreater(errors, 0)
                self.assertIn("GitHub emphasis contains", output)

    def test_emphasis_checks_rendered_text_outside_math(self):
        for tag in ("em", "strong"):
            for body in ("\\", "$", "&#92;", "&#36;", r'<a href="target">\alpha</a>',
                         "<code>$</code>", r"<span>\</span>"):
                with self.subTest(tag=tag, body=body):
                    _, output = self.validate(f"<{tag}>{body}</{tag}>")
                    self.assertIn("GitHub emphasis contains", output)

    def test_nested_emphasis_keeps_each_elements_text(self):
        rendered = (r"<em>before <strong>\alpha</strong> after "
                    r"<math-renderer>\beta</math-renderer></em>")
        parser = check.RenderedHTML()
        parser.feed(rendered)
        self.assertEqual(parser.emphasis, [r"\alpha", r"before \alpha after "])
        _, output = self.validate(rendered)
        self.assertEqual(output.count("GitHub emphasis contains"), 2)

    def test_markup_inside_math_renderers_is_rejected(self):
        for tag in ("em", "strong", "a", "code"):
            with self.subTest(tag=tag):
                rendered = (f"<math-renderer>x<sup><{tag}>i</{tag}></sup>"
                            "</math-renderer>")
                errors, output = self.validate(rendered)
                self.assertEqual(errors, 1)
                self.assertIn(f"markup inside GitHub math-renderer: {tag}", output)

    def test_each_broken_math_renderer_is_reported_once(self):
        rendered = ("<math-renderer><em>i</em><strong>j</strong></math-renderer>"
                    "<math-renderer><a>k</a><code>l</code></math-renderer>")
        errors, output = self.validate(rendered)
        self.assertEqual(errors, 2)
        self.assertEqual(output.count("markup inside GitHub math-renderer"), 2)

    def test_emphasis_inside_math_does_not_close_outer_emphasis(self):
        rendered = (r"<em>before <math-renderer><em>\alpha</em></math-renderer>"
                    "$after</em>")
        parser = check.RenderedHTML()
        parser.feed(rendered)
        self.assertEqual(parser.emphasis, ["before $after"])
        _, output = self.validate(rendered)
        self.assertIn("GitHub emphasis contains", output)
        self.assertIn("markup inside GitHub math-renderer", output)

    def test_escaped_tags_and_attributes_are_not_math_markup(self):
        rendered = (r'<math-renderer data-example="&lt;em&gt;">'
                    r"\alpha &lt;em&gt;i&lt;/em&gt;</math-renderer>")
        self.assertEqual(self.validate(rendered)[0], 0)

    def test_other_online_prose_checks_are_preserved(self):
        for rendered in ("<p>$raw$</p>", r"<p>\alpha</p>"):
            with self.subTest(rendered=rendered):
                self.assertIn("unrendered math in GitHub HTML", self.validate(rendered)[1])
        rendered = (r"<pre>\alpha $</pre><code>\beta $</code>"
                    r'<div class="mermaid">\gamma $</div>')
        self.assertEqual(self.validate(rendered)[0], 0)

    def test_raw_math_pre_is_rejected(self):
        errors, output = self.validate('<pre class="code" lang="math"><code>z=1</code></pre>')
        self.assertEqual(errors, 1)
        self.assertIn("unrendered display math", output)

    def test_svg_and_encoded_visible_math(self):
        self.assertEqual(self.validate(r'<svg><text>$ \alpha</text></svg>')[0], 0)
        self.assertIn("unrendered math", self.validate('<p>&#36;x&#36; &#92;alpha</p>')[1])


class FakeGit:
    """In-memory Git object graph; never calls Git or touches a real ref."""
    def __init__(self, fail=None):
        self.calls = []
        self.objects = {}
        self.fail = fail

    def __call__(self, args, data=None):
        self.calls.append((args, data))
        if args[:3] == ["remote", "get-url", "origin"]:
            return "git@github.com:example/theory.git"
        if args[0] == "push":
            if self.fail == "push" and "-f" in args or self.fail == "delete" and "--delete" in args:
                raise RuntimeError("Injected Git failure")
            return ""
        kind = "blob" if args[0] == "hash-object" else "tree" if args[0] == "mktree" else "commit"
        oid = hashlib.sha1(kind.encode() + b"\0" + data).hexdigest()
        self.objects[oid] = (kind, data, args)
        return oid


class BlobTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        base = Path(self.temporary.name)
        self.root = base / "repository"
        self.cache = base / "cache"
        self.paths = [self.root / "theoretical_framework/figures/specs/中文 page.md",
                      self.root / "docs/direction/v015_extension/page.md"]
        for path in self.paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# Title\n", encoding="utf-8")
        self.git = FakeGit()
        self.requests = []
        self.html = '<h1 id="user-content-title">Title</h1><math-renderer>z=1</math-renderer>'
        self.report = check.Report()
        self.out = io.StringIO()
        self.sleeps = []
        # No test may accidentally fall through to live network/Git/credentials.
        for obj, name in ((check, "urlopen"), (check, "credential"), (check.subprocess, "run")):
            guard = patch.object(obj, name, side_effect=AssertionError("live I/O attempted"))
            guard.start()
            self.addCleanup(guard.stop)

    def opener(self, request, timeout):
        self.requests.append(request)
        return io.BytesIO(self.html.encode())

    def run_blob(self, **kwargs):
        with contextlib.redirect_stdout(self.out):
            check.blob(self.paths, self.cache, self.report, root=self.root, git=self.git,
                       get_credential=lambda: "fake-token", opener=kwargs.pop("opener", self.opener),
                       sleep=self.sleeps.append, **kwargs)

    def test_working_tree_snapshot_preserves_paths_and_has_no_parent(self):
        self.paths[0].write_bytes(b"working version, not index or HEAD\n")
        sha = check.build_blob_commit(self.paths, self.root, self.git)
        kind, _, args = self.git.objects[sha]
        self.assertEqual(kind, "commit")
        self.assertNotIn("-p", args)
        self.assertEqual(args[-2], "commit-tree")
        reconstructed = {}

        def walk(oid, prefix=""):
            kind, data, args = self.git.objects[oid]
            self.assertEqual(kind, "tree")
            self.assertEqual(args, ["mktree", "-z"])
            for record in data.rstrip(b"\0").split(b"\0"):
                info, name = record.decode().split("\t")
                mode, kind, child = info.split()
                if kind == "tree":
                    self.assertEqual(mode, "040000")
                    walk(child, prefix + name + "/")
                else:
                    self.assertEqual(mode, "100644")
                    reconstructed[prefix + name] = self.git.objects[child][1]
        walk(args[-1])
        self.assertEqual(reconstructed, {p.relative_to(self.root).as_posix(): p.read_bytes() for p in self.paths})
        self.assertTrue(all(args == ["hash-object", "-w", "--stdin"]
                            for kind, _, args in self.git.objects.values() if kind == "blob"))

    def test_real_file_view_request_and_cleanup(self):
        self.run_blob()
        self.assertEqual(self.report.errors, 0)
        self.assertEqual(len(self.requests), len(self.paths))
        push = next(args for args, _ in self.git.calls if args[:2] == ["push", "-f"])
        sha, ref = push[-1].split(":")
        self.assertEqual(push[:3], ["push", "-f", "origin"])
        self.assertEqual(ref, "refs/render-check/tmp")
        self.assertEqual(self.git.calls[-1][0], ["push", "origin", "--delete", check.BLOB_REF])
        for request in self.requests:
            self.assertEqual(request.get_method(), "GET")
            self.assertEqual(request.get_header("Accept"), "application/vnd.github.html")
            self.assertEqual(request.get_header("Authorization"), "Bearer fake-token")
            self.assertTrue(request.full_url.startswith("https://api.github.com/repos/example/theory/contents/"))
            self.assertTrue(request.full_url.endswith(f"?ref={sha}"))
            self.assertNotIn("fake-token", request.full_url)
        self.assertIn("%E4%B8%AD%E6%96%87%20page.md", self.requests[0].full_url)
        for cached in self.cache.glob("*.html"):
            self.assertEqual(cached.read_text(), self.html)
        self.assertNotIn("fake-token", self.out.getvalue() + repr(self.git.calls))

    def test_cleanup_after_fetch_failure(self):
        def fail(request, timeout):
            raise URLError("untrusted network error with fake-token")
        with self.assertRaisesRegex(RuntimeError, "GitHub Markdown request failed") as error:
            self.run_blob(opener=fail)
        self.assertNotIn("fake-token", str(error.exception))
        self.assertEqual(self.git.calls[-1][0], ["push", "origin", "--delete", check.BLOB_REF])

    def test_cleanup_after_push_failure_or_interruption(self):
        self.git.fail = "push"
        with self.assertRaisesRegex(RuntimeError, "Injected Git failure"):
            self.run_blob()
        self.assertEqual(self.git.calls[-1][0], ["push", "origin", "--delete", check.BLOB_REF])
        self.git.fail = None
        with self.assertRaises(KeyboardInterrupt):
            self.run_blob(opener=lambda *a, **k: (_ for _ in ()).throw(KeyboardInterrupt()))
        self.assertEqual(self.git.calls[-1][0], ["push", "origin", "--delete", check.BLOB_REF])

    def test_cleanup_after_parser_failure(self):
        with patch.object(check, "html_report", side_effect=ValueError("parse failure")):
            with self.assertRaisesRegex(ValueError, "parse failure"):
                self.run_blob()
        self.assertEqual(self.git.calls[-1][0], ["push", "origin", "--delete", check.BLOB_REF])

    def test_cleanup_failure_is_not_silenced(self):
        self.git.fail = "delete"
        with self.assertRaisesRegex(RuntimeError, "Could not delete refs/render-check/tmp"):
            self.run_blob()

    def test_rendered_failure_still_deletes_ref(self):
        self.html += '<pre lang="math">z=1</pre><p>$raw$</p><strong>&#92;alpha</strong>'
        self.run_blob()
        for message in ("unrendered display math", "unrendered math", "GitHub emphasis contains"):
            self.assertIn(message, self.out.getvalue())
        self.assertEqual(self.git.calls[-1][0], ["push", "origin", "--delete", check.BLOB_REF])

    def test_blob_anchors_use_actual_ids_and_percent_decoded_links(self):
        self.paths[0].write_text('# Title\n\n[other](../../../docs/direction/v015_extension/page.md#%E4%B8%AD%E6%96%87)\n')
        self.paths[1].write_text('# 中文\n\n<a id="legacy"></a>\n')
        self.html += '<h2 id="user-content-中文">中文</h2><a id="user-content-legacy"></a>'
        self.run_blob()
        self.assertEqual(self.report.errors, 0)
        self.html = '<h1 id="user-content-title">Title</h1><a href="#中文">not an ID</a>'
        for path in self.cache.glob("*.html"):
            path.unlink()
        self.run_blob()
        self.assertIn("missing GitHub HTML anchor: 中文", self.out.getvalue())
        self.assertIn("missing GitHub HTML anchor: legacy", self.out.getvalue())
        self.assertIn("missing GitHub HTML link anchor", self.out.getvalue())

    def test_cache_reuse_avoids_http_but_still_cleans_ref(self):
        self.run_blob()
        self.requests.clear()
        self.run_blob()
        self.assertEqual(self.requests, [])
        self.assertEqual(self.git.calls[-1][0], ["push", "origin", "--delete", check.BLOB_REF])

    def test_rate_limit_retry_uses_shared_policy(self):
        attempts = []
        def limited(request, timeout):
            attempts.append(request)
            if len(attempts) == 1:
                raise HTTPError(request.full_url, 429, "rate limit", {"Retry-After": "2"}, io.BytesIO())
            return self.opener(request, timeout)
        self.run_blob(opener=limited)
        self.assertEqual(self.sleeps, [2.0])
        self.assertEqual(len(attempts), 3)

    def test_rate_limit_exhaustion_cleans_ref(self):
        def limited(request, timeout):
            raise HTTPError(request.full_url, 403, "rate limit", {"Retry-After": "1"}, io.BytesIO())
        with self.assertRaisesRegex(RuntimeError, "rate-limit retries exhausted"):
            self.run_blob(opener=limited)
        self.assertEqual(len(self.sleeps), 8)
        self.assertEqual(self.git.calls[-1][0], ["push", "origin", "--delete", check.BLOB_REF])

    def test_cache_must_be_outside_repository(self):
        self.cache = self.root / "cache"
        with self.assertRaisesRegex(RuntimeError, "outside the repository"):
            self.run_blob()
        self.assertEqual(self.git.calls, [])

    def test_remote_url_parsing(self):
        for url in ("git@github.com:example/theory.git", "https://github.com/example/theory.git",
                    "ssh://git@github.com/example/theory.git", "https://github.com/example/theory/"):
            with self.subTest(url=url):
                self.assertEqual(check.github_repository(url), "example/theory")
        for url in ("https://fake-token@example.net/example/theory.git", "/tmp/theory", "https://github.com/one"):
            with self.subTest(url=url):
                with self.assertRaises(RuntimeError) as error:
                    check.github_repository(url)
                self.assertNotIn(url, str(error.exception))

    def test_blob_cli_dispatch(self):
        with patch.object(check, "scope", return_value=([], set())), \
                patch.object(check, "blob") as run, contextlib.redirect_stdout(self.out):
            self.assertEqual(check.main(["--blob", "--cache", str(self.cache)]), 0)
        run.assert_called_once()
        self.assertIn("GitHub file view", self.out.getvalue())


if __name__ == "__main__":
    unittest.main()
