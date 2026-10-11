"""Offline regressions for GitHub math syntax; no credentials or network."""
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


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


if __name__ == "__main__":
    unittest.main()
