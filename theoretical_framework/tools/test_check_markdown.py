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

    def test_math_expression_budget_includes_inline_and_display(self):
        for inline, display in ((600, 0), (0, 600), (599, 1)):
            with self.subTest(inline=inline, display=display):
                text = "$`x`$\n\n" * inline + "```math\nx=1\n```\n\n" * display
                self.assertEqual(check.math_expression_count(text), 600)
                self.assertEqual(self.validate(text), (0, ""))
                errors, output = self.validate(text + "$`y`$\n")
                self.assertEqual(errors, 1, output)
                self.assertIn("too many math expressions: 601 (maximum 600)", output)

    def test_math_expression_budget_ignores_literal_examples(self):
        examples = ("`` $`x`$ ``\n\n"
                    "````text\n$`x`$\n\n```math\nx=1\n```\n````\n\n"
                    r"\$`x`\$" + "\n\n")
        text = "$`x`$\n\n" * 600 + examples
        self.assertEqual(check.math_expression_count(text), 600)
        self.assertEqual(self.validate(text), (0, ""))
        # A multiline display is one expression; code and TeX inside it are
        # never counted as additional inline expressions or opening fences.
        self.assertEqual(check.math_expression_count(
            "```math\na+b\nc+d\n```\n\n> $`z`$\n\n- $`w`$\n"), 3)

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
                     "theoretical_framework/statements/02_page.md",
                     "theoretical_framework/statements/02_page_ZH.md",
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
            self.assertIn(root / names[9], labels)
            self.assertIn(root / names[10], labels)
            for index in (4, 5, 6, 7, 8):
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


class FrameworkPolicyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.framework = self.root / "theoretical_framework"
        for name in ("README.md", "README_ZH.md", "figures/README.md"):
            self.write(name, "")
        self.write("docs/direction/2026-10-11_v015_progress_ZH.md", "")
        for obj, name in ((check, "urlopen"), (check, "credential"), (check.subprocess, "run")):
            guard = patch.object(obj, name, side_effect=AssertionError("live I/O attempted"))
            guard.start()
            self.addCleanup(guard.stop)

    def write(self, name, text):
        path = (self.root if name.startswith("docs/") else self.framework) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def pair(self, name, en, zh=None):
        self.write(name, en)
        self.write(name.removesuffix(".md") + "_ZH.md", en if zh is None else zh)

    def statement(self, word, number, zh=False):
        title = "（标题）" if zh else " (Title)"
        return f'<a id="s{number.replace(".", "-")}"></a>\n\n**{word} {number}{title}.**\n\n'

    def declaration_check(self, number):
        return f'<a id="c{number.replace(".", "-")}"></a>\n\n## Check {number} · Title\n\n'

    def validate(self, *names):
        files, s1 = check.scope(self.root)
        if names:
            files = [(self.root if name.startswith("docs/") else self.framework) / name for name in names]
        report, output = check.Report(), io.StringIO()
        with contextlib.redirect_stdout(output):
            check.offline(files, s1, report)
        self.assertEqual(report.errors, sum(report.counts.values()))
        return report.errors, output.getvalue()

    def result(self, number, word="Theorem", *, slug="item", proof=False, checks=None):
        section, item = map(int, number.split("."))
        basename = f"{section:02d}-{item:02d}_{slug}"
        for zh in (False, True):
            suffix = "_ZH" if zh else ""
            name = basename + suffix + ".md"
            links = f"[Framework](../README{suffix}.md)\n\n"
            statement_link = f"[{word} {number}](../statements/{name}#s{number.replace('.', '-')})"
            if proof:
                links += f"[Proof](../proofs/{name})\n\n"
                self.write(f"proofs/{name}", f"**Statement ({statement_link}).**\n\nRestated text.\n")
            if checks is not None:
                links += f"[Experiments](../experiments/{name})\n\n"
                self.write(f"experiments/{name}", f"**Statement under test ({statement_link}).**\n\n"
                           + "".join(self.declaration_check(n) for n in checks))
            self.write(f"statements/{name}", links + self.statement(word, number, zh))
        return basename + ".md"

    def append(self, name, text):
        path = self.framework / name
        self.write(name, path.read_text(encoding="utf-8") + text)

    def test_all_statement_types_and_zh_declarations(self):
        for j, word in enumerate(check.STATEMENT_WORDS, 1):
            self.result(f"2.{j}", word)
        self.pair("README.md", "".join(self.statement("Definition", f"1.{j}") for j in range(1, 6)))
        self.write("figures/README.md", "Definitions 1.1 and 2.1; Theorem A, Theorem B, Theorem C; Theorems A, B and C.")
        self.assertEqual(self.validate(), (0, ""))

    def test_references_resolve_in_unselected_pages(self):
        name = self.result("6.1")
        self.write("README.md", f"Theorem 6.1 and [Theorem 6.1](statements/{name}#s6-1).")
        self.assertEqual(self.validate("README.md"), (0, ""))
        (self.framework / "statements" / name).unlink()
        self.write("README.md", "Theorem 6.1.")
        self.assertEqual(self.validate("README.md"), (0, ""))

    def test_undefined_and_wrong_type_in_prose_links_and_zh(self):
        name = self.result("6.1")
        for page in ("README.md", "README_ZH.md"):
            for phrase, error in (("Theorem 6.2", "undefined framework statement"),
                                  ("Proposition 6.1", "statement type mismatch"),
                                  (f"[Lemma 6.1](statements/{name}#s6-1)", "statement type mismatch")):
                with self.subTest(page=page, phrase=phrase):
                    self.write(page, "参见 " + phrase + "。")
                    self.assertIn(error, self.validate(page)[1])

    def test_plural_ranges_and_lists_check_endpoints(self):
        for j in range(1, 10):
            self.result(f"5.{j}", "Proposition")
        words = ["Definition", "Assumption", "Theorem", "Lemma", "Theorem", "Remark", "Corollary", "Corollary"]
        for j, word in enumerate(words, 1):
            self.result(f"6.{j}", word)
        self.pair("README.md", "".join(self.statement("Definition", f"1.{j}") for j in range(1, 5)))
        for phrase in ("Propositions 5.5–5.9", "Theorems 6.3 and 6.5", "Corollaries 6.7–6.8",
                       "Definitions 1.3–1.4", "Propositions 5.5, 5.7, and 5.9", "Propositions 5.5、5.6 和 5.9",
                       "Theorems [6.3](../statements/06-03_item.md#s6-3) and [6.5](../statements/06-05_item.md#s6-5)",
                       "Theorems 6.3–6.5"):
            with self.subTest(phrase=phrase):
                self.write("figures/README.md", phrase + ".")
                self.assertEqual(self.validate(), (0, ""))
        for phrase, error in (("Propositions 5.5–5.10", "undefined framework statement: Proposition 5.10"),
                              ("Propositions 5.12–5.13", "undefined framework statement: Proposition 5.12"),
                              ("Theorems 6.3 and 6.4", "statement type mismatch: Theorem 6.4"),
                              ("Propositions 5.5–6.5", "framework range crosses sections"),
                              ("Definitions 1.3–6.1", "framework range crosses sections")):
            with self.subTest(phrase=phrase):
                self.write("figures/README.md", phrase)
                self.assertIn(error, self.validate()[1])

    def test_declarations_not_mentions_headings_code_or_appendix(self):
        examples = ("Theorem 2.1 (Title).", "## Theorem 2.1 (Title)",
                    "> **Theorem 2.1 (Title).**", "`**Theorem 2.1 (Title).**`",
                    "```text\n**Theorem 2.1 (Title).**\n```", "**[Theorem 2.1](#title) · Title**")
        for example in examples:
            with self.subTest(example=example):
                self.pair("statements/02-01_item.md", "# Title\n\n" + example)
                self.write("README.md", "Theorem 2.1.")
                self.assertIn("undefined framework statement", self.validate("README.md")[1])
                self.assertIn("statement declaration count: expected 1, got 0",
                              self.validate("statements/02-01_item.md")[1])
        self.write("README.md", "## Appendix · Sources\n\n" + self.statement("Theorem", "1.1"))
        self.write("README_ZH.md", "Theorem 1.1.")
        self.assertIn("undefined framework statement", self.validate("README_ZH.md")[1])

    def test_wrapped_references_report_the_relevant_line(self):
        self.result("2.1")
        self.result("2.2")
        self.write("README.md", "Theorems 2.1 and\n2.2.\n\nTheorem\n2.1.")
        self.assertEqual(self.validate(), (0, ""))
        self.write("README.md", "Theorems 2.1 and\n2.99.")
        self.assertIn("README.md:2: undefined framework statement: Theorem 2.99", self.validate()[1])
        self.write("README.md", "Theorem 2.1 and\n\n```text\n2.99\n```\n\n2.99")
        self.assertEqual(self.validate(), (0, ""))

    def test_legacy_pages_do_not_supply_or_shadow_declarations(self):
        name = self.result("2.1", "Definition", checks=["2.1"])
        for folder in ("statements", "proofs", "experiments"):
            for old in ("02_old.md", "09_old_ZH.md", "N01_old.md"):
                self.write(f"{folder}/{old}", self.statement("Theorem", "2.1")
                           + self.statement("Theorem", "2.2") + self.declaration_check("2.2"))
        self.assertEqual(self.validate(f"statements/{name}"), (0, ""))
        self.write("README.md", "Theorem 2.1. Theorem 2.2.")
        errors, output = self.validate("README.md")
        self.assertEqual(errors, 2, output)
        self.assertIn("statement type mismatch", output)
        self.assertIn("undefined framework statement", output)
        for folder in ("statements", "proofs", "experiments"):
            output = self.validate(f"{folder}/02_old.md")[1]
            for rule in ("duplicate framework", "declared in wrong", "numbering", "EN/ZH parity",
                         "declaration count", "missing required page link", "missing matching statement"):
                self.assertNotIn(rule, output)
            self.append(f"{folder}/02_old.md", "$bad$ [link](missing.md)\nN01")
            output = self.validate(f"{folder}/02_old.md")[1]
            for rule in ("inline math must use", "missing link target", "retired node label"):
                self.assertIn(rule, output)

    def test_exactly_one_declaration_and_duplicates_per_language(self):
        name = self.result("2.1")
        self.assertEqual(self.validate(), (0, ""))
        self.append(f"statements/{name}", self.statement("Theorem", "2.1"))
        output = self.validate(f"statements/{name}")[1]
        self.assertIn("statement declaration count: expected 1, got 2", output)
        self.assertIn("duplicate framework statement: 2.1", output)
        self.result("2.1")
        self.result("2.1", "Proposition", slug="duplicate")
        self.assertIn("duplicate framework statement: 2.1", self.validate(f"statements/{name}")[1])
        self.write("README.md", self.statement("Definition", "1.1") * 2)
        self.assertIn("duplicate framework statement: 1.1", self.validate("README.md")[1])

    def test_declarations_belong_to_filename_section_and_page_kind(self):
        for page, body, kind in (("README.md", self.statement("Theorem", "2.1"), "statement"),
                                 ("statements/02-01_item.md", self.statement("Definition", "1.1"), "statement"),
                                 ("statements/02-01_item.md", self.statement("Theorem", "2.2"), "statement"),
                                 ("statements/02-01_item.md", self.statement("Theorem", "3.1"), "statement"),
                                 ("proofs/02-01_item.md", self.statement("Theorem", "2.1"), "statement"),
                                 ("experiments/02-01_item.md", self.statement("Theorem", "2.1"), "statement"),
                                 ("references.md", self.statement("Theorem", "2.1"), "statement"),
                                 ("proofs/02-01_item.md", self.declaration_check("2.1"), "check"),
                                 ("statements/02-01_item.md", self.declaration_check("2.1"), "check"),
                                 ("README.md", self.declaration_check("2.1"), "check"),
                                 ("experiments/02-01_item.md", self.declaration_check("3.1"), "check")):
            with self.subTest(page=page, kind=kind):
                self.pair(page, body)
                self.assertIn(f"{kind} declared in wrong section/page", self.validate(page)[1])
        self.write("figures/README.md", self.statement("Theorem", "3.1"))
        self.write("README.md", "Theorem 3.1")
        self.assertIn("undefined framework statement", self.validate("README.md")[1])

    def test_result_filenames_require_two_digit_prefix_and_slug(self):
        for name in ("2-01_item.md", "02-1_item.md", "02-01_.md", "01-01_item.md", "page.md"):
            with self.subTest(name=name):
                self.write("statements/" + name, self.statement("Theorem", "2.1"))
                self.assertIn("invalid result page filename", self.validate("statements/" + name)[1])

    def test_anchor_adjacency_uses_three_nonblank_lines(self):
        name = self.result("2.1", checks=["2.1"])
        for kind, folder, prefix, declaration in (
            ("statement", "statements", "s", "**Theorem 2.1（标题）.**"),
            ("check", "experiments", "c", "## Check 2.1 · 标题"),
        ):
            for middle in ("\n\n\n\n", "\nA line.\n\nAnother line.\n"):
                self.pair(f"{folder}/{name}", f'<a id="{prefix}2-1"></a>\n{middle}{declaration}')
                self.assertNotIn(f"missing adjacent {kind} anchor", self.validate()[1])
            for text in (declaration, f'<a id="{prefix}2-2"></a>\n{declaration}',
                         f'<a id="{prefix}2-1"></a>\nA\nB\nC\n{declaration}',
                         f'{declaration}\n<a id="{prefix}2-1"></a>',
                         f'```html\n<a id="{prefix}2-1"></a>\n```\n{declaration}',
                         f'`<a id="{prefix}2-1"></a>`\n{declaration}'):
                with self.subTest(kind=kind, text=text):
                    self.pair(f"{folder}/{name}", text)
                    self.assertIn(f"missing adjacent {kind} anchor", self.validate()[1])

    def test_section_numbering_has_no_gaps_in_each_language(self):
        self.result("5.1", checks=["5.1"])
        name = self.result("5.3", checks=["5.3"])
        for kind, folder in (("statement", "statements"), ("check", "experiments")):
            for selection in ((), (f"{folder}/{name}",)):
                self.assertIn(f"{kind} numbering gap: missing 5.2 across section 5 (EN)",
                              self.validate(*selection)[1])
        self.result("5.2", checks=["5.2"])
        self.assertEqual(self.validate(), (0, ""))
        (self.framework / "statements/05-02_item_ZH.md").unlink()
        self.write("experiments/05-02_item_ZH.md", "")
        output = self.validate()[1]
        for kind in ("statement", "check"):
            self.assertNotIn(f"{kind} numbering gap: missing 5.2 across section 5 (EN)", output)
            self.assertIn(f"{kind} numbering gap: missing 5.2 across section 5 (ZH)", output)

    def test_selected_proof_checks_numbering_on_unselected_statement_pages(self):
        name = self.result("2.2", proof=True)
        self.assertIn("statement numbering gap: missing 2.1",
                      self.validate("proofs/" + name)[1])
        self.result("2.1")
        self.assertEqual(self.validate("proofs/" + name), (0, ""))

    def test_readme_numbering_gaps_and_parity(self):
        self.pair("README.md", self.statement("Definition", "1.1") + self.statement("Definition", "1.3"))
        self.assertIn("statement numbering gap: missing 1.2", self.validate()[1])
        self.pair("README.md", self.statement("Definition", "1.1"), self.statement("Theorem", "1.1", True))
        self.assertIn("statement EN/ZH parity", self.validate("README.md")[1])
        self.pair("README.md", self.statement("Definition", "1.1"))
        self.assertEqual(self.validate(), (0, ""))

    def test_check_distribution_need_not_follow_statement_number_or_page_order(self):
        self.result("5.1", "Definition", checks=["5.3", "5.1"])
        self.result("5.2", "Theorem", checks=["5.2", "5.4"])
        self.result("5.3", "Remark", checks=[])
        self.assertEqual(self.validate(), (0, ""))
        self.assertEqual(self.validate("experiments/05-02_item.md"), (0, ""))
        self.append("experiments/05-02_item.md", self.declaration_check("5.1"))
        self.assertIn("duplicate framework check: 5.1", self.validate("experiments/05-01_item.md")[1])
        self.append("experiments/05-01_item.md", self.declaration_check("5.0"))
        self.assertIn("check numbering invalid", self.validate()[1])

    def test_page_parity_checks_items_anchors_and_missing_counterparts(self):
        name = self.result("2.1", proof=True, checks=["2.1"])
        self.assertEqual(self.validate(), (0, ""))
        self.append("proofs/02-01_item.md", '<a id="s2-1"></a>\n')
        self.assertIn("anchor EN/ZH parity", self.validate("proofs/" + name)[1])
        self.result("2.1", proof=True, checks=["2.1"])
        self.write("statements/02-01_item_ZH.md", self.statement("Proposition", "2.1", True))
        self.assertIn("statement EN/ZH parity", self.validate("statements/" + name)[1])
        self.write("experiments/02-01_item_ZH.md", "")
        self.assertIn("check EN/ZH parity", self.validate("experiments/" + name)[1])
        for folder in ("statements", "proofs", "experiments"):
            (self.framework / folder / "02-01_item_ZH.md").unlink()
            self.assertIn("page EN/ZH parity: missing counterpart", self.validate(f"{folder}/{name}")[1])

    def test_check_ownership_parity_is_per_page_not_just_section(self):
        self.result("2.1", checks=["2.1"])
        self.result("2.2", checks=["2.2"])
        for item, check_number in ((1, "2.2"), (2, "2.1")):
            self.write(f"experiments/02-0{item}_item_ZH.md", self.declaration_check(check_number))
        output = self.validate()[1]
        self.assertIn("check EN/ZH parity", output)
        self.assertNotIn("numbering gap", output)

    def test_matching_statement_required_for_proofs_and_experiments(self):
        self.result("2.1", proof=True, checks=[])
        self.assertEqual(self.validate(), (0, ""))
        (self.framework / "statements/02-01_item.md").unlink()
        self.result("2.1", slug="different")
        for folder in ("proofs", "experiments"):
            self.assertIn("missing matching statement page: 02-01_item.md",
                          self.validate(f"{folder}/02-01_item.md")[1])

    def test_required_navigation_is_language_specific_and_ignores_examples(self):
        name = self.result("2.1", proof=True, checks=[])
        self.assertEqual(self.validate(), (0, ""))
        for folder, target in (("statements", "../README.md"),
                               ("statements", "../proofs/" + name),
                               ("statements", "../experiments/" + name),
                               ("proofs", "../statements/" + name),
                               ("experiments", "../statements/" + name)):
            for replacement in ("other.md", target.replace(".md", "_ZH.md")):
                with self.subTest(folder=folder, target=target, replacement=replacement):
                    self.result("2.1", proof=True, checks=[])
                    page = f"{folder}/{name}"
                    text = (self.framework / page).read_text().replace(target, replacement)
                    # A link in fenced/inline code or an image is not navigation.
                    text += f"\n```text\n[x]({target})\n```\n\n`[x]({target})`\n![x]({target})\n"
                    self.write(page, text)
                    self.assertIn("missing required page link", self.validate(page)[1])
        self.result("2.1", proof=True, checks=[])
        self.write("statements/02-01_item_ZH.md", self.statement("Theorem", "2.1", True)
                   + "[Framework](../README.md)")
        self.assertIn("missing required page link: README_ZH.md", self.validate("statements/02-01_item_ZH.md")[1])

    def test_secondary_checks_must_link_to_the_primary_anchor(self):
        name = self.result("2.1", checks=["2.1"])
        other = self.result("2.2", checks=[])
        link = f"[Check 2.1](../experiments/{name}#c2-1)"
        self.append(f"experiments/{other}", f"### {link} · Title\n\nSummary.\n")
        self.assertEqual(self.validate(), (0, ""))
        for mention in ("Check 2.1", "### Check 2.1 · Title", f"[Check 2.1]({name})",
                        "[Check 2.1](../README.md#c2-1)", "[Check 2.1](https://example.org/#c2-1)"):
            with self.subTest(mention=mention):
                self.write(f"experiments/{other}", mention)
                self.assertRegex(self.validate(f"experiments/{other}")[1],
                                 "unlinked secondary check|secondary check link is not primary")
        self.write(f"experiments/{other}", self.declaration_check("2.1"))
        self.assertIn("duplicate framework check", self.validate()[1])

    def test_wrapped_secondary_check_mentions_and_ranges(self):
        name = self.result("2.1", checks=["2.1", "2.2"])
        self.write("README.md", f"[Check\n2.1](experiments/{name}#c2-1)")
        self.assertEqual(self.validate("README.md"), (0, ""))
        self.write("README.md", "See Check\n2.1.")
        self.assertIn("README.md:1: unlinked secondary check: Check 2.1", self.validate("README.md")[1])
        self.write("README.md", f"[Checks 2.1–2.2](experiments/{name}#c2-1)")
        self.assertIn("secondary check link is not primary: Check 2.2", self.validate("README.md")[1])
        self.write(f"experiments/{name}", f"[Statement](../statements/{name})\n"
                   f"[Check\n2.1](#c2-1)\n<a id=\"c2-1\"></a>\n## Check 2.1 · Title\n")
        self.assertNotIn("unlinked secondary check", self.validate(f"experiments/{name}")[1])
        self.append(f"experiments/{name}", "\nCheck\n2.1.\n")
        self.assertIn(f"{name}:7: unlinked secondary check", self.validate(f"experiments/{name}")[1])

    def test_check_references_and_link_ownership(self):
        name = self.result("2.1", checks=["2.1"])
        self.write("README.md", f"[Check 2.1](experiments/{name}#c2-1)\n[Theorem 2.1](statements/{name}#s2-1)")
        self.assertEqual(self.validate(), (0, ""))
        self.write("README.md", "Check 2.2. Checks 2.1–2.3.")
        output = self.validate("README.md")[1]
        self.assertIn("undefined framework check: Check 2.2", output)
        self.assertIn("undefined framework check: Check 2.3", output)
        for anchor in ("s2-1", "c2-1"):
            self.write("figures/README.md", f'<a id="{anchor}"></a>')
            self.write("README.md", f"[wrong](figures/README.md#{anchor})")
            self.assertIn("framework anchor on wrong page", self.validate("README.md")[1])
        self.write("README.md", f"[wrong](statements/{name}#c2-1)")
        self.assertIn("missing heading anchor", self.validate("README.md")[1])

    def test_reference_keys_must_exist_in_both_languages_even_outside_policy(self):
        key = "kingma2015adam"
        self.pair("references.md", f'<a id="{key}"></a>\nCitation.\n')
        self.write("tools/citing.md", f"[Citation](../references.md#{key})")
        self.assertEqual(self.validate("tools/citing.md"), (0, ""))
        for reference in ("references.md", "references_ZH.md"):
            with self.subTest(reference=reference):
                self.write(reference, "# Other\n")
                output = self.validate("tools/citing.md")[1]
                self.assertIn(f"reference key missing in {check.language(Path(reference))}: {key}", output)
                if reference == "references.md":
                    self.assertIn("missing heading anchor", output)
                self.write(reference, f'<a id="{key}"></a>')
        (self.framework / "references.md").unlink()
        self.assertIn("reference key missing in EN", self.validate("tools/citing.md")[1])
        self.assertIn("missing link target", self.validate("tools/citing.md")[1])

    def test_reference_keys_are_explicit_anchors_and_decode_fragments(self):
        self.write("tools/citing.md", "[Citation](../references_ZH.md#key%2D2025)")
        self.pair("references.md", '<a id="key-2025"></a>')
        self.assertEqual(self.validate("tools/citing.md"), (0, ""))
        for text in ("# key-2025", '`<a id="key-2025"></a>`',
                     '```html\n<a id="key-2025"></a>\n```'):
            self.write("references.md", text)
            self.assertIn("reference key missing in EN", self.validate("tools/citing.md")[1])
        self.pair("references.md", '<a id="key-2025"></a>')
        self.append("tools/citing.md", "\n[Again](../references.md#absent)\n[Again](../references.md#absent)")
        output = self.validate("tools/citing.md")[1]
        self.assertEqual(output.count("reference key missing in EN: absent"), 1)
        self.assertEqual(output.count("reference key missing in ZH: absent"), 1)

    def test_math_expression_budget_applies_outside_policy_and_in_appendix(self):
        for name, prefix in (("tools/README.md", ""), ("README.md", "## Appendix · Sources\n\n"),
                             ("statements/02_legacy.md", "")):
            with self.subTest(name=name):
                self.write(name, prefix + "$`x`$\n\n" * 601)
                errors, output = self.validate(name)
                self.assertEqual(errors, 1, output)
                self.assertIn("too many math expressions: 601 (maximum 600)", output)

    def test_section_references_and_bare_source_labels(self):
        self.write("README.md", "§1, §2–§9, §1–§3, § 9。")
        self.assertEqual(self.validate(), (0, ""))
        for text in ("§0", "§10", "§5.2", "§5.2.1", "§1–§12", "A5", "eqs. (A11)"):
            with self.subTest(text=text):
                self.write("README.md", text)
                self.assertGreater(self.validate()[0], 0)

    def test_retired_node_and_result_labels_including_links_and_zh(self):
        for label in ("N01", "N20", "N12(b)", "N19(c)", "W0", "W3", "`N12`",
                      "[N12(b)](README.md)"):
            self.write("README.md", label)
            self.assertIn("retired node label", self.validate("README.md")[1])
        for label in ("Theorem 7", "Thm. C4", "Prop. A5", "Corollary 9", "Lemma W0", "定理 3",
                      "Props. A5–A7", "Theorem C4", "Prop. 2.3", "Theorem 10.1"):
            with self.subTest(label=label):
                self.write("README_ZH.md", label)
                self.assertIn("original/nonlocal result label", self.validate("README_ZH.md")[1])

    def test_figure_provenance_code_spans_allow_literal_keys_and_ids(self):
        labels = [f"N{i:02d}" for i in range(1, 21)] + [f"W{i}" for i in range(4)]
        labels += ["C5", "E8", "V7", "T03", "C4.amplitude",
                   "V:groups[id=E8].data.{gaps,W3.per_step,W1.panels}"]
        for name in ("figures/README.md", "figures/specs/fig5_3.md"):
            for marker in ("`", "``"):
                with self.subTest(name=name, marker=marker):
                    self.write(name, " ".join(f"{marker}{label}{marker}" for label in labels))
                    self.assertEqual(self.validate(name), (0, ""))

    def test_figure_provenance_table_path_at_reported_line(self):
        row = "| `fig5_3_warmth` | `V:groups[id=E8].data.{gaps,W3.per_step,W1.panels}` |"
        for name in ("figures/README.md", "figures/specs/fig5_3.md"):
            with self.subTest(name=name):
                self.write(name, "\n" * 41 + "| Figure | Provenance |\n|---|---|\n" + row)
                self.assertEqual(self.validate(name), (0, ""))

    def test_figure_provenance_keeps_existing_prose_rules(self):
        for label in ("N01", "N20", "W0", "W3", "[N12](README.md)"):
            with self.subTest(label=label):
                self.write("figures/README.md", f"`N02 W1 C5 E8` {label}")
                errors, output = self.validate("figures/README.md")
                self.assertEqual(errors, 1, output)
                self.assertIn("retired node label", output)
        # Figure check codes already allow prose, and specs remain outside the
        # source-label policy scope. Do not expand either policy here.
        self.write("figures/README.md", "C5 E8")
        self.write("figures/specs/fig5_3.md", "N01 N20 W0 W3 C5 E8")
        self.assertEqual(self.validate("figures/README.md", "figures/specs/fig5_3.md"), (0, ""))

    def test_figure_readme_unclosed_code_span_keeps_node_policy(self):
        self.write("figures/README.md", "`N01 W3")
        errors, output = self.validate("figures/README.md")
        self.assertEqual(errors, 2, output)
        self.assertIn("retired node label: N01", output)
        self.assertIn("retired node label: W3", output)

    def test_figure_provenance_keeps_rendering_and_link_checks(self):
        for name in ("figures/README.md", "figures/specs/fig5_3.md"):
            with self.subTest(name=name):
                self.write(name, "`N01 W3 C5 E8` $bad$ [missing](missing.md)")
                errors, output = self.validate(name)
                self.assertEqual(errors, 2, output)
                self.assertIn("inline math must use", output)
                self.assertIn("missing link target", output)

    def test_main_pages_keep_node_policy_in_code(self):
        for name in ("README.md", "README_ZH.md", "proofs/N01.md", "experiments/N01.md"):
            with self.subTest(name=name):
                self.write(name, "`N01` `W3`")
                errors, output = self.validate(name)
                self.assertEqual(errors, 2, output)
                self.assertIn("retired node label: N01", output)
                self.assertIn("retired node label: W3", output)

    def test_only_exact_readme_appendices_exempt_labels(self):
        labels = "N01 N20 W0 W3 Theorem 7 Props. A5–A7 定理 3 §5.2 eqs. (A11) Theorem 6.99 Check 2.99"
        for name, title in (("README.md", "Appendix · Sources"),
                            ("README_ZH.md", "附录 · 出处")):
            self.write(name, f"## {title}\n\n{labels}\n\n### Nested\n{labels}")
            self.assertEqual(self.validate(name), (0, ""))
            self.write(name, f"## {title}\n{labels}\n## Later\nN12")
            self.assertIn("retired node label", self.validate(name)[1])
            self.write(name, f"## {title}\n$bad$ [missing](missing.md)")
            output = self.validate(name)[1]
            self.assertIn("inline math must use", output)
            self.assertIn("missing link target", output)
        for name, title in (("README.md", "Source concordance"), ("README_ZH.md", "出处对照"),
                            ("README.md", "Appendix · Correspondence with earlier labels and sources"),
                            ("README_ZH.md", "附录 · 与早期编号及出处的对照"),
                            ("proofs/02_page.md", "Appendix · Sources")):
            self.write(name, f"## {title}\nN12")
            self.assertIn("retired node label", self.validate(name)[1])

    def test_readme_check_codes_must_be_code_even_in_links_and_appendix(self):
        for name in ("README.md", "README_ZH.md"):
            for prefix in ("", "## Appendix · Sources\n"):
                for code in ("C5", "V7", "E12", "T03", "v016:C4.amplitude", "[C5](README.md)"):
                    self.write(name, prefix + code)
                    self.assertIn("check code in README prose", self.validate(name)[1])
            self.write(name, "`C5` `V7` `E12` `T03` `v016:C4.amplitude`")
            self.assertEqual(self.validate(name), (0, ""))

    def test_experiment_check_codes_need_suite_and_code_span(self):
        for text in ("`v014:C5` `v015:V7` `v015:E6` `pkg:T12` `v016:C4.amplitude` `v016:C5b`",
                     "[`v016:C4.amplitude`](../README.md)"):
            self.pair("experiments/02_page.md", text)
            self.assertEqual(self.validate(), (0, ""))
        for text in ("C5", "V7", "E12", "T03", "`C5`", "v014:C5", "`other:C5`",
                     "[C5](../README.md)", "`v015:E6, E12`", "`T03`", "`C4.amplitude`"):
            with self.subTest(text=text):
                self.pair("experiments/02_page.md", text)
                self.assertIn("unprefixed check code", self.validate()[1])

    def test_legacy_pages_and_historical_documents(self):
        self.write("proofs/N12_old.md", "**Source:** Thm. 7\n\n**(W0)**\n\nN12(b)")
        output = self.validate("proofs/N12_old.md")[1]
        self.assertIn("original/nonlocal result label", output)
        self.assertIn("retired node label", output)
        self.assertNotIn("EN/ZH parity", output)
        for name in ("docs/direction/2026-10-11_v015_progress_ZH.md", "docs/direction/v015_extension/extra.md"):
            self.write(name, "N12 W0 Theorem 7 Theorem 6.99 Check 6.99 C5 §5.2")
            self.assertEqual(self.validate(name), (0, ""))
            self.write(name, "N12 W0 $bad$ [link](missing.md)")
            errors, output = self.validate(name)
            self.assertEqual(errors, 2, output)
            self.assertIn("inline math must use", output)
            self.assertIn("missing link target", output)


class PathSelectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.base = Path(temporary.name)
        self.root = self.base / "repository"
        for name in (
            "theoretical_framework/README.md",
            "theoretical_framework/README_ZH.md",
            "theoretical_framework/proofs/N01.md",
            "theoretical_framework/experiments/N01.md",
            "theoretical_framework/figures/README.md",
            "theoretical_framework/figures/specs/N01.md",
            "theoretical_framework/tools/README.md",
            "docs/direction/2026-10-11_v015_progress_ZH.md",
            "docs/direction/v015_extension/page.md",
        ):
            self.write(name)
        self.extra = self.write("docs/direction/v015_revision/中文 page.md")
        root_patch = patch.object(check, "ROOT", self.root)
        root_patch.start()
        self.addCleanup(root_patch.stop)
        for obj, name in ((check, "urlopen"), (check, "credential"), (check.subprocess, "run")):
            guard = patch.object(obj, name, side_effect=AssertionError("live I/O attempted"))
            guard.start()
            self.addCleanup(guard.stop)

    def write(self, name, text="# Title\n"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def run_main(self, args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            status = check.main(args)
        return status, out.getvalue()

    def test_explicit_selection_is_shared_by_all_modes(self):
        # An unselected default file has bad rendering but remains a valid link target.
        self.write("theoretical_framework/README.md", "# Title\n\n$bad$\n")
        self.extra.write_text("[target](../../../theoretical_framework/README.md#title)\n",
                              encoding="utf-8")
        second = self.write("notes/second.md")
        for flags in ([], ["--online"], ["--blob"], ["--online", "--blob"]):
            with self.subTest(flags=flags), contextlib.chdir(self.base), \
                    patch.object(check, "offline", wraps=check.offline) as offline, \
                    patch.object(check, "online") as online, patch.object(check, "blob") as blob:
                status, output = self.run_main([
                    *flags, str(self.extra.relative_to(self.root)), str(second)])
                self.assertEqual((status, output.splitlines()[-1].split(";")[0]),
                                 (0, "PASS: 2 files"))
                self.assertEqual(offline.call_args.args[0], [self.extra, second])
                for flag, run in (("--online", online), ("--blob", blob)):
                    if flag in flags:
                        run.assert_called_once()
                        self.assertEqual(run.call_args.args[0], [self.extra, second])
                    else:
                        run.assert_not_called()

    def test_no_paths_keeps_default_scope_in_all_modes(self):
        files, s1 = check.scope(self.root)
        self.extra.write_text("$bad$\n", encoding="utf-8")
        for flags in ([], ["--online"], ["--blob"], ["--online", "--blob"]):
            with self.subTest(flags=flags), \
                    patch.object(check, "offline", wraps=check.offline) as offline, \
                    patch.object(check, "online") as online, patch.object(check, "blob") as blob:
                status, output = self.run_main(flags)
                self.assertEqual(status, 0, output)
                self.assertIn(f"PASS: {len(files)} files; 0 errors", output)
                self.assertEqual(offline.call_args.args[:2], (files, s1))
                for flag, run in (("--online", online), ("--blob", blob)):
                    if flag in flags:
                        self.assertEqual(run.call_args.args[0], files)
                    else:
                        run.assert_not_called()

    def test_selected_s1_files_keep_label_and_check_id_rules(self):
        _, s1 = check.scope(self.root)
        for path in sorted(s1):
            with self.subTest(path=path):
                path.write_text("Thm. 7, (32), C23.\n", encoding="utf-8")
                normalized = path.parent / ".." / path.parent.name / path.name
                status, output = self.run_main([str(normalized)])
                self.assertEqual(status, 1)
                self.assertIn("original/nonlocal result label", output)
                self.assertIn("nonlocal equation/number label", output)
                if check.is_readme(path):
                    self.assertIn("check code in README prose", output)
                elif path.parent.name == "experiments":
                    self.assertIn("unprefixed check code", output)

    def test_other_selected_files_get_rendering_and_link_checks_only(self):
        for path in (self.extra, self.root / "theoretical_framework/figures/specs/N01.md",
                     self.root / "theoretical_framework/tools/README.md",
                     self.root / "docs/direction/2026-10-11_v015_progress_ZH.md",
                     self.root / "docs/direction/v015_extension/page.md"):
            with self.subTest(path=path):
                path.write_text("Thm. 7, (32), C23.\n", encoding="utf-8")
                self.assertEqual(self.run_main([str(path)])[0], 0)
                path.write_text("Thm. 7, (32), C23. $bad$ [link](missing.md)\n",
                                encoding="utf-8")
                status, output = self.run_main([str(path)])
                self.assertEqual(status, 1)
                self.assertIn("inline math must use", output)
                self.assertIn("missing link target", output)
                self.assertNotIn("nonlocal", output)
                self.assertNotIn("unlinked", output)

    def test_duplicate_and_equivalent_paths_are_checked_once(self):
        status, output = self.run_main([
            str(self.extra), str(self.extra.relative_to(self.root)),
            str(self.extra.parent / ".." / self.extra.parent.name / self.extra.name)])
        self.assertEqual(status, 0, output)
        self.assertIn("PASS: 1 files; 0 errors", output)

    def test_symlink_to_s1_file_keeps_s1_rules(self):
        target = self.write("theoretical_framework/README.md", "Thm. 7, C23.\n")
        alias = self.root / "alias.md"
        alias.symlink_to(target)
        status, output = self.run_main([str(alias)])
        self.assertEqual(status, 1)
        self.assertIn("original/nonlocal result label", output)
        self.assertIn("check code in README prose", output)

    def test_invalid_paths_fail_before_checks_in_all_modes(self):
        outside = self.base / "outside.md"
        outside.write_text("# Outside\n", encoding="utf-8")
        self.write("notes/plain.txt")
        (self.root / "directory.md").mkdir()
        (self.root / "escape.md").symlink_to(outside)
        (self.root / "escape-dir").symlink_to(self.base, target_is_directory=True)
        (self.root / "dangling.md").symlink_to(self.root / "missing.md")
        cases = [
            (str(outside), "outside the repository"),
            ("../outside.md", "outside the repository"),
            ("escape.md", "outside the repository"),
            ("escape-dir/outside.md", "outside the repository"),
            ("missing.md", "not an existing file"),
            ("dangling.md", "not an existing file"),
            ("directory.md", "not an existing file"),
            ("notes/plain.txt", "not a .md file"),
        ]
        for flags in ([], ["--online"], ["--blob"], ["--online", "--blob"]):
            for path, message in cases:
                with self.subTest(flags=flags, path=path), \
                        patch.object(check, "offline") as offline, \
                        patch.object(check, "online") as online, patch.object(check, "blob") as blob:
                    err = io.StringIO()
                    with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as error:
                        self.run_main([*flags, str(self.extra), path])
                    self.assertEqual(error.exception.code, 2)
                    self.assertIn(message, err.getvalue())
                    for run in (offline, online, blob):
                        run.assert_not_called()


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
