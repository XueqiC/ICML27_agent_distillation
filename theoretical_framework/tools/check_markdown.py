#!/usr/bin/env python3
"""Validate the scoped theory Markdown, offline by default (Python 3.11, stdlib)."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import html
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import unicodedata
from typing import NamedTuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
DENIED = re.compile(
    r"\\(?:operatorname|boldsymbol|tag|label|eqref|ref|require|href|url|"
    r"newcommand|renewcommand|def|class|style|cssId|html|unicode|bbox)(?![A-Za-z])"
)
FENCE = re.compile(r"^\s*(`{3,}|~{3,})([^\n]*)$")
DOLLAR = re.compile(r"(?<!\\)(?:\\\\)*\$")
STATEMENT_WORDS = ("Definition", "Assumption", "Theorem", "Proposition", "Lemma",
                   "Corollary", "Remark", "Example", "Algorithm")
WORD_TYPES = {form: word for word in STATEMENT_WORDS
              for form in (word, "Corollaries" if word == "Corollary" else word + "s")}
WORDS = "|".join(sorted(WORD_TYPES, key=len, reverse=True))
NUMBER = r"[1-9]\.\d+(?!\d|\.\d|[A-Za-z_])"
RESULT = re.compile(
    rf"(?<![A-Za-z0-9_])(?P<word>{WORDS}|Props?\.?|Thms?\.?|Cor\.?|Def\.?|Ass\.?|"
    r"定理|命题|引理|推论|定义|假设)\s*"
    r"(?P<number>[A-Z]?\d+(?:\.\d+)*(?:[abc′″])?|[ABC])(?![A-Za-z0-9_])"
)
CONTINUATION = re.compile(
    rf"\s*(?:\([a-g]\)|（[a-g]）)?\s*(?P<join>[-–—]|,\s*(?:and\s+)?|and\b|or\b|和|与|、|及|至)"
    rf"\s*(?P<number>{NUMBER})"
)
STATEMENT = re.compile(
    rf"^\*\*(?P<word>{'|'.join(STATEMENT_WORDS)})\s+(?P<number>{NUMBER})\s*"
    r"(?:\(.+\)|（.+）)\.\*\*"
)
CHECK = re.compile(rf"^## Check\s+(?P<number>{NUMBER})\s+·\s+\S")
CHECK_REF = re.compile(rf"(?<![A-Za-z0-9_])Checks?\s+(?P<number>{NUMBER})")
CODE_ID = re.compile(r"(?<![A-Za-z0-9_])[CVET]\d+[a-z]?(?:\.[A-Za-z][A-Za-z0-9_-]*)?(?![A-Za-z0-9_])")
NODE = re.compile(r"(?<![A-Za-z0-9_])(?:N(?:0[1-9]|1\d|20)|W[0-3])(?:\([a-z]\))?(?![A-Za-z0-9_])")
APPENDICES = {"Appendix · Correspondence with earlier labels and sources",
              "附录 · 与早期编号及出处的对照"}
EQ_REF = re.compile(r"[（(]([AB]?\d+[a-z]?)[)）]")
DISPLAY = re.compile(r"\\[\[\]]|\\begin\{(?:equation\*?|align\*?|gather\*?|multline\*?|displaymath)\}")


def scope(root: Path = ROOT) -> tuple[list[Path], set[Path]]:
    framework = root / "theoretical_framework"
    s1 = [framework / "README.md", framework / "README_ZH.md"]
    s1 += sorted((framework / "proofs").glob("*.md"))
    s1 += sorted((framework / "experiments").glob("*.md"))
    s1 += [framework / "figures/README.md"]
    files = sorted(framework.rglob("*.md"))
    files += [root / "docs/direction/2026-10-11_v015_progress_ZH.md"]
    files += sorted((root / "docs/direction/v015_extension").glob("*.md"))
    # New pages (notably figure specs and tool docs) get rendering/link checks,
    # without extending the source-label and check-ID policy to them.
    return files, set(s1)


def select_files(paths: list[Path], root: Path = ROOT) -> list[Path]:
    """Resolve explicit repository paths before validation or rendering."""
    root = root.resolve()
    files = []
    for path in paths:
        try:
            resolved = (root / path).resolve()
        except (OSError, RuntimeError, ValueError):
            raise ValueError(f"cannot resolve Markdown path: {path}") from None
        if not resolved.is_relative_to(root):
            raise ValueError(f"path is outside the repository: {path}")
        if not resolved.is_file():
            raise ValueError(f"path is not an existing file: {path}")
        if path.suffix != ".md" or resolved.suffix != ".md":
            raise ValueError(f"path is not a .md file: {path}")
        if resolved not in files:
            files.append(resolved)
    return files


def inline_text(text: str) -> str:
    """Text that contributes to a heading slug, with inline markup removed."""
    text = re.sub(r"!?\[((?:[^\[\]]|\[[^\]]*\])*)\]\([^\n]*?\)", r"\1", text)
    text = re.sub(r"<[^>]*>", "", text)
    # Emphasis/code delimiters do not contribute, but literal connector punctuation does.
    text = text.replace("`", "").replace("*", "")
    text = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"\1", text)
    return html.unescape(text)


def github_slug(text: str) -> str:
    text = inline_text(text).lower()
    # Match html-pipeline's Ruby Word categories, plus literal space and hyphen.
    return "".join(c for c in text if unicodedata.category(c)[0] in "LM"
                   or unicodedata.category(c) in {"Nd", "Pc"} or c in " -").replace(" ", "-")


LIST_ITEM = re.compile(r"^( *)([-+*]|\d{1,9}[.)])(?: +|$)")
QUOTE = re.compile(r"^ {0,3}> ?")


def block_lines(text: str):
    """Yield line, fence and container context, without parsing fenced contents.

    List continuations use the item's content column, including numbered and
    nested markers. Blank lines retain that context; a dedented block ends it.
    Blockquotes have their own context and may contain top-level math fences.
    """
    marker = None
    language = ""
    lists: list[int] = []
    quote_depth = 0
    blank = True
    for n, source in enumerate(text.splitlines(), 1):
        line = source.expandtabs(4)
        source_indent = len(line) - len(line.lstrip())
        depth = 0
        while q := QUOTE.match(line):
            line = line[q.end():]
            depth += 1
        if depth != quote_depth:
            lists.clear()
            quote_depth = depth
        m = FENCE.match(line)
        if marker:
            yield n, line, language or "code", False, bool(lists), max(source_indent, len(line) - len(line.lstrip()))
            if m and m[1][0] == marker[0] and len(m[1]) >= len(marker) and not m[2].strip():
                marker = None
                language = ""
            continue
        indent = len(line) - len(line.lstrip())
        item = LIST_ITEM.match(line)
        if line.strip():
            # Lazy prose can continue a list paragraph without indentation,
            # but a fence (or a block following a blank) cannot do so.
            if item or m or blank or re.match(r"^ {0,3}(?:#{1,6} |>|---+$)", line):
                while lists and indent < lists[-1]:
                    lists.pop()
            if item:
                rest = line
                offset = 0
                while True:
                    item = LIST_ITEM.match(rest)
                    q = QUOTE.match(rest)
                    if not item and not q:
                        break
                    container = item or q
                    offset += container.end()
                    if item:
                        lists.append(offset)
                    rest = rest[container.end():]
                # A fence may begin on the list marker's own line.
                if FENCE.match(rest):
                    line = rest
                    m = FENCE.match(line)
        blank = not line.strip()
        if m:
            marker = m[1]
            language = m[2].strip()
            yield n, line, language or "code", True, bool(lists), max(source_indent, indent)
        else:
            yield n, line, None, False, bool(lists), indent


def lines_outside_fences(text: str):
    """Yield (line number, line, enclosing fence language); fences are included."""
    for n, line, language, _, _, _ in block_lines(text):
        yield n, line, language


def headings(text: str) -> dict[str, int]:
    """GitHub heading slugs and explicit anchors used to preserve old links."""
    result = {}
    previous = None
    for n, line, fence in lines_outside_fences(text):
        if fence:
            previous = None
            continue
        m = re.match(r"^ {0,3}#{1,6}\s+(.+?)(?:\s+#+\s*)?$", line)
        heading = m[1] if m else None
        if previous and re.fullmatch(r" {0,3}(?:=+|-+)\s*", line):
            heading = previous[1]
            n = previous[0]
        if heading is not None:
            base = github_slug(heading)
            slug = base
            counter = 0
            while slug in result:
                counter += 1
                slug = f"{base}-{counter}"
            result[slug] = n
        previous = (n, line.strip()) if line.strip() else None
    for n, line, fence in lines_outside_fences(text):
        if not fence:
            for anchor in re.finditer(r'<a\s+(?:id|name)=["\']([^"\']+)["\']\s*>', line):
                result.setdefault(html.unescape(anchor[1]), n)
    return result


def inline_tokens(line: str):
    """Yield (kind, start, end, body, backtick) without losing source columns.

    Recognize math before code: the backticks in $`x`$ protect the formula,
    whereas dollars inside an ordinary code span are literal examples.
    """
    pos = 0
    while pos < len(line):
        if line[pos] == "\\":
            pos += 2
            continue
        if line[pos] == "`":
            marker = re.match(r"`+", line[pos:])[0]
            close = re.search(r"(?<!`)" + re.escape(marker) + r"(?!`)", line[pos + len(marker):])
            if close:
                end = pos + len(marker) + close.end()
                yield "code", pos, end, "", False
                pos = end
                continue
            pos += len(marker)
            continue
        if line[pos] != "$":
            pos += 1
            continue
        backtick = line.startswith("$`", pos)
        if backtick:
            close = line.find("`$", pos + 2)
            end = close + 2
            body_start = pos + 2
        else:
            match = DOLLAR.search(line, pos + 1)
            close = match.end() - 1 if match else -1
            end = close + 1
            body_start = pos + 1
        if close < 0:
            yield "dollar", pos, pos + 1, "", backtick
            pos += 1
        else:
            yield "math", pos, end, line[body_start:close], backtick
            pos = end


def mask_inline(line: str, kinds: set[str]) -> str:
    for kind, start, end, _, _ in reversed(list(inline_tokens(line))):
        if kind in kinds:
            line = line[:start] + " " * (end - start) + line[end:]
    return line


def bracket_labels(line: str):
    """Balanced labels, with brackets in code/math excluded from nesting."""
    plain = mask_inline(line, {"code", "math"})
    stack = []
    pos = 0
    while pos < len(plain):
        if plain[pos] == "\\":
            pos += 2
            continue
        if plain[pos] == "[":
            stack.append(pos)
        elif plain[pos] == "]" and stack:
            start = stack.pop()
            if not stack:
                yield line[start + 1:pos], start, pos + 1
        pos += 1


def markdown_links(line: str):
    """Inline links/images with balanced parentheses; reference definitions too."""
    line = mask_inline(line, {"code"})
    for label, label_start, label_end in bracket_labels(line):
        if line[label_end:label_end + 1] != "(":
            continue
        start = label_end + 1
        pos, depth = start, 1
        while pos < len(line) and depth:
            if line[pos] == "\\":
                pos += 2
                continue
            if line[pos] == "(":
                depth += 1
            elif line[pos] == ")":
                depth -= 1
            pos += 1
        if not depth:
            dest = line[start:pos - 1].strip()
            if dest.startswith("<") and ">" in dest:
                dest = dest[1:dest.index(">")]
            else:
                dest = re.split(r'\s+["\']', dest, maxsplit=1)[0]
            yield label, dest, label_start, pos
    ref = re.match(r"^\s*\[[^\]]+\]:\s*<?([^\s>]+)>?", line)
    if ref:
        yield "reference definition", ref[1], 0, len(line)


def mask_links(line: str) -> str:
    for _, _, start, end in reversed(list(markdown_links(line))):
        line = line[:start] + " " * (end - start) + line[end:]
    return line


def prose_only(line: str) -> str:
    line = mask_inline(line, {"code", "math"})
    line = mask_links(line)
    return line


def is_readme(path: Path) -> bool:
    return path.parent.name == "theoretical_framework" and path.name in {"README.md", "README_ZH.md"}


def is_figure_provenance(path: Path) -> bool:
    """Figure provenance documents use literal JSON keys and IDs in code spans."""
    return (path.parts[-3:] == ("theoretical_framework", "figures", "README.md")
            or path.suffix == ".md"
            and path.parts[-4:-1] == ("theoretical_framework", "figures", "specs"))


def language(path: Path) -> str:
    return "ZH" if path.stem.endswith("_ZH") else "EN"


def policy_lines(path: Path, text: str):
    """The exact README appendix exempts labels, never rendering or links."""
    appendix = False
    for n, line, fence in lines_outside_fences(text):
        if not fence and (m := re.match(r"^ {0,3}(#{1,2})\s+(.+?)(?:\s+#+)?\s*$", line)):
            appendix = is_readme(path) and m[1] == "##" and m[2] in APPENDICES
        yield n, line, fence, appendix


def policy_text(line: str, *, keep_code: bool = False) -> str:
    """Visible labels count as prose; URLs, HTML attributes and math do not."""
    for label, _, start, end in reversed(list(markdown_links(line))):
        replacement = "" if label == "reference definition" else label
        line = line[:start] + replacement + line[end:]
    line = mask_inline(line, {"math"} if keep_code else {"code", "math"})
    return re.sub(r"<[^>]*>", "", line).replace("*", "")


def reference_paragraphs(path: Path, text: str):
    """Join wrapped prose, without letting references cross fenced examples."""
    paragraph: list[str] = []
    first = 1
    for n, line, fence, appendix in policy_lines(path, text):
        prose = "" if fence or appendix else policy_text(line)
        if not prose.strip():
            if paragraph:
                yield first, "\n".join(paragraph)
                paragraph.clear()
        else:
            if not paragraph:
                first = n
            paragraph.append(prose)
    if paragraph:
        yield first, "\n".join(paragraph)


class Declaration(NamedTuple):
    path: Path
    line: int
    word: str
    number: str
    anchored: bool


class FrameworkIndex:
    """Index the entire policy scope, even when validating selected files only.

    Read each page once so declarations and its validation use the same snapshot
    while section writers work. Missing pages remain ordinary link/parity errors.
    """
    def __init__(self, texts: dict[Path, str], s1: set[Path]):
        self.pages: dict[Path, list[Declaration]] = defaultdict(list)
        self.statements: dict[str, list[Declaration]] = defaultdict(list)
        self.checks: dict[str, list[Declaration]] = defaultdict(list)
        for path in sorted(s1 & texts.keys()):
            if path.parent.name in {"proofs", "experiments"} and re.match(r"N\d{2}(?:_|\.)", path.name):
                # Old local numbering must not supply or shadow v0.16 statements.
                continue
            previous: list[tuple[str, bool]] = []
            raw_lines = texts[path].splitlines()
            for n, line, fence, appendix in policy_lines(path, texts[path]):
                # block_lines removes quote markers for rendering, but quoted
                # examples are not declarations at the start of a source line.
                raw = raw_lines[n - 1]
                if not fence and not appendix:
                    statement, check = STATEMENT.match(raw), CHECK.match(raw)
                    if statement or check:
                        match = statement or check
                        number = match["number"]
                        word = statement["word"] if statement else "Check"
                        anchor = ("s" if statement else "c") + number.replace(".", "-")
                        anchored = any(eligible and re.fullmatch(
                            rf'<a\s+id=["\']{anchor}["\']\s*></a>', prior.strip())
                            for prior, eligible in previous[-3:])
                        declaration = Declaration(path, n, word, number, anchored)
                        self.pages[path].append(declaration)
                        if statement and (is_readme(path) or path.parent.name == "proofs"):
                            self.statements[number].append(declaration)
                        elif check and path.parent.name == "experiments":
                            self.checks[number].append(declaration)
                if raw.strip():
                    previous.append((raw, not fence and not appendix))

    def validate_page(self, path: Path, report: Report):
        counters: dict[str, int] = defaultdict(int)
        for declaration in self.pages[path]:
            _, n, word, number, anchored = declaration
            kind = "check" if word == "Check" else "statement"
            section, item = map(int, number.split("."))
            page_section = re.match(r"^0([2-9])_", path.name)
            correct_page = (
                is_readme(path) and section == 1 and kind == "statement"
                or path.parent.name == ("experiments" if kind == "check" else "proofs")
                and page_section and int(page_section[1]) == section
            )
            if not correct_page:
                report.hit(path, n, f"{kind} declared in wrong section/page: {word} {number}")
            if not anchored:
                anchor = ("c" if kind == "check" else "s") + number.replace(".", "-")
                report.hit(path, n, f"missing adjacent {kind} anchor: {anchor} (previous 3 non-blank lines)")
            expected = counters[kind] + 1
            if item != expected:
                reason = "gap" if item > expected else "out-of-order declaration"
                report.hit(path, n, f"{kind} numbering {reason}: expected {section}.{expected}, got {number}")
            counters[kind] = item
            registry = self.checks if kind == "check" else self.statements
            same_language = [d for d in registry[number] if language(d.path) == language(path)]
            if len(same_language) > 1:
                locations = ", ".join(f"{d.path.name}:{d.line}" for d in same_language)
                report.hit(path, n, f"duplicate framework {kind}: {number} ({language(path)}; {locations})")

    def validate_parity(self, files: list[Path], texts: dict[Path, str], report: Report):
        visited = set()
        selected = set(files)
        for path in sorted(selected):
            if path.parent.name not in {"proofs", "experiments"}:
                continue
            # Legacy N-pages still get rendering, links and label checks, but do
            # not impose the new page-pair structure while they are being removed.
            if not re.match(r"^0[2-9]_", path.name):
                continue
            en = path.with_name(path.stem.removesuffix("_ZH") + ".md")
            zh = en.with_name(en.stem + "_ZH.md")
            if en in visited:
                continue
            visited.add(en)
            kind = "statement" if path.parent.name == "proofs" else "check"
            if en not in texts or zh not in texts:
                missing = en if en not in texts else zh
                report.hit(path, 1, f"{kind} EN/ZH parity: missing counterpart {missing.name}")
                continue
            def pairs(p):
                return {(d.word, d.number) for d in self.pages[p]
                        if (d.word == "Check") == (kind == "check")}
            en_pairs, zh_pairs = pairs(en), pairs(zh)
            if en_pairs != zh_pairs:
                report.hit(path, 1, f"{kind} EN/ZH parity: EN only {sorted(en_pairs - zh_pairs)}; "
                           f"ZH only {sorted(zh_pairs - en_pairs)}")

    def references(self, path: Path, n: int, prose: str, report: Report):
        def verify(word, number, line):
            if word == "Check":
                if not self.checks.get(number):
                    report.hit(path, line, f"undefined framework check: Check {number}")
            elif not self.statements.get(number):
                report.hit(path, line, f"undefined framework statement: {word} {number}")
            elif word not in {d.word for d in self.statements[number]}:
                declared = ", ".join(sorted({d.word for d in self.statements[number]}))
                report.hit(path, line, f"statement type mismatch: {word} {number} (declared {declared})")

        for pattern in (RESULT, CHECK_REF):
            for match in pattern.finditer(prose):
                line = n + prose.count("\n", 0, match.start())
                word = match["word"] if pattern is RESULT else "Check"
                number = match["number"]
                if WORD_TYPES.get(word) == "Theorem" and number in {"A", "B", "C"}:
                    continue
                if word != "Check" and (word not in WORD_TYPES or not re.fullmatch(NUMBER, number)):
                    label = " ".join(match[0].split())
                    report.hit(path, line, f"original/nonlocal result label: {label}")
                    continue
                word = WORD_TYPES.get(word, word)
                verify(word, number, line)
                end = match.end()
                while continuation := CONTINUATION.match(prose, end):
                    line = n + prose.count("\n", 0, continuation.start("number"))
                    following = continuation["number"]
                    if continuation["join"] in {"-", "–", "—", "至"} and number.split(".")[0] != following.split(".")[0]:
                        report.hit(path, line, f"framework range crosses sections: {number}–{following}")
                    verify(word, following, line)
                    number, end = following, continuation.end()


class Report:
    def __init__(self):
        self.errors = 0
        self.counts: Counter[str] = Counter()

    def hit(self, path: Path, line: int, message: str):
        try:
            name = path.relative_to(ROOT)
        except ValueError:
            name = path
        print(f"{name}:{line}: {message}")
        self.errors += 1
        rule = message.split(":", 1)[0].split(" at column ", 1)[0]
        if rule.startswith("denylisted macro "):
            rule = "denylisted macro"
        self.counts[rule] += 1


def offline(files: list[Path], s1: set[Path], report: Report):
    anchor_cache: dict[Path, dict[str, int]] = {}
    texts = {}
    for path in sorted(set(files) | s1):
        try:
            texts[path] = path.read_text(encoding="utf-8")
        except (FileNotFoundError, IsADirectoryError):
            # Old pages can disappear during the concurrent restructuring.
            continue
    framework = FrameworkIndex(texts, s1)
    for path in files:
        if path not in texts:
            report.hit(path, 1, "missing scoped file")
            continue
        text = texts[path]
        if path in s1:
            framework.validate_page(path, report)
            for n, paragraph in reference_paragraphs(path, text):
                framework.references(path, n, paragraph, report)
        lines = [line for _, line, _ in lines_outside_fences(text)]
        for n, line, language, opening, in_list, indent in block_lines(text):
            if language == "math" and FENCE.match(line) and (in_list or indent):
                if opening or indent:
                    report.hit(path, n, "math-fence-in-list: math fences must start at column 0 outside lists")
        visible_numbers = set(re.findall(r"\\qquad\(([^)]+)\)", "\n".join(
            l for _, l, f in lines_outside_fences(text) if f == "math")))
        heading_lines = set(headings(text).values())
        reference_labels = {" ".join(m[1].split()).casefold()
                            for _, l, f in lines_outside_fences(text) if not f
                            if (m := re.match(r"^ {0,3}\[([^\]]+)\]:", l))}
        for n, line, fence, appendix in policy_lines(path, text):
            for m in re.finditer(r"\$\$", line):
                report.hit(path, n, f"display dollars at column {m.start() + 1}")
            for m in DENIED.finditer(line):
                report.hit(path, n, f"denylisted macro {m[0]}")
            fm = FENCE.match(line)
            if fm and fm[2].strip() == "math":
                if n > 1 and lines[n - 2].strip():
                    report.hit(path, n, "math fence needs a blank line before it")
            if fm and fence == "math" and not fm[2].strip():
                if n < len(lines) and lines[n].strip():
                    report.hit(path, n, "math fence needs a blank line after it")
            if fence:
                continue
            if DISPLAY.search(line) or re.search(r"\\[A-Za-z]+", prose_only(line)):
                report.hit(path, n, "display math outside a math fence")
            tokens = list(inline_tokens(line))
            plain = mask_inline(line, {"code"})
            alternate_math = re.search(r"\\\(.+?\\\)", plain)
            if alternate_math:
                report.hit(path, n, "inline math must use $`...`$")
            if n in heading_lines and (alternate_math or any(t[0] in {"math", "dollar"} for t in tokens)):
                report.hit(path, n, "math in heading")
            for kind, start, end, math, backtick in tokens:
                if kind == "code":
                    continue
                if kind == "dollar":
                    report.hit(path, n, f"unbalanced inline dollars at column {start + 1}")
                    continue
                if not backtick:
                    report.hit(path, n, f"inline math must use $`...`$ at column {start + 1}")
                if math != math.strip():
                    report.hit(path, n, "space at an inline math boundary")
                if line.lstrip().startswith("|") and "|" in math:
                    report.hit(path, n, "raw pipe in table math")
                if re.search(r"\\[{}#%]", math):
                    report.hit(path, n, "unsafe inline math escape")
            for label, start, end in bracket_labels(line):
                # Include inline, full/collapsed reference, and defined shortcut links.
                is_link = (line[end:end + 1] in {"(", "["}
                           or " ".join(label.split()).casefold() in reference_labels)
                if is_link and (any(t[0] in {"math", "dollar"} for t in inline_tokens(label))
                                or re.search(r"\\\(.+?\\\)", mask_inline(label, {"code"}))):
                    report.hit(path, n, "math in link text")
            for label, dest, _, _ in markdown_links(line):
                u = urlsplit(html.unescape(dest))
                if u.scheme or u.netloc or dest.startswith("/"):
                    continue
                target = (path.parent / unquote(u.path)).resolve() if u.path else path.resolve()
                if not target.exists():
                    report.hit(path, n, f"missing link target: {dest}")
                    continue
                if u.fragment:
                    if target not in anchor_cache:
                        target_text = texts.get(target)
                        if target_text is None and target.suffix.lower() == ".md":
                            try:
                                target_text = target.read_text(encoding="utf-8")
                            except FileNotFoundError:
                                report.hit(path, n, f"missing link target: {dest}")
                                continue
                        anchor_cache[target] = headings(target_text) if target_text is not None else {}
                    fragment = unquote(u.fragment)
                    if fragment not in anchor_cache[target]:
                        report.hit(path, n, f"missing heading anchor: {dest}")
                    elif path in s1 and (m := re.fullmatch(r"([sc])([1-9])-([0-9]+)", fragment)):
                        registry = framework.statements if m[1] == "s" else framework.checks
                        if not any(d.path == target for d in registry.get(f"{m[2]}.{m[3]}", [])):
                            report.hit(path, n, f"framework anchor on wrong page: {dest}")
            if path not in s1:
                continue
            prose = policy_text(line)
            # Code IDs never belong in README prose, including its appendix.
            if is_readme(path):
                for m in CODE_ID.finditer(prose):
                    report.hit(path, n, f"check code in README prose: {m[0]}")
            elif path.parent.name == "experiments":
                for m in CODE_ID.finditer(prose):
                    report.hit(path, n, f"unprefixed check code: {m[0]} (use a suite-prefixed code span)")
                for kind, start, end, _, _ in tokens:
                    if kind != "code":
                        continue
                    code = line[start:end].strip("`")
                    for m in CODE_ID.finditer(code):
                        if not re.search(r"(?<![A-Za-z0-9_])(?:v\d+|pkg):$", code[:m.start()]):
                            report.hit(path, n, f"unprefixed check code: {m[0]} (missing suite prefix)")
            if appendix:
                continue
            for m in NODE.finditer(policy_text(line, keep_code=not is_figure_provenance(path))):
                report.hit(path, n, f"retired node label: {m[0]}")
            for m in EQ_REF.finditer(prose):
                k = m[1]
                if k in visible_numbers or (k.isdigit() and int(k) >= 1900):
                    continue
                # Numbered steps/equivalent conditions refer to local enumerated clauses.
                if k in {"1", "2", "3"} and re.search(r"^(?:" + k + r"\. |\*\*" + k + r"\.\*\* )", text, re.M):
                    continue
                if re.search(r"(?:Optimizers|优化器)\s*$", prose[:m.start()]):
                    continue
                report.hit(path, n, f"nonlocal equation/number label: {m[0]}")
            for m in re.finditer(r"(?:eqs?\.\s*|式\s*)[（(](A?\d+[a-z]?)[)）]", prose):
                if m[1] not in visible_numbers:
                    report.hit(path, n, f"nonlocal equation citation: {m[0]}")
            for m in re.finditer(r"§+\s*\d+(?:\.\d+)*|(?<![A-Za-z0-9_])A\d+[′″]?(?![A-Za-z0-9_])", prose):
                if not re.fullmatch(r"§\s*[1-9]", m[0]):
                    report.hit(path, n, f"bare source label: {m[0]}")
        state = list(lines_outside_fences(text))
        if state and state[-1][2] and not (FENCE.match(state[-1][1]) and not FENCE.match(state[-1][1])[2].strip()):
            report.hit(path, len(lines), "unterminated fenced block")
    framework.validate_parity([path for path in files if path in s1], texts, report)


class RenderedHTML(HTMLParser):
    """Collect prose, emphasis outside math, and markup inside math renderers."""
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, bool]] = []
        self.text: list[str] = []
        self.emphasis: list[str] = []
        self.em_stack: list[tuple[int, list[str]]] = []
        self.math_stack: list[tuple[int, set[str]]] = []
        self.math_markup: list[set[str]] = []
        self.raw_math_blocks = 0
        self.anchors: set[str] = set()

    def handle_starttag(self, tag, attrs):
        attr = dict(attrs)
        if tag == "pre" and attr.get("lang", "").lower() == "math":
            self.raw_math_blocks += 1
        anchor = attr.get("id", "")
        if anchor.startswith("user-content-"):
            self.anchors.add(anchor.removeprefix("user-content-"))
        parent_hidden = self.stack[-1][1] if self.stack else False
        hidden = parent_hidden or tag in {"math-renderer", "pre", "code", "svg"} or "mermaid" in attr.get("class", "").split()
        if tag not in {"br", "hr", "img", "input", "meta", "link", "wbr", "source", "area", "base", "embed", "param", "track"}:
            self.stack.append((tag, hidden))
        if tag in {"em", "strong", "a", "code"}:
            for _, tags in self.math_stack:
                tags.add(tag)
        if tag == "math-renderer":
            self.math_stack.append((len(self.stack), set()))
        elif tag in {"em", "strong"} and not self.math_stack:
            self.em_stack.append((len(self.stack), []))
        if not hidden and tag in {"p", "li", "td", "th", "div", "span", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self.text.append("\n")

    def handle_endtag(self, tag):
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break
        while self.em_stack and self.em_stack[-1][0] > len(self.stack):
            self.emphasis.append("".join(self.em_stack.pop()[1]))
        while self.math_stack and self.math_stack[-1][0] > len(self.stack):
            tags = self.math_stack.pop()[1]
            if tags:
                self.math_markup.append(tags)

    def handle_data(self, data):
        # Emphasis uses rendered text, including code, but excludes all math
        # descendants. Attribute values (including link hrefs) never enter it.
        if not self.math_stack:
            for _, parts in self.em_stack:
                parts.append(data)
        if self.stack and self.stack[-1][1]:
            return
        self.text.append(data)


def credential() -> str:
    # No shell, command-line token, persisted credential, or raw subprocess exception.
    try:
        env = dict(os.environ)
        env["GIT_TERMINAL_PROMPT"] = "0"
        result = subprocess.run(["git", "credential", "fill"],
            input="protocol=https\nhost=github.com\n\n", text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False, timeout=30, env=env)
    except (OSError, subprocess.TimeoutExpired):
        raise RuntimeError("Git credential lookup failed") from None
    fields = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    token = fields.get("password")
    if result.returncode or not token or "\n" in token or "\r" in token:
        raise RuntimeError("No GitHub token returned by git credential fill")
    return token


def retry_delay(headers) -> float:
    value = headers.get("Retry-After", "")
    try:
        return max(1.0, float(value))
    except ValueError:
        if value:
            from email.utils import parsedate_to_datetime
            try:
                return max(1.0, parsedate_to_datetime(value).timestamp() - time.time())
            except (TypeError, ValueError, OverflowError):
                pass
    try:
        return max(1.0, float(headers.get("X-RateLimit-Reset", "0")) - time.time() + 1)
    except ValueError:
        return 60.0


def cached_html(request: Request, key: str, cache: Path, *, opener=None, sleep=None) -> str:
    """Shared rate-limit handling; injected transports keep tests offline."""
    opener = opener or urlopen
    sleep = sleep or time.sleep
    stored = cache / f"{key}.html"
    if stored.is_file():
        return stored.read_text(encoding="utf-8")
    for attempt in range(8):
        try:
            with opener(request, timeout=60) as response:
                rendered = response.read().decode("utf-8")
            stored.write_text(rendered, encoding="utf-8")
            return rendered
        except HTTPError as error:
            status = error.code
            headers = error.headers
            error.close()
            if status in (403, 429):
                # GitHub distinguishes primary and secondary rate limits via these headers.
                sleep(retry_delay(headers) if headers.get("Retry-After") or headers.get("X-RateLimit-Reset") else 60 * (attempt + 1))
                continue
            raise RuntimeError(f"GitHub Markdown request failed (HTTP {status})") from None
        except (URLError, TimeoutError, OSError, UnicodeError):
            raise RuntimeError("GitHub Markdown request failed") from None
    raise RuntimeError("GitHub rate-limit retries exhausted")


def fetch_html(path: Path, token: str, cache: Path) -> str:
    body = json.dumps({"text": path.read_text(encoding="utf-8"), "mode": "gfm"}).encode()
    key = hashlib.sha256(body).hexdigest()
    request = Request("https://api.github.com/markdown", data=body, method="POST", headers={
        "Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
        "Content-Type": "application/json", "User-Agent": "theory-markdown-validator",
        "X-GitHub-Api-Version": "2022-11-28"})
    return cached_html(request, key, cache)


def html_report(path: Path, rendered: str, report: Report) -> RenderedHTML:
    parser = RenderedHTML()
    parser.feed(rendered)
    parser.close()
    for _ in range(parser.raw_math_blocks):
        report.hit(path, 1, 'unrendered display math in GitHub HTML: <pre lang="math">')
    for m in re.finditer(r"\$|\\[A-Za-z]+", "".join(parser.text)):
        report.hit(path, 1, f"unrendered math in GitHub HTML: {m[0]}")
    for emphasis in parser.emphasis:
        if "\\" in emphasis or "$" in emphasis:
            report.hit(path, 1, "GitHub emphasis contains a backslash or dollar sign outside math")
    for tags in parser.math_markup:
        report.hit(path, 1, f"markup inside GitHub math-renderer: {', '.join(sorted(tags))}")
    return parser


def prepare_cache(cache: Path, root: Path = ROOT) -> Path:
    cache = cache.resolve()
    root = root.resolve()
    if cache == root or root in cache.parents:
        raise RuntimeError("The HTML cache must be outside the repository")
    cache.mkdir(parents=True, exist_ok=True)
    return cache


def online(files, cache, report):
    cache = prepare_cache(cache)
    token = credential()
    # /markdown omits heading IDs; heading anchors are checked offline only.
    for path in files:
        html_report(path, fetch_html(path, token, cache), report)


BLOB_REF = "refs/render-check/tmp"


def git_command(args: list[str], data: bytes | None = None, *, root: Path = ROOT) -> str:
    """Never expose Git's output on failure (remote URLs may hold credentials)."""
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    try:
        result = subprocess.run(["git", *args], input=data, cwd=root, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                check=False, timeout=120)
        if result.returncode:
            raise RuntimeError("Git render-check operation failed")
        return result.stdout.decode("utf-8").strip()
    except (OSError, subprocess.TimeoutExpired, UnicodeError):
        raise RuntimeError("Git render-check operation failed") from None


def github_repository(remote: str) -> str:
    """Parse HTTPS, SSH and scp-style GitHub origins without exposing the URL."""
    if remote.startswith("git@github.com:"):
        path = remote[len("git@github.com:"):]
    else:
        try:
            parsed = urlsplit(remote)
        except ValueError:
            raise RuntimeError("Origin must be a GitHub repository URL") from None
        if parsed.scheme not in {"https", "ssh", "git"} or parsed.hostname != "github.com":
            raise RuntimeError("Origin must be a GitHub repository URL")
        path = parsed.path.lstrip("/")
    path = path.rstrip("/").removesuffix(".git")
    if not re.fullmatch(r"[A-Za-z0-9_-]+/[A-Za-z0-9_.-]+", path):
        raise RuntimeError("Origin must be a GitHub repository URL")
    return path


def object_id(value: str) -> str:
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", value):
        raise RuntimeError("Git returned an invalid render-check object ID")
    return value


def build_blob_commit(files: list[Path], root: Path, git) -> str:
    """Snapshot working files, never the index or HEAD; create a parentless commit."""
    tree: dict = {}
    for path in sorted(files):
        parts = path.absolute().relative_to(root.absolute()).parts
        node = tree
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        oid = object_id(git(["hash-object", "-w", "--stdin"], path.read_bytes()))
        mode = "100755" if path.stat().st_mode & 0o111 else "100644"
        node[parts[-1]] = (mode, "blob", oid)

    def write_tree(node):
        entries = []
        for name, value in sorted(node.items()):
            mode, kind, oid = ("040000", "tree", write_tree(value)) if isinstance(value, dict) else value
            entries.append(f"{mode} {kind} {oid}\t{name}\0".encode("utf-8"))
        return object_id(git(["mktree", "-z"], b"".join(entries)))

    tree_id = write_tree(tree)
    return object_id(git(["-c", "user.name=Theory render check",
                          "-c", "user.email=render-check@localhost",
                          "commit-tree", tree_id], b"Temporary Markdown rendering check\n"))


def fetch_blob_html(path: Path, root: Path, repository: str, sha: str,
                    token: str, cache: Path, *, opener=None, sleep=None) -> str:
    relative = quote(path.relative_to(root).as_posix(), safe="/")
    url = f"https://api.github.com/repos/{repository}/contents/{relative}?ref={sha}"
    request = Request(url, method="GET", headers={
        "Authorization": f"Bearer {token}", "Accept": "application/vnd.github.html",
        "User-Agent": "theory-markdown-validator", "X-GitHub-Api-Version": "2022-11-28"})
    key = hashlib.sha256(("blob:" + url).encode()).hexdigest()
    return cached_html(request, key, cache, opener=opener, sleep=sleep)


def blob_anchors(files: list[Path], rendered: dict[Path, RenderedHTML], report: Report):
    for path in files:
        # Verify expected headings/explicit anchors and local links against the
        # file-view IDs, including percent-encoded cross-file fragments.
        for anchor, n in headings(path.read_text(encoding="utf-8")).items():
            if anchor not in rendered[path.resolve()].anchors:
                report.hit(path, n, f"missing GitHub HTML anchor: {anchor}")
        for n, line, fence in lines_outside_fences(path.read_text(encoding="utf-8")):
            if fence:
                continue
            for _, dest, _, _ in markdown_links(line):
                u = urlsplit(html.unescape(dest))
                if u.scheme or u.netloc or dest.startswith("/") or not u.fragment:
                    continue
                target = (path.parent / unquote(u.path)).resolve() if u.path else path.resolve()
                if target in rendered and unquote(u.fragment) not in rendered[target].anchors:
                    report.hit(path, n, f"missing GitHub HTML link anchor: {dest}")


def blob(files, cache, report, *, root=ROOT, git=None, get_credential=None,
         opener=None, sleep=None):
    cache = prepare_cache(cache, root)
    git = git or (lambda args, data=None: git_command(args, data, root=root))
    token = (get_credential or credential)()
    repository = github_repository(git(["remote", "get-url", "origin"]))
    sha = build_blob_commit(files, root, git)
    # Include the push in try: a timeout can happen after the server accepted it.
    try:
        git(["push", "-f", "origin", f"{sha}:{BLOB_REF}"])
        rendered = {}
        for path in files:
            output = fetch_blob_html(path, root, repository, sha, token, cache,
                                     opener=opener, sleep=sleep)
            rendered[path.resolve()] = html_report(path, output, report)
        blob_anchors(files, rendered, report)
    finally:
        try:
            git(["push", "origin", "--delete", BLOB_REF])
        except RuntimeError:
            raise RuntimeError("Could not delete refs/render-check/tmp; remove the hidden ref manually") from None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online", action="store_true", help="also POST Markdown to GitHub; credentials stay in memory")
    parser.add_argument("--blob", action="store_true", help="authoritative file-view check; temporarily pushes a parentless commit to a hidden ref")
    parser.add_argument("--cache", type=Path, help="HTML cache directory outside the repository (default: temporary directory)")
    parser.add_argument("paths", metavar="PATH", nargs="*", type=Path,
                        help="check only these .md files (repository-relative or absolute inside it)")
    args = parser.parse_args(argv)
    files, s1 = scope(ROOT)
    if args.paths:
        try:
            files = select_files(args.paths, ROOT)
        except ValueError as error:
            parser.error(str(error))
        s1 = {path.resolve() for path in s1}
    report = Report()
    offline(files, s1, report)
    if args.online or args.blob:
        try:
            def render(cache):
                if args.online:
                    online(files, cache, report)
                if args.blob:
                    blob(files, cache, report)
            if args.cache:
                render(args.cache)
            else:
                with tempfile.TemporaryDirectory(prefix="theory-markdown-") as directory:
                    render(Path(directory))
        except RuntimeError as error:
            # RuntimeError messages above are fixed strings, never raw SDK/network errors.
            print(str(error), file=sys.stderr)
            return 1
    for rule, count in sorted(report.counts.items()):
        print(f"  {rule}: {count}")
    print(f"{'FAIL' if report.errors else 'PASS'}: {len(files)} files; {report.errors} errors"
          + ("; offline + GitHub file view" if args.blob else
             "; offline + GitHub Markdown API" if args.online else "; offline"))
    return int(bool(report.errors))


if __name__ == "__main__":
    sys.exit(main())
