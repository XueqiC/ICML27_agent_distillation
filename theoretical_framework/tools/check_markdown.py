#!/usr/bin/env python3
"""Validate the scoped theory Markdown, offline by default (Python 3.11, stdlib)."""
from __future__ import annotations

import argparse
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
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[2]
DENIED = re.compile(
    r"\\(?:operatorname|boldsymbol|tag|label|eqref|ref|require|href|url|"
    r"newcommand|renewcommand|def|class|style|cssId|html|unicode|bbox)(?![A-Za-z])"
)
FENCE = re.compile(r"^\s*(`{3,}|~{3,})([^\n]*)$")
DOLLAR = re.compile(r"(?<!\\)(?:\\\\)*\$")
RESULT = re.compile(
    r"(?<![A-Za-z0-9_])(?:Props?\.|Propositions?|Thms?\.?|Theorems?|Lemma|"
    r"Cor\.|Corollary|Corollaries|定理|命题|引理|推论)\s*"
    r"(?:[ACW]?\d+(?:\.\d+|[abc′″])?(?:\([a-g]\))?|[ABC](?![A-Za-z]))"
)
EQ_REF = re.compile(r"[（(]([AB]?\d+[a-z]?)[)）]")
SOURCE_LINE = re.compile(r"^\*\*(?:Source:|出处：)\*\* ")
DISPLAY = re.compile(r"\\[\[\]]|\\begin\{(?:equation\*?|align\*?|gather\*?|multline\*?|displaymath)\}")


def scope(root: Path = ROOT) -> tuple[list[Path], set[Path]]:
    framework = root / "theoretical_framework"
    s1 = [framework / "README.md", framework / "README_ZH.md"]
    s1 += sorted((framework / "proofs").glob("*.md"))
    s1 += sorted((framework / "experiments").glob("*.md"))
    s1 += [framework / "figures/README.md",
           root / "docs/direction/2026-10-11_v015_progress_ZH.md"]
    s2 = [root / "docs/direction/v015_extension" / name for name in (
        "Warm_Adam_Response_EN.md", "Optimizer_Transformed_Linear_Response_EN.md")]
    return s1 + s2, set(s1)


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
    return "".join(c for c in text if unicodedata.category(c)[0] in "LMN"
                   or unicodedata.category(c) == "Pc" or c in " -").replace(" ", "-")


def lines_outside_fences(text: str):
    """Yield (line number, line, enclosing fence language); fences are included."""
    marker = None
    language = ""
    for n, line in enumerate(text.splitlines(), 1):
        m = FENCE.match(line)
        if marker:
            yield n, line, language or "code"
            if m and m[1][0] == marker[0] and len(m[1]) >= len(marker) and not m[2].strip():
                marker = None
                language = ""
        elif m:
            marker = m[1]
            language = m[2].strip()
            yield n, line, language or "code"
        else:
            yield n, line, None


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


def local_results(text: str) -> set[str]:
    result = set()
    for _, line, fence in lines_outside_fences(text):
        if fence or SOURCE_LINE.match(line):
            continue
        # A result is locally defined only by a bold declaration or a heading,
        # never merely by a mention in a proof or a source line.
        m = re.match(r"^(?:#{1,6}\s+|\*\*)(Lemma|Corollary|引理|推论)\s+"
                     r"([123](?:\.0)?|[ABC])(?=\s*[（(])", line)
        if m:
            result.add(m[0].lstrip("#* ").strip())
    return result


class Report:
    def __init__(self):
        self.errors = 0

    def hit(self, path: Path, line: int, message: str):
        try:
            name = path.relative_to(ROOT)
        except ValueError:
            name = path
        print(f"{name}:{line}: {message}")
        self.errors += 1


def offline(files: list[Path], s1: set[Path], report: Report):
    anchor_cache: dict[Path, dict[str, int]] = {}
    for path in files:
        if not path.is_file():
            report.hit(path, 1, "missing scoped file")
            continue
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        visible_numbers = set(re.findall(r"\\qquad\(([^)]+)\)", "\n".join(
            l for _, l, f in lines_outside_fences(text) if f == "math")))
        local = local_results(text)
        local_warm = set(re.findall(r"\*\*\((W[0-3])\)\*\*", text))
        heading_lines = set(headings(text).values())
        reference_labels = {" ".join(m[1].split()).casefold()
                            for _, l, f in lines_outside_fences(text) if not f
                            if (m := re.match(r"^ {0,3}\[([^\]]+)\]:", l))}
        concordance = False
        for n, line, fence in lines_outside_fences(text):
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
                        anchor_cache[target] = headings(target.read_text(encoding="utf-8")) if target.suffix.lower() == ".md" else {}
                    if unquote(u.fragment) not in anchor_cache[target]:
                        report.hit(path, n, f"missing heading anchor: {dest}")
            if path not in s1:
                continue
            if re.match(r"^## (Source concordance|出处对照)\s*$", line):
                concordance = True
            elif re.match(r"^#{1,2} ", line):
                concordance = False
            if concordance or (path.parent.name == "proofs" and SOURCE_LINE.match(line)):
                continue
            prose = prose_only(line)
            # Link labels and image alt text are visible references too. Do not
            # exempt an old result name merely because it is inside a link.
            labels = " ".join(label for label, _, _, _ in markdown_links(line))
            labels = re.sub(r"(?<!\\)\$(?:\\.|[^$])*?\$", "", labels)
            for m in RESULT.finditer(prose + " " + labels):
                if m[0] not in local:
                    report.hit(path, n, f"original/nonlocal result label: {m[0]}")
            for m in EQ_REF.finditer(prose):
                k = m[1]
                if k in visible_numbers or (k.isdigit() and int(k) >= 1900):
                    continue
                # Numbered steps/equivalent conditions refer to local enumerated clauses.
                if k in {"1", "2", "3"} and re.search(r"^" + k + r"\. ", text, re.M):
                    continue
                if re.search(r"(?:Optimizers|优化器)\s*$", prose[:m.start()]):
                    continue
                report.hit(path, n, f"nonlocal equation/number label: {m[0]}")
            for m in re.finditer(r"(?:eqs?\.\s*|式\s*)[（(](A?\d+[a-z]?)[)）]", prose):
                if m[1] not in visible_numbers:
                    report.hit(path, n, f"nonlocal equation citation: {m[0]}")
            for m in re.finditer(r"§+\d+(?:\.\d+)?|(?<![A-Za-z0-9_])A\d+[′″]?(?![A-Za-z0-9_])", prose):
                if m[0] not in visible_numbers:
                    report.hit(path, n, f"bare source label: {m[0]}")
            if (path.parent.name == "theoretical_framework" and path.name.startswith("README")) or "v015_progress" in path.name:
                for m in re.finditer(r"(?<![A-Za-z0-9_])[CVET]\d+(?![A-Za-z0-9_])", prose):
                    report.hit(path, n, f"unlinked check ID: {m[0]}")
            for m in re.finditer(r"(?<![A-Za-z0-9_-])W[0-3](?:\([ab]\))?(?![A-Za-z0-9_])", prose):
                if not path.name.startswith("N12_") and m[0] not in local_warm:
                    report.hit(path, n, f"bare warm-Adam label: {m[0]}")
        state = list(lines_outside_fences(text))
        if state and state[-1][2] and not (FENCE.match(state[-1][1]) and not FENCE.match(state[-1][1])[2].strip()):
            report.hit(path, len(lines), "unterminated fenced block")


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

    def handle_starttag(self, tag, attrs):
        attr = dict(attrs)
        parent_hidden = self.stack[-1][1] if self.stack else False
        hidden = parent_hidden or tag in {"math-renderer", "pre", "code"} or "mermaid" in attr.get("class", "").split()
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


def fetch_html(path: Path, token: str, cache: Path) -> str:
    body = json.dumps({"text": path.read_text(encoding="utf-8"), "mode": "gfm"}).encode()
    key = hashlib.sha256(body).hexdigest()
    stored = cache / f"{key}.html"
    if stored.is_file():
        return stored.read_text(encoding="utf-8")
    request = Request("https://api.github.com/markdown", data=body, method="POST", headers={
        "Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
        "Content-Type": "application/json", "User-Agent": "theory-markdown-validator",
        "X-GitHub-Api-Version": "2022-11-28"})
    for attempt in range(8):
        try:
            with urlopen(request, timeout=60) as response:
                rendered = response.read().decode("utf-8")
            stored.write_text(rendered, encoding="utf-8")
            return rendered
        except HTTPError as error:
            status = error.code
            headers = error.headers
            error.close()
            if status in (403, 429):
                # GitHub distinguishes primary and secondary rate limits via these headers.
                time.sleep(retry_delay(headers) if headers.get("Retry-After") or headers.get("X-RateLimit-Reset") else 60 * (attempt + 1))
                continue
            raise RuntimeError(f"GitHub Markdown request failed (HTTP {status})") from None
        except (URLError, TimeoutError, OSError, UnicodeError):
            raise RuntimeError("GitHub Markdown request failed") from None
    raise RuntimeError("GitHub rate-limit retries exhausted")


def online(files, cache, report):
    cache = cache.resolve()
    if cache == ROOT or ROOT in cache.parents:
        raise RuntimeError("The HTML cache must be outside the repository")
    cache.mkdir(parents=True, exist_ok=True)
    token = credential()
    # /markdown omits heading IDs; heading anchors are checked offline only.
    for path in files:
        parser = RenderedHTML()
        parser.feed(fetch_html(path, token, cache))
        for m in re.finditer(r"\$|\\[A-Za-z]+", "".join(parser.text)):
            report.hit(path, 1, f"unrendered math in GitHub HTML: {m[0]}")
        for emphasis in parser.emphasis:
            if "\\" in emphasis or "$" in emphasis:
                report.hit(path, 1, "GitHub emphasis contains a backslash or dollar sign outside math")
        for tags in parser.math_markup:
            report.hit(path, 1, f"markup inside GitHub math-renderer: {', '.join(sorted(tags))}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online", action="store_true", help="also POST Markdown to GitHub; credentials stay in memory")
    parser.add_argument("--cache", type=Path, help="HTML cache directory outside the repository (default: temporary directory)")
    args = parser.parse_args(argv)
    files, s1 = scope()
    report = Report()
    offline(files, s1, report)
    if args.online:
        try:
            if args.cache:
                online(files, args.cache, report)
            else:
                with tempfile.TemporaryDirectory(prefix="theory-markdown-") as directory:
                    online(files, Path(directory), report)
        except RuntimeError as error:
            # RuntimeError messages above are fixed strings, never raw SDK/network errors.
            print(str(error), file=sys.stderr)
            return 1
    print(f"{'FAIL' if report.errors else 'PASS'}: {len(files)} files; {report.errors} errors"
          + ("; offline + GitHub HTML" if args.online else "; offline"))
    return int(bool(report.errors))


if __name__ == "__main__":
    sys.exit(main())
