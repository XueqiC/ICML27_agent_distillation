#!/usr/bin/env python3
"""Synchronize per-page bibliographies and global usage indexes (stdlib, offline)."""
from __future__ import annotations

import argparse
from collections import Counter
import html
import os
from pathlib import Path
import re
import sys
from typing import NamedTuple
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import check_markdown as md

ROOT = Path(__file__).resolve().parents[2]
REFERENCE_NAMES = {"references.md", "references_ZH.md"}
TITLES = {"References", "参考文献"}
ANCHOR = re.compile(r'<a\s+id=["\']([^"\']+)["\']\s*></a>')
ITEM = re.compile(r"(?:Definition|Assumption|Theorem|Proposition|Lemma|Corollary|Remark|Example|Algorithm|定义|假设|定理|命题|引理|推论|注|备注|例|算法)\s+(\d+)\.(\d+)(?!\d)")
HEADER_SENTENCES = {
    "EN": '"Used in" lists the framework items and pages whose pages cite the work; item links go to the statement pages.',
    "ZH": '"引用于"列出其页面引用该文献的框架条目及页面；条目链接指向 statement 页。',
}


class ReferenceError(ValueError):
    def __init__(self, path: Path, line: int, message: str):
        self.path, self.line, self.message = path, line, message
        super().__init__(f"{path}:{line}: {message}")


def framework_root(path: Path) -> Path | None:
    return next((p for p in path.resolve().parents if p.name == "theoretical_framework"), None)


def is_ref_page(path: Path, framework: Path) -> bool:
    return (path.suffix == ".md" and path.is_relative_to(framework)
            and path not in {framework / name for name in REFERENCE_NAMES})


def masked_source(text: str) -> str:
    """Keep offsets/newlines, masking fenced blocks and even multiline code spans."""
    lines = text.splitlines(keepends=True)
    for n, _, fence in md.lines_outside_fences(text):
        if fence:
            lines[n - 1] = re.sub(r"[^\n]", " ", lines[n - 1])
    source = "".join(lines)
    pos = 0
    while pos < len(source):
        if source[pos] == "\\":
            pos += 2
            continue
        if source[pos] != "`":
            pos += 1
            continue
        marker = re.match(r"`+", source[pos:])[0]
        close = re.search(r"(?<!`)" + re.escape(marker) + r"(?!`)", source[pos + len(marker):])
        if close:
            end = pos + len(marker) + close.end()
            source = source[:pos] + re.sub(r"[^\n]", " ", source[pos:end]) + source[end:]
            pos = end
        else:
            pos += len(marker)
    return source


class Link(NamedTuple):
    target: str
    start: int
    end: int
    line: int
    link_start: int


def inline_links(text: str):
    source = masked_source(text)
    for label, dest, start, _ in md.markdown_links(source):
        # Images and reference definitions are not inline citation links.
        if source[start + len(label) + 1:start + len(label) + 3] != "](":
            continue
        if start and source[start - 1] == "!" and (start < 2 or source[start - 2] != "\\"):
            continue
        offset = start + len(label) + 3
        while offset < len(source) and source[offset].isspace():
            offset += 1
        if source[offset:offset + 1] == "<":
            offset += 1
        yield Link(dest, offset, offset + len(dest), text.count("\n", 0, start) + 1, start)


def citation_key(path: Path, target: str, framework: Path) -> tuple[str, bool] | None:
    url = urlsplit(html.unescape(target))
    if url.scheme or url.netloc or target.startswith("/") or not url.fragment:
        return None
    fragment = unquote(url.fragment)
    if not url.path and not url.query and fragment.startswith("ref-"):
        return fragment[4:], False
    if url.path and (path.parent / unquote(url.path)).resolve() in {
            framework / name for name in REFERENCE_NAMES}:
        return fragment, True
    return None


def citations(path: Path, text: str, framework: Path):
    for link in inline_links(text):
        key = citation_key(path, link.target, framework)
        if key is not None:
            yield link, *key


class Section(NamedTuple):
    title: str
    start: int
    content: int
    end: int
    line: int


def h2_sections(text: str) -> list[Section]:
    source = masked_source(text)
    raw_lines = text.splitlines(keepends=True)
    headings = []
    offset = 0
    previous = None
    for n, line in enumerate(source.splitlines(keepends=True), 1):
        # Masking determines whether a heading is visible, but its title must
        # retain inline code ("References `notes`" is not "References").
        match = (re.match(r"^ {0,3}##\s+(.+?)(?:\s+#+)?\s*$", raw_lines[n - 1])
                 if re.match(r"^ {0,3}##\s+", line) else None)
        if match:
            headings.append((match[1], offset, offset + len(line), n))
        elif previous and re.fullmatch(r" {0,3}-+\s*", line) and previous[0].strip():
            headings.append((previous[0].strip(), previous[1], offset + len(line), n - 1))
        previous = (raw_lines[n - 1] if line.strip() else "", offset)
        offset += len(line)
    return [Section(*row[:3], headings[i + 1][1] if i + 1 < len(headings) else len(text), row[3])
            for i, row in enumerate(headings)]


def body_text(text: str) -> str:
    """Exclude all bibliography sections, preserving diagnostic line numbers."""
    for section in reversed(h2_sections(text)):
        if section.title in TITLES:
            text = text[:section.start] + re.sub(r"[^\n]", " ", text[section.start:section.end]) + text[section.end:]
    return text


class Catalog:
    def __init__(self, framework: Path, texts: dict[Path, str]):
        self.framework = framework
        self.entries: dict[str, dict[str, str]] = {}
        self.blocks: dict[str, list[tuple[str, int, int]]] = {}
        self.texts = texts
        for lang, name in (("EN", "references.md"), ("ZH", "references_ZH.md")):
            path = framework / name
            if path not in texts:
                raise ReferenceError(path, 1, "missing reference catalog")
            text = texts[path]
            anchors = list(ANCHOR.finditer(masked_source(text)))
            entries, blocks = {}, []
            for i, anchor in enumerate(anchors):
                key = html.unescape(anchor[1])
                line = text.count("\n", 0, anchor.start()) + 1
                if key in entries:
                    raise ReferenceError(path, line, f"duplicate reference key: {key}")
                end = anchors[i + 1].start() if i + 1 < len(anchors) else len(text)
                rest = text[anchor.end():end].lstrip("\r\n")
                paragraph = re.split(r"\n[ \t]*\n", rest, maxsplit=1)[0].rstrip("\r\n")
                if not paragraph.startswith("**") or re.search(r"^(?:Used in:|引用于：)", paragraph, re.M):
                    raise ReferenceError(path, line, f"missing bibliographic paragraph: {key}")
                entries[key] = paragraph
                blocks.append((key, anchor.start(), end))
            self.entries[lang], self.blocks[lang] = entries, blocks
        if self.entries["EN"].keys() != self.entries["ZH"].keys():
            difference = sorted(self.entries["EN"].keys() ^ self.entries["ZH"].keys())
            raise ReferenceError(framework / "references_ZH.md", 1,
                                 "reference key sets differ: " + ", ".join(difference))
        self.keys = tuple(self.entries["EN"])


def relative_url(target: Path, page: Path) -> str:
    return quote(Path(os.path.relpath(target, page.parent)).as_posix(), safe="/._-~")


def intro(path: Path, framework: Path) -> str:
    if md.language(path) == "ZH":
        return f"本页引用的文献。完整列表及每篇文献的使用位置见 [参考文献]({relative_url(framework / 'references_ZH.md', path)})。"
    return ("Works cited on this page. The complete list, with the pages that use each work, "
            f"is in [References]({relative_url(framework / 'references.md', path)}).")


def entry_for_page(path: Path, key: str, catalog: Catalog) -> str:
    lang = md.language(path)
    entry = catalog.entries[lang][key]
    source = catalog.framework / ("references_ZH.md" if lang == "ZH" else "references.md")
    for link in reversed(list(inline_links(entry))):
        url = urlsplit(html.unescape(link.target))
        if url.scheme or url.netloc or link.target.startswith("/"):
            continue
        target = (source.parent / unquote(url.path)).resolve() if url.path else source
        dest = urlunsplit(("", "", relative_url(target, path), url.query, url.fragment))
        entry = entry[:link.start] + dest + entry[link.end:]
    return entry


def generate_section(path: Path, keys: set[str], catalog: Catalog) -> str:
    if not keys:
        return ""
    unknown = keys - set(catalog.keys)
    if unknown:
        raise ReferenceError(path, 1, "unknown reference key: " + ", ".join(sorted(unknown)))
    title = "参考文献" if md.language(path) == "ZH" else "References"
    parts = [f"## {title}", intro(path, catalog.framework)]
    for key in catalog.keys:
        if key in keys:
            parts.extend((f'<a id="ref-{key}"></a>', entry_for_page(path, key, catalog)))
    return "\n\n".join(parts) + "\n"


def check_old_section(path: Path, text: str, section: Section, catalog: Catalog) -> set[str]:
    old = text[section.content:section.end]
    base_line = text.count("\n", 0, section.content) + 1
    nonblank = [(base_line + i, line) for i, line in enumerate(old.splitlines()) if line.strip()]
    keys = {key for _, key, _ in citations(path, old, catalog.framework)}
    if nonblank and nonblank[0][1] == intro(path, catalog.framework):
        # A generated section has only its intro and anchor/paragraph pairs.
        # Permit stale bibliographic prose so catalog corrections can propagate.
        chunks = list(re.finditer(r"\S[^\n]*(?:\n(?![ \t]*\n)[^\n]+)*", old))
        expect_anchor = True
        for chunk in chunks[1:]:
            value = chunk[0].rstrip()
            line = base_line + old.count("\n", 0, chunk.start())
            if expect_anchor:
                match = ANCHOR.fullmatch(value)
                if not match or not match[1].startswith("ref-"):
                    raise ReferenceError(path, line, "unsafe references section: expected generated anchor")
                keys.add(html.unescape(match[1])[4:])
            elif (not value.startswith("**") or re.search(
                    r"(?m)^(?:[ \t]*[-+*] |#{1,6} |```|~~~|<a |Used in:|引用于：|\$)", value)):
                raise ReferenceError(path, line, "unsafe references section: expected entry paragraph")
            expect_anchor = not expect_anchor
        if not expect_anchor:
            raise ReferenceError(path, nonblank[-1][0], "unsafe references section: missing entry paragraph")
    else:
        for n, line in nonblank:
            marker = re.match(r"^ {0,3}(?:[-+*]|\d+[.)]) +", line)
            links = list(inline_links(line))
            if (not marker or not links or links[0].link_start != marker.end()
                    or citation_key(path, links[0].target, catalog.framework) is None):
                raise ReferenceError(path, n, "unsafe references section: expected citation-led list item")
    unknown = keys - set(catalog.keys)
    if unknown:
        raise ReferenceError(path, section.line, "unknown reference key: " + ", ".join(sorted(unknown)))
    return keys


class PageResult(NamedTuple):
    text: str
    keys: set[str]
    disappeared: set[str]
    count: int


def synchronize_page(path: Path, text: str, catalog: Catalog) -> PageResult:
    sections = h2_sections(text)
    references = [section for section in sections if section.title in TITLES]
    old_keys = set()
    if references:
        if len(references) != 1 or references[0] != sections[-1]:
            raise ReferenceError(path, references[0].line, "references section must be unique and the last H2")
        old_keys = check_old_section(path, text, references[0], catalog)
        text = text[:references[0].start]
    links = list(citations(path, text, catalog.framework))
    keys = {key for _, key, _ in links}
    for link, key, legacy in reversed(links):
        if key not in catalog.keys:
            raise ReferenceError(path, link.line, f"unknown reference key: {key}")
        if legacy:
            text = text[:link.start] + "#ref-" + key + text[link.end:]
    body = text.rstrip()
    section = generate_section(path, keys, catalog)
    result = (body + "\n\n" if body and section else body) + section
    return PageResult(result.rstrip("\r\n") + "\n", keys, old_keys - keys, len(links))


def counterpart(path: Path, lang: str) -> Path:
    return path.with_name(path.stem.removesuffix("_ZH") + ("_ZH" if lang == "ZH" else "") + ".md")


def usage_lines(catalog: Catalog, texts: dict[Path, str]) -> dict[str, dict[str, str]]:
    """Union bilingual citations by item/page; localize links to available counterparts."""
    framework = catalog.framework
    uses: dict[str, set[Path]] = {key: set() for key in catalog.keys}
    for path, text in texts.items():
        if is_ref_page(path, framework):
            for _, key, _ in citations(path, body_text(text), framework):
                if key in uses:
                    uses[key].add(path)
    result = {}
    for lang, name in (("EN", "references.md"), ("ZH", "references_ZH.md")):
        result[lang] = {}
        for key, pages in uses.items():
            items, other = {}, {}
            for path in sorted(pages):
                rel = path.relative_to(framework)
                if len(rel.parts) == 2 and rel.parts[0] in md.PAGE_KINDS and md.page_number(path):
                    statement = framework / "statements" / path.name
                    localized = counterpart(statement, lang)
                    target = localized if localized in texts else statement
                    if target not in texts:
                        raise ReferenceError(path, 1, "missing statement page for Used in index")
                    h1 = re.search(r"(?m)^# (.+)$", masked_source(texts[target]))
                    match = ITEM.match(h1[1]) if h1 else None
                    if not match:
                        raise ReferenceError(target, 1, "missing item name in statement H1")
                    number = tuple(map(int, match.groups()))
                    if md.page_number(target) != f"{number[0]}.{number[1]}":
                        raise ReferenceError(target, 1, "statement H1 number differs from filename")
                    items[number] = (match[0], target)
                else:
                    identity = counterpart(path, "EN")
                    localized = counterpart(path, lang)
                    # A non-item link always leads to a page actually citing this key.
                    target = localized if localized in pages else path
                    if identity in other and md.language(other[identity][1]) == lang:
                        continue
                    if rel.parent == Path(".") and identity.name == "README.md":
                        label = "框架总览" if lang == "ZH" else "Framework overview"
                    elif rel.parent == Path("tools") and identity.name == "README.md":
                        label = "校验工具" if lang == "ZH" else "Validation tools"
                    elif rel.parent == Path("figures/specs") and (figure := re.search(r"fig(\d+)[_-](\d+)", path.stem)):
                        number = f"{int(figure[1])}.{int(figure[2])}"
                        label = f"图 {number} 说明" if lang == "ZH" else f"Figure {number} spec"
                    elif rel.parent == Path("figures") and identity.name == "README.md":
                        label = "图表总览" if lang == "ZH" else "Figure overview"
                    else:
                        heading = re.search(r"(?m)^# (.+)$", texts[target])
                        label = heading[1] if heading else rel.as_posix()
                    other[identity] = (label, target)
            links = [items[number] for number in sorted(items)] + [other[p] for p in sorted(other)]
            value = ", ".join(f"[{label}]({relative_url(target, framework / name)})" for label, target in links)
            result[lang][key] = ("引用于：" if lang == "ZH" else "Used in: ") + (value or ("无" if lang == "ZH" else "none"))
    return result


def update_catalog(catalog: Catalog, lang: str, usage: dict[str, str]) -> str:
    path = catalog.framework / ("references_ZH.md" if lang == "ZH" else "references.md")
    text = catalog.texts[path]
    blocks = catalog.blocks[lang]
    header = text[:blocks[0][1]] if blocks else text
    pattern = r'["“]引用于["”][^。]*。' if lang == "ZH" else r'["“]Used in["”][^.]*\.'
    if re.search(pattern, header):
        header = re.sub(pattern, HEADER_SENTENCES[lang], header, count=1)
    else:
        header = header.rstrip() + " " + HEADER_SENTENCES[lang] + "\n\n"
    chunks = [header]
    for key, start, end in blocks:
        block = text[start:end]
        pattern = r"(?m)^(?:Used in:|引用于：)[^\n]*"
        matches = list(re.finditer(pattern, block))
        if matches:
            for i, match in reversed(list(enumerate(matches))):
                block = block[:match.start()] + (usage[key] if i == 0 else "") + block[match.end():]
        else:
            block = block.rstrip() + "\n\n" + usage[key] + "\n\n"
        chunks.append(block)
    return "".join(chunks).rstrip() + "\n"


def main(argv=None, *, root: Path = ROOT) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report changes without writing")
    parser.add_argument("paths", nargs="*", type=Path, metavar="PATH", help="repository-relative ref pages or directories")
    args = parser.parse_args(argv)
    framework = (root / "theoretical_framework").resolve()
    paths = sorted(framework.rglob("*.md"))
    selected = set(p for p in paths if is_ref_page(p, framework))
    if args.paths:
        selected = set()
        for raw in args.paths:
            path = (root / raw).resolve()
            if not path.is_relative_to(framework) or not path.exists():
                parser.error(f"not an existing framework path: {raw}")
            candidates = path.rglob("*.md") if path.is_dir() else [path]
            for candidate in candidates:
                candidate = candidate.resolve()
                if not is_ref_page(candidate, framework):
                    if path.is_file():
                        parser.error(f"not a ref page: {raw}")
                    continue
                selected.add(candidate)
    # Snapshot and plan everything before the first write. Errors cannot truncate
    # any page, and selected runs still recompute usage from the whole framework.
    try:
        texts = {p: p.read_text(encoding="utf-8") for p in paths}
        catalog = Catalog(framework, texts)
        changed, errors = {}, []
        for path in sorted(selected):
            try:
                result = synchronize_page(path, texts[path], catalog)
                if result.text != texts[path]:
                    changed[path] = result.text
                for key in sorted(result.disappeared):
                    print(f"WARNING {path.relative_to(root)}: citation disappears from page: {key}")
            except ReferenceError as error:
                errors.append(error)
        usage = usage_lines(catalog, texts)
        for lang, name in (("EN", "references.md"), ("ZH", "references_ZH.md")):
            path = framework / name
            updated = update_catalog(catalog, lang, usage[lang])
            if updated != texts[path]:
                changed[path] = updated
        for error in errors:
            print(f"ERROR {error}")
        for path in sorted(changed):
            print(f"{'Would change' if args.check or errors else 'Rewrite'}: {path.relative_to(root)}")
        if not args.check and not errors:
            # Refuse to overwrite an edit made since the snapshot, including an
            # unselected page whose new citations would invalidate the usage index.
            for path, original in texts.items():
                if path.read_text(encoding="utf-8") != original:
                    raise ReferenceError(path, 1, "file changed during synchronization; rerun")
            for path, updated in changed.items():
                path.write_text(updated, encoding="utf-8")
        print(f"Pages rewritten: {0 if args.check or errors else len(selected & changed.keys())}; "
              f"would rewrite: {len(selected & changed.keys())}; files that would change: {len(changed)}")
        print(f"Pages with errors: {len(errors)}")
        print("Citations per page (body links / distinct works):")
        page_citations = {p: list(citations(p, body_text(texts[p]), framework)) for p in selected}
        counts = Counter((len(links), len({key for _, key, _ in links})) for links in page_citations.values())
        for (count, distinct), pages in sorted(counts.items()):
            print(f"  {count} / {distinct}: {pages} pages")
        empty = [p for p, links in page_citations.items() if not links]
        print(f"Pages without citations: {len(empty)}")
        for path in sorted(empty):
            print(f"  {path.relative_to(root)}")
        return int(bool(errors or args.check and changed))
    except (ReferenceError, OSError) as error:
        print(f"ERROR {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
