"""Temporary-tree tests for reference synchronization; no real-page writes."""
import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import sync_references as refs


class ReferenceFixture:
    """Shared complete, tiny bilingual catalog, independent of the real pages."""
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.framework = self.root / "theoretical_framework"
        for lang, suffix, prefix in (("EN", "", "Used in: "), ("ZH", "_ZH", "引用于：")):
            self.write("references" + suffix + ".md",
                       '# References\n\nHeader remains. ' + refs.HEADER_SENTENCES[lang] + '\n\n'
                       '<a id="beta"></a>\n\n**Beta (2020).** Second work. [DOI](https://example.org/b).\n\n'
                       + prefix + ('none' if lang == 'EN' else '无') + '\n\n'
                       '<a id="alpha"></a>\n\n**Alpha (2019).** First work.\n'
                       'Wrapped venue and [source](https://example.org/a).\n\n'
                       + prefix + ('none' if lang == 'EN' else '无') + '\n')
        for obj, name in ((refs.md, "urlopen"), (refs.md, "credential"), (refs.md.subprocess, "run")):
            guard = patch.object(obj, name, side_effect=AssertionError("live I/O attempted"))
            guard.start()
            self.addCleanup(guard.stop)

    def write(self, name, text):
        path = self.framework / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def snapshot(self):
        return {p: p.read_text(encoding="utf-8") for p in self.framework.rglob("*.md")}

    def catalog(self):
        return refs.Catalog(self.framework, self.snapshot())

    def run_sync(self, *args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = refs.main(list(args), root=self.root)
        return code, output.getvalue()

    def page(self, name="tools/README.md", text="# Tools\n\n[Beta](#ref-beta).\n"):
        path = self.write(name, text)
        result = refs.synchronize_page(path, text, self.catalog())
        path.write_text(result.text, encoding="utf-8")
        return path


class SyncReferencesTests(ReferenceFixture, unittest.TestCase):
    def test_legacy_rewrite_preserves_labels_titles_and_navigation(self):
        path = self.write("figures/specs/fig2_1.md", '# Figure\n\n'
                          '[Alpha](../../references_ZH.md#alpha "title") and '
                          '[B [nested]](<../../references.md#beta>).\n'
                          '[All](../../references.md) [All ZH](../../references_ZH.md).\n')
        result = refs.synchronize_page(path, path.read_text(), self.catalog())
        self.assertIn('[Alpha](#ref-alpha "title")', result.text)
        self.assertIn('[B [nested]](<#ref-beta>)', result.text)
        self.assertIn('[All](../../references.md)', result.text)
        self.assertIn('[All ZH](../../references_ZH.md)', result.text)
        self.assertEqual(result.keys, {"alpha", "beta"})
        self.assertLess(result.text.index('id="ref-beta"'), result.text.index('id="ref-alpha"'))
        self.assertIn('Wrapped venue and [source](https://example.org/a).', result.text)
        self.assertNotIn('Used in:', result.text)
        self.assertTrue(result.text.endswith('.\n'))
        self.assertFalse(result.text.endswith('\n\n'))

    def test_regenerates_legacy_section_and_warns_disappearing_keys(self):
        path = self.write("README.md", '# Overview\n\n[Beta](references.md#beta).\n\n'
                          '## References\n\n- [Alpha](references.md#alpha). Full legacy entry.\n'
                          '- [Beta](references.md#beta). Another entry.\n')
        code, output = self.run_sync()
        self.assertEqual(code, 0, output)
        self.assertIn('citation disappears from page: alpha', output)
        text = path.read_text()
        self.assertIn('<a id="ref-beta"></a>\n\n**Beta', text)
        self.assertNotIn('ref-alpha', text)
        self.assertNotIn('legacy entry', text)
        self.assertIn('Used in: none', (self.framework / 'references.md').read_text())

    def test_abort_with_line_number_and_no_writes_on_unsafe_old_content(self):
        for content in ('A paragraph.', '- ordinary bullet', '```math\nx=1\n```',
                        '- [navigation](../references.md)', '- [a](#ref-alpha)\n  wrapped continuation',
                        '- `code` [a](#ref-alpha)', '### Notes', '<a id="other"></a>'):
            with self.subTest(content=content):
                self.write("tools/README.md", '# Tools\n\n[A](#ref-alpha)\n\n## References\n\n' + content + '\n')
                before = self.snapshot()
                code, output = self.run_sync()
                self.assertEqual(code, 1)
                self.assertRegex(output, r'tools/README.md:[78]: unsafe references section')
                self.assertEqual(self.snapshot(), before)

    def test_generated_section_rejects_extra_paragraph_or_bullet(self):
        path = self.page()
        good = path.read_text()
        for extra in ('User notes.\n', '- notes\n', '```math\nx\n```\n'):
            path.write_text(good + '\n' + extra)
            before = self.snapshot()
            code, output = self.run_sync()
            self.assertEqual(code, 1, output)
            self.assertEqual(self.snapshot(), before)

    def test_regeneration_tracks_catalog_edits_and_removed_body_citation(self):
        path = self.page(text='# Tools\n\n[A](#ref-alpha) [B](#ref-beta)\n')
        path.write_text(path.read_text().replace('[A](#ref-alpha) ', ''))
        reference = self.framework / 'references.md'
        reference.write_text(reference.read_text().replace('Second work.', 'Corrected work.'))
        code, output = self.run_sync()
        self.assertEqual(code, 0, output)
        self.assertIn('Corrected work.', path.read_text())
        self.assertNotIn('id="ref-alpha"', path.read_text())
        self.assertIn('citation disappears from page: alpha', output)

    def test_idempotence_and_check_exit_codes(self):
        self.write('README.md', '# Overview\n\n[A](references.md#alpha) [B](#ref-beta)\n')
        before = self.snapshot()
        code, output = self.run_sync('--check')
        self.assertEqual(code, 1, output)
        self.assertIn('Would change: theoretical_framework/README.md', output)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.run_sync()[0], 0)
        after = self.snapshot()
        code, output = self.run_sync('--check')
        self.assertEqual(code, 0, output)
        self.assertIn('files that would change: 0', output)
        self.assertEqual(self.run_sync()[0], 0)
        self.assertEqual(self.snapshot(), after)

    def test_empty_page_gets_no_section_and_is_listed_even_in_clean_check(self):
        path = self.write('README.md', '# Overview\n\n[All](references.md)\n')
        self.assertEqual(self.run_sync()[0], 0)
        code, output = self.run_sync('--check')
        self.assertEqual(code, 0, output)
        self.assertIn('Pages without citations: 1\n  theoretical_framework/README.md', output)
        self.assertNotIn('## References', path.read_text())

    def test_section_only_citations_do_not_create_section_or_usage(self):
        path = self.write('README.md', '# Overview\n\n## References\n\n- [A](references.md#alpha)\n')
        code, output = self.run_sync()
        self.assertEqual(code, 0, output)
        self.assertEqual(path.read_text(), '# Overview\n')
        self.assertIn('citation disappears from page: alpha', output)
        self.assertIn('Pages without citations: 1', output)
        self.assertNotIn('[Framework overview]', (self.framework / 'references.md').read_text())

    def test_unknown_keys_and_mismatched_catalogs_abort(self):
        for target in ('#ref-missing', '../references.md#missing'):
            self.write('tools/README.md', f'[Missing]({target})\n')
            before = self.snapshot()
            code, output = self.run_sync()
            self.assertEqual(code, 1)
            self.assertIn('unknown reference key: missing', output)
            self.assertEqual(self.snapshot(), before)
        self.write('tools/README.md', '## References\n\n- [Bad](#ref-missing)\n')
        self.assertIn('unknown reference key', self.run_sync()[1])
        path = self.framework / 'references_ZH.md'
        path.write_text(path.read_text().replace('id="alpha"', 'id="different"'))
        before = self.snapshot()
        self.assertIn('reference key sets differ', self.run_sync()[1])
        self.assertEqual(self.snapshot(), before)

    def test_duplicate_catalog_anchor_is_an_error(self):
        path = self.framework / 'references.md'
        path.write_text(path.read_text() + '\n<a id="beta"></a>\n\n**Duplicate.**\n')
        self.assertIn('duplicate reference key', self.run_sync()[1])

    def test_code_spans_fences_images_definitions_and_navigation_are_not_citations(self):
        text = ('[Real](#ref-beta)\n'
                '`[example](../references.md#unknown)` ``[example](#ref-unknown)``\n'
                '`multiline\n[example](#ref-unknown)\n`\n'
                '```md\n[example](#ref-unknown)\n## References\n```\n'
                '~~~\n[example](#ref-unknown)\n~~~\n'
                '> ```\n> [example](#ref-unknown)\n> ```\n'
                '- ```md\n  [example](#ref-unknown)\n  ```\n'
                '![image](#ref-unknown)\n[ref]: #ref-unknown\n'
                '\\[escaped](#ref-unknown)\n[All](../references.md)\n')
        path = self.write('tools/README.md', text)
        result = refs.synchronize_page(path, text, self.catalog())
        self.assertEqual(result.keys, {'beta'})
        self.assertTrue(result.text.startswith(text))

    def test_relative_links_rebased_and_zh_paragraph_copied(self):
        source = self.framework / 'references_ZH.md'
        source.write_text(source.read_text().replace('First work.', '原文标题。 [local](notes/a.md#part).'))
        path = self.write('figures/specs/fig2_1_ZH.md', '[A](../../references_ZH.md#alpha)\n')
        result = refs.synchronize_page(path, path.read_text(), self.catalog())
        self.assertIn('## 参考文献\n\n本页引用的文献。', result.text)
        self.assertIn('[参考文献](../../references_ZH.md)', result.text)
        self.assertIn('原文标题。 [local](../../notes/a.md#part).', result.text)

    def test_used_in_items_deduplicate_sort_and_localize_non_item_pages(self):
        for number, word, zh_word in (('02-10', 'Theorem', '定理'), ('02-02', 'Lemma', '引理')):
            display = '.'.join(str(int(n)) for n in number.split('-'))
            for suffix, label in (('', word), ('_ZH', zh_word)):
                name = f'{number}_item{suffix}.md'
                self.write(f'statements/{name}', f'# {label} {display} · Title\n\n[A](#ref-alpha)\n')
                for folder in ('proofs', 'experiments'):
                    self.write(f'{folder}/{name}', '# Evidence\n\n[A](#ref-alpha)\n')
        self.write('README.md', '# Framework\n\n[A](#ref-alpha)\n')
        self.write('README_ZH.md', '# 框架\n\n[A](#ref-alpha)\n')
        self.write('figures/specs/fig2_1.md', '# Figure\n\n[A](#ref-alpha)\n')
        self.write('tools/README.md', '# Tools\n\n[A](#ref-alpha)\n')
        self.write('figures/README.md', '# Figures\n\n[A](#ref-alpha)\n')
        code, output = self.run_sync()
        self.assertEqual(code, 0, output)
        en = (self.framework / 'references.md').read_text()
        zh = (self.framework / 'references_ZH.md').read_text()
        self.assertIn('Used in: [Lemma 2.2](statements/02-02_item.md), [Theorem 2.10](statements/02-10_item.md), '
                      '[Framework overview](README.md), [Figure overview](figures/README.md), '
                      '[Figure 2.1 spec](figures/specs/fig2_1.md), [Validation tools](tools/README.md)', en)
        self.assertIn('引用于：[引理 2.2](statements/02-02_item_ZH.md), [定理 2.10](statements/02-10_item_ZH.md)', zh)
        self.assertIn('[框架总览](README_ZH.md)', zh)
        self.assertIn('[图 2.1 说明](figures/specs/fig2_1.md)', zh)
        self.assertIn('[校验工具](tools/README.md)', zh)
        self.assertIn('Header remains.', en)
        self.assertIn('Used in: none', en)
        self.assertIn('引用于：无', zh)
        self.assertEqual(self.run_sync('--check')[0], 0)

    def test_header_update_preserves_other_sentences(self):
        for name, old in (('references.md', '"Used in" lists the framework items whose statement, proof or experiments page cites the work; the links go to the statement pages.'),
                          ('references_ZH.md', '"引用于"列出在 statement、proof 或 experiments 页中引用该文献的框架条目，链接指向 statement 页。')):
            path = self.framework / name
            path.write_text(path.read_text().replace(refs.HEADER_SENTENCES[refs.md.language(path)], old))
        self.assertEqual(self.run_sync()[0], 0)
        for name in ('references.md', 'references_ZH.md'):
            text = (self.framework / name).read_text()
            self.assertIn('Header remains. ' + refs.HEADER_SENTENCES[refs.md.language(Path(name))], text)

    def test_selected_pages_only_but_usage_reads_whole_framework(self):
        first = self.write('tools/README.md', '[A](../references.md#alpha)\n')
        second = self.write('README.md', '[B](references.md#beta)\n')
        original = second.read_text()
        code, output = self.run_sync('theoretical_framework/tools')
        self.assertEqual(code, 0, output)
        self.assertIn('#ref-alpha', first.read_text())
        self.assertEqual(second.read_text(), original)
        self.assertIn('[Framework overview](README.md)', (self.framework / 'references.md').read_text())

    def test_duplicate_or_nonfinal_reference_sections_abort(self):
        for tail in ('## Later\n', '## References\n'):
            self.write('README.md', '[A](#ref-alpha)\n\n## References\n\n- [A](#ref-alpha)\n\n' + tail)
            before = self.snapshot()
            self.assertIn('unique and the last H2', self.run_sync()[1])
            self.assertEqual(before, self.snapshot())

    def test_multiline_inline_links_and_encoded_fragments(self):
        path = self.write('README.md', '[wrapped\nlabel](./references.md#%61lpha)\n')
        result = refs.synchronize_page(path, path.read_text(), self.catalog())
        self.assertIn('[wrapped\nlabel](#ref-alpha)', result.text)
        self.assertEqual(result.keys, {'alpha'})

    def test_similar_heading_with_inline_code_is_not_removed(self):
        text = '# Tools\n\n## References `notes`\n\nKeep these notes. [A](#ref-alpha)\n'
        path = self.write('tools/README.md', text)
        result = refs.synchronize_page(path, text, self.catalog())
        self.assertTrue(result.text.startswith(text))
        self.assertIn('## References\n\nWorks cited', result.text)

    def test_duplicate_usage_lines_are_replaced_with_one_line(self):
        for name, line in (('references.md', 'Used in: stale'), ('references_ZH.md', '引用于：旧')):
            path = self.framework / name
            path.write_text(path.read_text() + '\n' + line + '\n')
        self.assertEqual(self.run_sync()[0], 0)
        self.assertEqual((self.framework / 'references.md').read_text().count('Used in:'), 2)
        self.assertEqual((self.framework / 'references_ZH.md').read_text().count('引用于：'), 2)
        self.assertEqual(self.run_sync('--check')[0], 0)

    def test_copied_entry_preserves_trailing_spaces_verbatim(self):
        source = self.framework / 'references.md'
        source.write_text(source.read_text().replace('https://example.org/b).', 'https://example.org/b).  '))
        path = self.page()
        section = refs.generate_section(path, {'beta'}, self.catalog())
        self.assertTrue(section.endswith('b).  \n'))
        self.assertTrue(path.read_text().endswith(section))
        result = refs.synchronize_page(path, path.read_text(), self.catalog())
        self.assertEqual(result.text, path.read_text())


if __name__ == '__main__':
    unittest.main()
