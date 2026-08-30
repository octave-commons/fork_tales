// SPDX-License-Identifier: GPL-3.0-or-later

import assert from 'node:assert/strict';
import { promises as fs } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import test from 'node:test';
import { classifyPath } from '../catalog.mjs';
import { buildArchive, buildSite, renderMarkdown, toPublicArchive } from '../lib.mjs';

async function writeFixture(root, relativePath, content) {
  const destination = path.join(root, ...relativePath.split('/'));
  await fs.mkdir(path.dirname(destination), { recursive: true });
  await fs.writeFile(destination, content);
}

async function makeFixture() {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), 'fork-tales-story-site-'));
  await writeFixture(root, 'narrative/Chapter_01_Chapter_1.md', '# Gates of Truth\n\n## Chapter 1 — First Witness\n\nThe archive begins.\n');
  await writeFixture(root, 'narrative/Chapter_03_Chapter_3.md', '# Gates of Truth\n\n## Chapter 3 — Unsafe Input\n\n<script>alert("no")</script>\n');
  const duplicate = '# Gates of Truth\n\n## Chapter 4 — Same Bytes\n\nRemoving the error changes the checksum.\n';
  await writeFixture(root, 'narrative/Chapter_04_Chapter_4.md', duplicate);
  await writeFixture(root, 'narrative/Chapter_05_Chapter_5.md', duplicate);
  await writeFixture(root, 'MANUSCRIPT_FULL.md', '# Gates of Truth — Assembly Witness\n\nNot declared complete.\n');
  await writeFixture(root, 'LIVE_CHOIR.md', '# Live Choir\n\nA chorus remains addressable.\n');
  await writeFixture(root, 'world_building/bible/fork-keys.md', '# Fork-Keys\n\nA world record.\n');
  await writeFixture(root, 'docs/ARTIFACT_memory.md', '# Memory Artifact\n\nA selected creative witness.\n');
  await writeFixture(root, 'world_building/analysis/unreviewed.jsonl', '{"status":"unreviewed"}\n');
  await writeFixture(root, 'world_building/notes/scratch.md', '# Scratch\n\nDo not publish automatically.\n');
  await writeFixture(root, 'docs/TECHNICAL_RUNBOOK.md', '# Runbook\n\nOperational material.\n');
  return root;
}

test('classification is explicit and conservative', () => {
  assert.deepEqual(classifyPath('narrative/Chapter_01.md'), {
    action: 'include',
    rule: 'chapter-witness',
    collection: 'narrative',
    status: 'chapter-witness',
    kind: 'markdown',
  });
  assert.equal(classifyPath('world_building/analysis/run.jsonl').disposition, 'held-out');
  assert.equal(classifyPath('world_building/notes/private.md').disposition, 'review-needed');
  assert.equal(classifyPath('docs/TECHNICAL.md').action, 'hold');
  assert.equal(classifyPath('docs/GATES_OF_TRUTH_ANNOUNCEMENT.md').action, 'include');
  assert.equal(classifyPath('part64/frontend/src/App.tsx').action, 'ignore');
});

test('archive preserves sequence gaps, duplicate witnesses, and held boundaries', async (context) => {
  const root = await makeFixture();
  context.after(() => fs.rm(root, { recursive: true, force: true }));

  const archive = await buildArchive({
    rootDir: root,
    sourceRevision: 'fixture-revision',
  });

  assert.equal(archive.entries.length, 8);
  assert.deepEqual(archive.integrity.missingSequences, [2]);
  assert.equal(archive.integrity.exactDuplicates.length, 1);
  assert.equal(archive.integrity.exactDuplicates[0].entries.length, 2);
  assert.ok(archive.boundary.some((item) => item.sourcePath === 'world_building/analysis/unreviewed.jsonl'));
  assert.ok(archive.boundary.some((item) => item.sourcePath === 'docs/TECHNICAL_RUNBOOK.md'));
  assert.ok(archive.archiveDigest.match(/^[a-f0-9]{64}$/));

  const publicArchive = toPublicArchive(archive);
  assert.ok(publicArchive.entries.every((entry) => !Object.hasOwn(entry, 'content')));
  assert.ok(publicArchive.entries.every((entry) => !Object.hasOwn(entry, 'searchText')));
});

test('markdown renderer escapes raw HTML and rejects script URLs', () => {
  const output = renderMarkdown([
    '# Safe heading',
    '',
    '<script>alert("no")</script>',
    '',
    '[bad](javascript:alert(1)) and [good](https://example.com).',
    '',
    '```js',
    '<tag>',
    '```',
  ].join('\n'));

  assert.ok(output.includes('&lt;script&gt;'));
  assert.ok(!output.includes('<script>'));
  assert.ok(!output.includes('href="javascript:'));
  assert.ok(output.includes('href="https://example.com"'));
  assert.ok(output.includes('&lt;tag&gt;'));
});

test('site build is deterministic and emits readable entry pages', async (context) => {
  const root = await makeFixture();
  const outA = path.join(root, '.out-a');
  const outB = path.join(root, '.out-b');
  context.after(() => fs.rm(root, { recursive: true, force: true }));

  const first = await buildSite({ rootDir: root, outputDir: outA, sourceRevision: 'fixture-revision' });
  const second = await buildSite({ rootDir: root, outputDir: outB, sourceRevision: 'fixture-revision' });

  assert.equal(first.archiveDigest, second.archiveDigest);
  assert.equal(await fs.readFile(path.join(outA, 'archive.json'), 'utf8'), await fs.readFile(path.join(outB, 'archive.json'), 'utf8'));
  assert.equal(await fs.readFile(path.join(outA, 'index.html'), 'utf8'), await fs.readFile(path.join(outB, 'index.html'), 'utf8'));

  const unsafeEntry = first.entries.find((entry) => entry.title.includes('Unsafe Input'));
  const entryHtml = await fs.readFile(path.join(outA, ...unsafeEntry.outputPath.split('/')), 'utf8');
  assert.ok(entryHtml.includes('&lt;script&gt;'));
  assert.ok(entryHtml.includes('SHA-256'));
  const searchIndex = JSON.parse(await fs.readFile(path.join(outA, 'search-index.json'), 'utf8'));
  assert.ok(searchIndex.some((record) => record.text.includes('archive begins')));
  assert.ok(await fs.stat(path.join(outA, 'assets', 'styles.css')));
  assert.ok(await fs.stat(path.join(outA, 'archaeology.html')));
});
