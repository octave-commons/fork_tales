// SPDX-License-Identifier: GPL-3.0-or-later

import path from 'node:path';
import { collections, normalizeSourcePath } from '../catalog.mjs';
import { renderMarkdown } from './markdown.mjs';
import { escapeAttribute, escapeHtml, sourceUrl } from './utils.mjs';
import { renderShell } from './shell.mjs';

function collectionMap() {
  return new Map(collections.map((collection) => [collection.id, collection]));
}

function entriesById(entries) {
  return new Map(entries.map((entry) => [entry.id, entry]));
}

function resolveEntryLink(rawLink, entry, bySourcePath) {
  if (/^(?:https?:|mailto:|#)/i.test(rawLink)) return rawLink;
  const [rawPath, fragment = ''] = rawLink.split('#', 2);
  const normalized = normalizeSourcePath(path.posix.normalize(path.posix.join(path.posix.dirname(entry.sourcePath), rawPath)));
  const target = bySourcePath.get(normalized);
  if (target) {
    const relative = path.posix.relative(path.posix.dirname(entry.outputPath), target.outputPath) || path.posix.basename(target.outputPath);
    return fragment ? `${relative}#${encodeURIComponent(fragment)}` : relative;
  }
  return sourceUrl(entry.sourceRepository, entry.sourceRevision, normalized);
}

export function renderEntryPage(entry, archive, index) {
  const byCollection = collectionMap();
  const bySourcePath = new Map(archive.entries.map((candidate) => [candidate.sourcePath, candidate]));
  const byId = entriesById(archive.entries);
  const siblings = archive.entries.filter((candidate) => candidate.collection === entry.collection);
  const siblingIndex = siblings.findIndex((candidate) => candidate.id === entry.id);
  const previous = siblingIndex > 0 ? siblings[siblingIndex - 1] : null;
  const next = siblingIndex < siblings.length - 1 ? siblings[siblingIndex + 1] : null;
  const collection = byCollection.get(entry.collection);

  const rendered = entry.kind === 'markdown'
    ? renderMarkdown(entry.content, { resolveLink: (rawLink) => resolveEntryLink(rawLink, entry, bySourcePath) })
    : `<div class="artifact-placeholder"><p>This project artifact is preserved by reference. The static reader does not copy its bytes.</p><a class="primary-action" href="${escapeAttribute(entry.sourceUrl)}">Open source artifact</a></div>`;

  const duplicateNotice = entry.exactDuplicates.length
    ? `<aside class="witness-warning"><strong>Exact duplicate witness</strong><p>This file is byte-identical to ${entry.exactDuplicates.map((id) => {
      const duplicate = byId.get(id);
      const relative = path.posix.relative(path.posix.dirname(entry.outputPath), duplicate.outputPath);
      return `<a href="${escapeAttribute(relative)}">${escapeHtml(duplicate.sourcePath)}</a>`;
    }).join(', ')}. Both addresses remain visible.</p></aside>`
    : '';

  const body = `<div class="reader-shell">
    <aside class="reader-rail">
      <a class="back-link" href="../index.html#archive">← Archive</a>
      <p class="eyebrow">${escapeHtml(collection?.label ?? entry.collection)}</p>
      <h1>${escapeHtml(entry.title)}</h1>
      <p>${escapeHtml(entry.excerpt)}</p>
      <dl class="provenance">
        <div><dt>Status</dt><dd>${escapeHtml(entry.status)}</dd></div>
        <div><dt>Source</dt><dd><a href="${escapeAttribute(entry.sourceUrl)}"><code>${escapeHtml(entry.sourcePath)}</code></a></dd></div>
        <div><dt>Revision</dt><dd><code>${escapeHtml(entry.sourceRevision.slice(0, 16))}</code></dd></div>
        <div><dt>SHA-256</dt><dd><code>${entry.digest ? escapeHtml(entry.digest) : 'not copied'}</code></dd></div>
        <div><dt>Bytes</dt><dd>${entry.bytes.toLocaleString('en-US')}</dd></div>
      </dl>
      ${duplicateNotice}
    </aside>
    <article class="reader-content" data-entry-index="${index}">${rendered}</article>
  </div>
  <nav class="reader-pagination" aria-label="Adjacent witnesses">
    ${previous ? `<a rel="prev" href="${escapeAttribute(path.posix.basename(previous.outputPath))}"><small>Previous</small><span>${escapeHtml(previous.title)}</span></a>` : '<span></span>'}
    ${next ? `<a rel="next" href="${escapeAttribute(path.posix.basename(next.outputPath))}"><small>Next</small><span>${escapeHtml(next.title)}</span></a>` : '<span></span>'}
  </nav>`;

  return renderShell({
    title: entry.title,
    body,
    outputPath: entry.outputPath,
    description: entry.excerpt,
    pageClass: 'reader-page',
  });
}

