// SPDX-License-Identifier: GPL-3.0-or-later

import { collections } from '../catalog.mjs';
import { escapeAttribute, escapeHtml } from './utils.mjs';
import { renderShell } from './shell.mjs';

function collectionMap() {
  return new Map(collections.map((collection) => [collection.id, collection]));
}

function entriesById(entries) {
  return new Map(entries.map((entry) => [entry.id, entry]));
}

function renderEntryCard(entry, collection) {
  return `<article class="entry-card" data-entry-card data-entry-id="${escapeAttribute(entry.id)}" data-collection="${escapeAttribute(entry.collection)}" data-search="${escapeAttribute(`${entry.title} ${entry.sourcePath} ${entry.excerpt}`.toLowerCase())}">
  <div class="entry-kicker"><span>${escapeHtml(collection?.label ?? entry.collection)}</span><span>${escapeHtml(entry.status)}</span></div>
  <h3><a href="${escapeAttribute(entry.outputPath)}">${escapeHtml(entry.title)}</a></h3>
  <p>${escapeHtml(entry.excerpt)}</p>
  <footer><code>${escapeHtml(entry.sourcePath)}</code>${entry.sequence !== null ? `<span>№ ${entry.sequence}</span>` : ''}</footer>
</article>`;
}

function renderBoundary(boundary) {
  const grouped = new Map();
  for (const item of boundary) {
    const key = `${item.disposition}:${item.rule}`;
    const group = grouped.get(key) ?? { disposition: item.disposition, rule: item.rule, reason: item.reason, items: [] };
    group.items.push(item);
    grouped.set(key, group);
  }

  return [...grouped.values()]
    .sort((left, right) => left.rule.localeCompare(right.rule))
    .map((group) => `<details class="boundary-group">
      <summary><span>${escapeHtml(group.rule)}</span><strong>${group.items.length}</strong></summary>
      <p>${escapeHtml(group.reason)}</p>
      <ul>${group.items.slice(0, 120).map((item) => `<li><code>${escapeHtml(item.sourcePath)}</code></li>`).join('')}</ul>
      ${group.items.length > 120 ? `<p>${group.items.length - 120} more paths remain in <a href="archive.json">archive.json</a>.</p>` : ''}
    </details>`)
    .join('\n');
}

function renderDuplicateGroups(groups, byId) {
  if (groups.length === 0) return '<p>No byte-identical published witnesses were detected.</p>';
  return `<ul class="integrity-list">${groups.map((group) => `<li><strong>${group.entries.length} byte-identical witnesses</strong><span>${group.entries.map((id) => {
    const entry = byId.get(id);
    return `<a href="${escapeAttribute(entry.outputPath)}">${escapeHtml(entry.sourcePath)}</a>`;
  }).join(' · ')}</span></li>`).join('')}</ul>`;
}

export function renderHome(archive) {
  const byCollection = collectionMap();
  const byId = entriesById(archive.entries);
  const publishedCollections = collections.filter((collection) => archive.entries.some((entry) => entry.collection === collection.id));
  const missing = archive.integrity.missingSequences;
  const cards = archive.entries.map((entry) => renderEntryCard(entry, byCollection.get(entry.collection))).join('\n');

  const body = `<section class="hero">
    <p class="eyebrow">Recovered from a mixed code-and-story repository</p>
    <h1>The story was never one file.</h1>
    <p class="lede">This reader exposes manuscript chapters, world records, character witnesses, and songs without pretending the repository already settled their canon.</p>
    <div class="hero-actions"><a class="primary-action" href="#archive">Enter the archive</a><a href="archaeology.html">Read the excavation record</a></div>
    <dl class="stats">
      <div><dt>Published witnesses</dt><dd>${archive.entries.length}</dd></div>
      <div><dt>Collections</dt><dd>${publishedCollections.length}</dd></div>
      <div><dt>Held boundaries</dt><dd>${archive.boundary.length}</dd></div>
      <div><dt>Archive digest</dt><dd><code>${escapeHtml(archive.archiveDigest.slice(0, 12))}</code></dd></div>
    </dl>
  </section>

  <section class="integrity-band" aria-labelledby="integrity-heading">
    <div><p class="eyebrow">Integrity before coherence</p><h2 id="integrity-heading">The fractures stay visible.</h2></div>
    <div class="integrity-copy">
      <p>${missing.length ? `The numbered narrative has missing sequence position${missing.length === 1 ? '' : 's'}: <strong>${missing.join(', ')}</strong>.` : 'No missing positions were detected in the numbered narrative range.'}</p>
      <p>${archive.integrity.exactDuplicates.length} exact duplicate group${archive.integrity.exactDuplicates.length === 1 ? '' : 's'} detected. The reader links witnesses; it does not silently collapse them.</p>
    </div>
  </section>

  <section class="collection-section" aria-labelledby="collections-heading">
    <div class="section-heading"><p class="eyebrow">Reading paths</p><h2 id="collections-heading">Collections, not a repaired timeline</h2></div>
    <div class="collection-grid">${publishedCollections.map((collection) => {
      const count = archive.entries.filter((entry) => entry.collection === collection.id).length;
      return `<button class="collection-card" type="button" data-collection-button="${escapeAttribute(collection.id)}"><span>${escapeHtml(collection.label)}</span><strong>${count}</strong><small>${escapeHtml(collection.description)}</small></button>`;
    }).join('')}</div>
  </section>

  <section id="archive" class="archive-section" aria-labelledby="archive-heading">
    <div class="section-heading archive-heading"><div><p class="eyebrow">Source-addressable archive</p><h2 id="archive-heading">Witnesses</h2></div><p id="result-count" aria-live="polite">${archive.entries.length} entries</p></div>
    <div class="archive-controls">
      <label><span>Search title, path, or text</span><input id="archive-search" type="search" placeholder="Search the archive…" autocomplete="off"></label>
      <label><span>Collection</span><select id="collection-filter"><option value="all">All collections</option>${publishedCollections.map((collection) => `<option value="${escapeAttribute(collection.id)}">${escapeHtml(collection.label)}</option>`).join('')}</select></label>
      <button id="clear-filters" type="button">Clear</button>
    </div>
    <div class="entry-grid" id="entry-grid">${cards}</div>
    <p class="empty-state" id="empty-state" hidden>No witness matches that view.</p>
  </section>

  <section class="ledger-section" aria-labelledby="ledger-heading">
    <div class="section-heading"><p class="eyebrow">Boundary ledger</p><h2 id="ledger-heading">Known, retained, not promoted</h2></div>
    <p class="section-intro">These paths were found in story-adjacent strata but remain held out or review-needed. Their absence from the reader is an explicit decision, not disappearance.</p>
    <div class="boundary-list">${renderBoundary(archive.boundary)}</div>
  </section>

  <section class="duplicate-section" aria-labelledby="duplicates-heading">
    <div class="section-heading"><p class="eyebrow">Duplicate witnesses</p><h2 id="duplicates-heading">Same bytes, different addresses</h2></div>
    ${renderDuplicateGroups(archive.integrity.exactDuplicates, byId)}
  </section>`;

  return renderShell({ title: 'Archive', body, description: 'A provenance-aware reader for the Fork Tales story corpus.' });
}

