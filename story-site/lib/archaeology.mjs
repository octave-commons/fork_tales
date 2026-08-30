// SPDX-License-Identifier: GPL-3.0-or-later

import { escapeAttribute, escapeHtml } from './utils.mjs';
import { renderShell } from './shell.mjs';

export function renderArchaeology(archive) {
  const body = `<section class="archaeology-hero">
    <p class="eyebrow">Repository archaeology</p>
    <h1>Do not clean the fracture until it has testified.</h1>
    <p class="lede">Fork Tales began as code, simulation, story, generated analysis, audio, and process record in one address. The reader is a projection over explicit witnesses—not a declaration that every nearby artifact belongs to one canon.</p>
  </section>

  <section class="strata" aria-labelledby="strata-heading">
    <div class="section-heading"><p class="eyebrow">Recovered lineage</p><h2 id="strata-heading">${archive.archaeologyAnchors.length} anchors survived the search</h2></div>
    <ol class="strata-list">${archive.archaeologyAnchors.map((anchor, index) => `<li>
      <span class="stratum-number">${String(index + 1).padStart(2, '0')}</span>
      <div><h3>${escapeHtml(anchor.label)}</h3><p>${escapeHtml(anchor.relation)}</p><a href="${escapeAttribute(anchor.url)}"><code>${escapeHtml(anchor.repository)}${anchor.path ? `/${escapeHtml(anchor.path)}` : ''}@${escapeHtml(anchor.revision.slice(0, 12))}</code></a></div>
    </li>`).join('')}</ol>
  </section>

  <section class="archaeology-grid">
    <article><p class="eyebrow">Recovered interface</p><h2>The reader survived inside Gates of Aker.</h2><p>The public <code>/fork-tales</code> route and <code>ForkTalesPanel</code> expose chapter history, selection, status, and source paths. This static site preserves the recovered choose → read → advance grammar.</p></article>
    <article><p class="eyebrow">Rejected coupling</p><h2>Reading and mutation are different surfaces.</h2><p>The recovered panel also sat beside preview and write-next-chapter controls backed by a story engine. The public reader omits mutation. Future generation belongs to a separately authenticated studio/editor, not the publication default.</p></article>
    <article><p class="eyebrow">Publication law</p><h2>Manifest first, movement second.</h2><p>This layer leaves every source file in place, records inclusions and exclusions in <a href="archive.json">archive.json</a>, and publishes a deterministic projection. A later repository split can preserve history after the boundary has been reviewed.</p></article>
  </section>

  <section class="integrity-band archaeology-integrity">
    <div><p class="eyebrow">What remains unresolved</p><h2>The archive refuses false completion.</h2></div>
    <div class="integrity-copy"><p>Missing chapter numbers: <strong>${archive.integrity.missingSequences.length ? archive.integrity.missingSequences.join(', ') : 'none detected'}</strong>.</p><p>Exact duplicate groups: <strong>${archive.integrity.exactDuplicates.length}</strong>. Review-needed paths: <strong>${archive.boundary.filter((item) => item.disposition === 'review-needed').length}</strong>.</p></div>
  </section>`;

  return renderShell({
    title: 'Archaeology',
    body,
    description: 'The recovered repository lineage and publication boundary for Fork Tales.',
    pageClass: 'archaeology-page',
  });
}
