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
    <div class="section-heading"><p class="eyebrow">Recovered lineage</p><h2 id="strata-heading">Five anchors survived the search</h2></div>
    <ol class="strata-list">${archive.archaeologyAnchors.map((anchor, index) => `<li>
      <span class="stratum-number">${String(index + 1).padStart(2, '0')}</span>
      <div><h3>${escapeHtml(anchor.label)}</h3><p>${escapeHtml(anchor.relation)}</p><a href="${escapeAttribute(anchor.url)}"><code>${escapeHtml(anchor.repository)}${anchor.path ? `/${escapeHtml(anchor.path)}` : ''}@${escapeHtml(anchor.revision.slice(0, 12))}</code></a></div>
    </li>`).join('')}</ol>
  </section>

  <section class="archaeology-grid">
    <article><p class="eyebrow">Recovered intent</p><h2>The corpus wanted its own address.</h2><p>The devel superproject later organized Fork Tales lore into characters, creative works, events, and world records. A descendant fork preserved runtime code without the story trees. Together they reveal a separation already trying to happen.</p></article>
    <article><p class="eyebrow">Rejected shortcut</p><h2>The old frontend is not the reader.</h2><p>The existing React application is a world-simulation and operator console. Its existence proves interface intent, but reusing its monolith would preserve the original entanglement.</p></article>
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

