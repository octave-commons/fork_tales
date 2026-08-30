// SPDX-License-Identifier: GPL-3.0-or-later

import { siteConfig } from '../catalog.mjs';
import { escapeAttribute, escapeHtml } from './utils.mjs';

function relativeAssetPrefix(outputPath) {
  const depth = outputPath.split('/').length - 1;
  return depth === 0 ? '' : '../'.repeat(depth);
}

export function renderShell({ title, body, outputPath = 'index.html', description = siteConfig.subtitle, pageClass = '' }) {
  const prefix = relativeAssetPrefix(outputPath);
  return `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="${escapeAttribute(description)}">
  <meta name="color-scheme" content="dark light">
  <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 64 64%27%3E%3Crect width=%2764%27 height=%2764%27 fill=%27%23141716%27/%3E%3Ctext x=%2732%27 y=%2741%27 text-anchor=%27middle%27 font-size=%2725%27 fill=%27%23d89c70%27%3E%CE%B7%CE%BC%3C/text%3E%3C/svg%3E">
  <title>${escapeHtml(title)} · ${escapeHtml(siteConfig.title)}</title>
  <link rel="stylesheet" href="${prefix}assets/styles.css">
  <script src="${prefix}assets/app.js" defer></script>
</head>
<body class="${escapeAttribute(pageClass)}">
  <a class="skip-link" href="#main">Skip to content</a>
  <header class="site-header">
    <a class="site-mark" href="${prefix}index.html" aria-label="Fork Tales home">
      <span class="site-glyph" aria-hidden="true">ημ</span>
      <span><strong>${escapeHtml(siteConfig.title)}</strong><small>${escapeHtml(siteConfig.subtitle)}</small></span>
    </a>
    <nav aria-label="Primary">
      <a href="${prefix}index.html#archive">Archive</a>
      <a href="${prefix}archaeology.html">Archaeology</a>
      <a href="${prefix}archive.json">Manifest</a>
    </nav>
  </header>
  <main id="main">${body}</main>
  <footer class="site-footer">
    <p>Creative witnesses: ${escapeHtml(siteConfig.creativeLicense)}. Reader code: ${escapeHtml(siteConfig.siteLicense)}.</p>
    <p>The archive is not the person. Missing data is not permission to repair.</p>
  </footer>
</body>
</html>`;
}
