// SPDX-License-Identifier: GPL-3.0-or-later

import { createHash } from 'node:crypto';

export function escapeHtml(value) {
  return String(value)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

export function escapeAttribute(value) {
  return escapeHtml(value).replaceAll('`', '&#96;');
}

export function sha256(value) {
  return createHash('sha256').update(value).digest('hex');
}

export function normalizeWhitespace(value) {
  return value.replace(/\s+/g, ' ').trim();
}

export function stripMarkdown(value) {
  return normalizeWhitespace(
    value
      .replace(/```[\s\S]*?```/g, ' ')
      .replace(/`([^`]+)`/g, '$1')
      .replace(/!\[([^\]]*)\]\([^)]*\)/g, '$1')
      .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
      .replace(/^#{1,6}\s+/gm, '')
      .replace(/^>\s?/gm, '')
      .replace(/[*_~]/g, '')
      .replace(/^\s*(?:[-+*]|\d+[.)])\s+/gm, ''),
  );
}

export function sourceUrl(repository, revision, sourcePath) {
  const encoded = sourcePath.split('/').map(encodeURIComponent).join('/');
  return `https://github.com/${repository}/blob/${encodeURIComponent(revision)}/${encoded}`;
}

export function repositoryUrl(repository, revision, sourcePath = '') {
  const base = `https://github.com/${repository}/tree/${encodeURIComponent(revision)}`;
  if (!sourcePath) return base;
  return `${base}/${sourcePath.split('/').map(encodeURIComponent).join('/')}`;
}
