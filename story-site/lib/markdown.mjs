// SPDX-License-Identifier: GPL-3.0-or-later

import { escapeAttribute, escapeHtml, stripMarkdown } from './utils.mjs';

function sanitizeUrl(rawUrl, resolveLink) {
  const value = rawUrl.trim();
  if (!value) return null;
  if (/^(?:https?:|mailto:|#)/i.test(value)) return value;
  if (/^(?:javascript|data|vbscript):/i.test(value)) return null;
  return resolveLink ? resolveLink(value) : value;
}

function renderEmphasis(escapedText) {
  return escapedText
    .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
    .replace(/__([^_\n]+)__/g, '<strong>$1</strong>')
    .replace(/\*([^*\n]+)\*/g, '<em>$1</em>')
    .replace(/~~([^~\n]+)~~/g, '<del>$1</del>');
}

function renderInline(source, { resolveLink } = {}) {
  const tokenPattern = /(!?\[[^\]\n]*\]\([^\n)]*\)|`[^`\n]+`)/g;
  let cursor = 0;
  let output = '';

  for (const match of source.matchAll(tokenPattern)) {
    output += renderEmphasis(escapeHtml(source.slice(cursor, match.index)));
    const token = match[0];

    if (token.startsWith('`')) {
      output += `<code>${escapeHtml(token.slice(1, -1))}</code>`;
    } else {
      const image = token.startsWith('!');
      const parts = token.match(/^!?\[([^\]]*)\]\((.*)\)$/s);
      const label = parts?.[1] ?? '';
      const rawUrl = parts?.[2] ?? '';
      const url = sanitizeUrl(rawUrl, resolveLink);
      if (!url) {
        output += renderEmphasis(escapeHtml(label));
      } else if (image) {
        output += `<img src="${escapeAttribute(url)}" alt="${escapeAttribute(label)}" loading="lazy">`;
      } else {
        output += `<a href="${escapeAttribute(url)}">${renderEmphasis(escapeHtml(label))}</a>`;
      }
    }
    cursor = match.index + token.length;
  }

  output += renderEmphasis(escapeHtml(source.slice(cursor)));
  return output;
}

function startsBlock(line) {
  return (
    /^\s*$/.test(line)
    || /^\s{0,3}#{1,6}\s+/.test(line)
    || /^\s{0,3}(?:```|~~~)/.test(line)
    || /^\s{0,3}>/.test(line)
    || /^\s{0,3}(?:[-+*]\s+|\d+[.)]\s+)/.test(line)
    || /^\s{0,3}(?:---+|___+|\*\*\*+)\s*$/.test(line)
  );
}

export function renderMarkdown(markdown, options = {}) {
  const lines = String(markdown).replaceAll('\r\n', '\n').split('\n');
  const output = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index];
    if (/^\s*$/.test(line)) {
      index += 1;
      continue;
    }

    const fence = line.match(/^\s{0,3}(```|~~~)\s*([^\s]*)\s*$/);
    if (fence) {
      const marker = fence[1];
      const language = fence[2];
      const code = [];
      index += 1;
      while (index < lines.length && !new RegExp(`^\\s{0,3}${marker}`).test(lines[index])) {
        code.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) index += 1;
      const className = language ? ` class="language-${escapeAttribute(language)}"` : '';
      output.push(`<pre><code${className}>${escapeHtml(code.join('\n'))}</code></pre>`);
      continue;
    }

    const heading = line.match(/^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$/);
    if (heading) {
      const level = heading[1].length;
      const text = stripMarkdown(heading[2]);
      const id = text.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
      output.push(`<h${level}${id ? ` id="${escapeAttribute(id)}"` : ''}>${renderInline(heading[2], options)}</h${level}>`);
      index += 1;
      continue;
    }

    if (/^\s{0,3}(?:---+|___+|\*\*\*+)\s*$/.test(line)) {
      output.push('<hr>');
      index += 1;
      continue;
    }

    if (/^\s{0,3}>/.test(line)) {
      const quote = [];
      while (index < lines.length && /^\s{0,3}>/.test(lines[index])) {
        quote.push(lines[index].replace(/^\s{0,3}>\s?/, ''));
        index += 1;
      }
      output.push(`<blockquote>${quote.map((value) => renderInline(value, options)).join('<br>')}</blockquote>`);
      continue;
    }

    const listMatch = line.match(/^\s{0,3}([-+*]|\d+[.)])\s+(.+)$/);
    if (listMatch) {
      const ordered = /^\d/.test(listMatch[1]);
      const tag = ordered ? 'ol' : 'ul';
      const items = [];
      while (index < lines.length) {
        const item = lines[index].match(/^\s{0,3}([-+*]|\d+[.)])\s+(.+)$/);
        if (!item || /^\d/.test(item[1]) !== ordered) break;
        items.push(`<li>${renderInline(item[2], options)}</li>`);
        index += 1;
      }
      output.push(`<${tag}>${items.join('')}</${tag}>`);
      continue;
    }

    const paragraph = [line.trim()];
    index += 1;
    while (index < lines.length && !startsBlock(lines[index])) {
      paragraph.push(lines[index].trim());
      index += 1;
    }
    output.push(`<p>${renderInline(paragraph.join(' '), options)}</p>`);
  }

  return output.join('\n');
}
