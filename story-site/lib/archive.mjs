// SPDX-License-Identifier: GPL-3.0-or-later

import { promises as fs } from 'node:fs';
import path from 'node:path';
import {
  archaeologyAnchors,
  candidateRootFiles,
  candidateRoots,
  classifyPath,
  collections,
  normalizeSourcePath,
  siteConfig,
} from '../catalog.mjs';
import {
  normalizeWhitespace,
  repositoryUrl,
  sha256,
  sourceUrl,
  stripMarkdown,
} from './utils.mjs';

const SKIP_DIRECTORIES = new Set(['.git', 'node_modules', 'dist', '.cache']);

function headingsFromMarkdown(markdown) {
  return markdown
    .split(/\r?\n/)
    .map((line) => line.match(/^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$/))
    .filter(Boolean)
    .map((match) => ({ level: match[1].length, text: stripMarkdown(match[2]) }));
}

export function extractTitle(markdown, sourcePath) {
  const headings = headingsFromMarkdown(markdown);
  if (headings.length === 0) {
    return path.posix.basename(sourcePath, path.posix.extname(sourcePath)).replaceAll(/[_-]+/g, ' ');
  }

  if (sourcePath.startsWith('narrative/')) {
    const chapterHeading = headings.find((heading) => /^chapter\s+\d+/i.test(heading.text));
    if (chapterHeading) return chapterHeading.text;
  }

  if (/^gates of truth$/i.test(headings[0].text) && headings.length > 1) {
    return headings[1].text;
  }

  return headings[0].text;
}

export function extractSequence(sourcePath, title = '') {
  const pathMatch = sourcePath.match(/(?:^|\/)Chapter_(\d+)(?:_|\.|$)/i);
  if (pathMatch) return Number.parseInt(pathMatch[1], 10);
  const titleMatch = title.match(/^Chapter\s+(\d+)/i);
  return titleMatch ? Number.parseInt(titleMatch[1], 10) : null;
}

function makeExcerpt(markdown, maxLength = 220) {
  const plain = stripMarkdown(markdown);
  if (plain.length <= maxLength) return plain;
  const cut = plain.slice(0, maxLength + 1);
  const boundary = cut.lastIndexOf(' ');
  return `${cut.slice(0, Math.max(boundary, maxLength - 30)).trim()}…`;
}

function makeSlug(sourcePath, digest) {
  const stem = sourcePath
    .replace(/\.[^.]+$/, '')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 96);
  return `${stem || 'entry'}-${digest.slice(0, 10)}`;
}

async function exists(target) {
  try {
    await fs.access(target);
    return true;
  } catch {
    return false;
  }
}

async function walkDirectory(rootDir, directory, output) {
  const absolute = path.join(rootDir, directory);
  if (!(await exists(absolute))) return;

  const dirents = await fs.readdir(absolute, { withFileTypes: true });
  dirents.sort((left, right) => left.name.localeCompare(right.name));

  for (const dirent of dirents) {
    const relative = normalizeSourcePath(path.posix.join(directory, dirent.name));
    if (dirent.isDirectory()) {
      if (!SKIP_DIRECTORIES.has(dirent.name)) await walkDirectory(rootDir, relative, output);
      continue;
    }
    if (dirent.isFile()) output.push(relative);
  }
}

export async function discoverCandidatePaths(rootDir) {
  const output = [];

  for (const filename of candidateRootFiles) {
    if (await exists(path.join(rootDir, filename))) output.push(filename);
  }
  for (const directory of candidateRoots) {
    await walkDirectory(rootDir, directory, output);
  }

  return [...new Set(output)].sort((left, right) => left.localeCompare(right));
}

function collectionOrder(collectionId) {
  const index = collections.findIndex((collection) => collection.id === collectionId);
  return index === -1 ? Number.MAX_SAFE_INTEGER : index;
}

function compareEntries(left, right) {
  const collectionDifference = collectionOrder(left.collection) - collectionOrder(right.collection);
  if (collectionDifference !== 0) return collectionDifference;

  if (left.sequence !== null || right.sequence !== null) {
    const leftSequence = left.sequence ?? Number.MAX_SAFE_INTEGER;
    const rightSequence = right.sequence ?? Number.MAX_SAFE_INTEGER;
    if (leftSequence !== rightSequence) return leftSequence - rightSequence;
  }

  return left.sourcePath.localeCompare(right.sourcePath);
}

function groupDuplicates(entries, selector) {
  const grouped = new Map();
  for (const entry of entries) {
    const key = selector(entry);
    if (!key) continue;
    const group = grouped.get(key) ?? [];
    group.push(entry.id);
    grouped.set(key, group);
  }
  return [...grouped.entries()]
    .filter(([, ids]) => ids.length > 1)
    .map(([key, ids]) => ({ key, entries: ids.sort() }))
    .sort((left, right) => left.key.localeCompare(right.key));
}

function sequenceGaps(entries) {
  const numbers = entries
    .filter((entry) => entry.collection === 'narrative' && Number.isInteger(entry.sequence))
    .map((entry) => entry.sequence)
    .sort((left, right) => left - right);

  if (numbers.length === 0) return [];
  const found = new Set(numbers);
  const gaps = [];
  for (let value = numbers[0]; value <= numbers.at(-1); value += 1) {
    if (!found.has(value)) gaps.push(value);
  }
  return gaps;
}

function archiveDigest(payload) {
  return sha256(JSON.stringify(payload));
}

export async function buildArchive({
  rootDir,
  sourceRepository = siteConfig.sourceRepository,
  sourceRevision = process.env.SOURCE_REF || process.env.GITHUB_SHA || siteConfig.sourceBranch,
} = {}) {
  if (!rootDir) throw new TypeError('rootDir is required');

  const candidatePaths = await discoverCandidatePaths(rootDir);
  const entries = [];
  const boundary = [];

  for (const sourcePath of candidatePaths) {
    const classification = classifyPath(sourcePath);
    const absolute = path.join(rootDir, ...sourcePath.split('/'));
    const stat = await fs.stat(absolute);

    if (classification.action === 'include') {
      if (classification.kind === 'markdown') {
        const content = await fs.readFile(absolute, 'utf8');
        const digest = sha256(content);
        const title = extractTitle(content, sourcePath);
        const id = makeSlug(sourcePath, digest);
        entries.push({
          id,
          title,
          collection: classification.collection,
          status: classification.status,
          kind: classification.kind,
          sourcePath,
          sourceRepository,
          sourceRevision,
          sourceUrl: sourceUrl(sourceRepository, sourceRevision, sourcePath),
          digest,
          bytes: Buffer.byteLength(content),
          sequence: extractSequence(sourcePath, title),
          excerpt: makeExcerpt(content),
          searchText: stripMarkdown(content).toLowerCase(),
          outputPath: `entry/${id}.html`,
          content,
        });
      } else {
        const digest = sha256(`${sourcePath}\0${stat.size}`);
        const id = makeSlug(sourcePath, digest);
        entries.push({
          id,
          title: path.posix.basename(sourcePath),
          collection: classification.collection,
          status: classification.status,
          kind: classification.kind,
          sourcePath,
          sourceRepository,
          sourceRevision,
          sourceUrl: sourceUrl(sourceRepository, sourceRevision, sourcePath),
          digest: null,
          bytes: stat.size,
          sequence: null,
          excerpt: 'Project artifact retained by reference; bytes are not copied into the static archive.',
          searchText: path.posix.basename(sourcePath).toLowerCase(),
          outputPath: `entry/${id}.html`,
          content: null,
        });
      }
      continue;
    }

    if (classification.action === 'hold') {
      boundary.push({
        sourcePath,
        rule: classification.rule,
        disposition: classification.disposition,
        reason: classification.reason,
        bytes: stat.size,
      });
    }
  }

  entries.sort(compareEntries);
  boundary.sort((left, right) => left.sourcePath.localeCompare(right.sourcePath));

  const exactDuplicates = groupDuplicates(
    entries.filter((entry) => entry.digest),
    (entry) => entry.digest,
  );
  const titleDuplicates = groupDuplicates(entries, (entry) => normalizeWhitespace(entry.title).toLowerCase());
  const missingSequences = sequenceGaps(entries);

  const duplicateByEntry = new Map();
  for (const group of exactDuplicates) {
    for (const id of group.entries) duplicateByEntry.set(id, group.entries.filter((candidate) => candidate !== id));
  }
  for (const entry of entries) entry.exactDuplicates = duplicateByEntry.get(entry.id) ?? [];

  const archive = {
    schema: 'fork-tales.archive/v1',
    site: siteConfig,
    source: { repository: sourceRepository, revision: sourceRevision },
    collections,
    archaeologyAnchors: archaeologyAnchors.map((anchor) => ({
      ...anchor,
      url: repositoryUrl(anchor.repository, anchor.revision, anchor.path),
    })),
    entries,
    boundary,
    integrity: {
      missingSequences,
      exactDuplicates,
      titleDuplicates,
    },
  };

  const publicArchive = toPublicArchive(archive);
  archive.archiveDigest = archiveDigest(publicArchive);
  return archive;
}

export function toPublicArchive(archive) {
  return {
    schema: archive.schema,
    site: archive.site,
    source: archive.source,
    collections: archive.collections,
    archaeologyAnchors: archive.archaeologyAnchors,
    entries: archive.entries.map(({ content: _content, searchText: _searchText, ...entry }) => entry),
    boundary: archive.boundary,
    integrity: archive.integrity,
    archiveDigest: archive.archiveDigest ?? null,
  };
}
