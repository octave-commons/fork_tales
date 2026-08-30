// SPDX-License-Identifier: GPL-3.0-or-later

import { promises as fs } from 'node:fs';
import path from 'node:path';
import { buildArchive, toPublicArchive } from './archive.mjs';
import { renderArchaeology, renderEntryPage, renderHome, renderShell } from './render.mjs';

async function writeFile(outputDir, relativePath, content) {
  const destination = path.join(outputDir, ...relativePath.split('/'));
  const resolvedOutput = path.resolve(outputDir);
  const resolvedDestination = path.resolve(destination);
  if (!resolvedDestination.startsWith(`${resolvedOutput}${path.sep}`) && resolvedDestination !== resolvedOutput) {
    throw new Error(`Refusing to write outside output directory: ${relativePath}`);
  }
  await fs.mkdir(path.dirname(destination), { recursive: true });
  await fs.writeFile(destination, content);
}

export async function buildSite({ rootDir, outputDir, sourceRepository, sourceRevision } = {}) {
  if (!rootDir || !outputDir) throw new TypeError('rootDir and outputDir are required');
  const archive = await buildArchive({ rootDir, sourceRepository, sourceRevision });

  await fs.rm(outputDir, { recursive: true, force: true });
  await fs.mkdir(outputDir, { recursive: true });

  await writeFile(outputDir, 'index.html', renderHome(archive));
  await writeFile(outputDir, 'archaeology.html', renderArchaeology(archive));
  await writeFile(outputDir, '404.html', renderShell({
    title: 'Not found',
    body: '<section class="not-found"><p class="eyebrow">Unresolved address</p><h1>This witness is not in the projection.</h1><p>The path may have moved, remained held out, or never existed.</p><a class="primary-action" href="index.html">Return to the archive</a></section>',
    description: 'Fork Tales archive path not found.',
  }));
  await writeFile(outputDir, 'archive.json', `${JSON.stringify(toPublicArchive(archive), null, 2)}\n`);
  await writeFile(outputDir, 'search-index.json', `${JSON.stringify(archive.entries.map((entry) => ({ id: entry.id, text: entry.searchText })))}\n`);
  await writeFile(outputDir, '.nojekyll', '');

  const localStatic = path.join(path.dirname(new URL(import.meta.url).pathname), '..', 'static');
  await fs.mkdir(path.join(outputDir, 'assets'), { recursive: true });
  await fs.copyFile(path.join(localStatic, 'styles.css'), path.join(outputDir, 'assets', 'styles.css'));

  await fs.copyFile(path.join(localStatic, 'app.js'), path.join(outputDir, 'assets', 'app.js'));

  for (const [index, entry] of archive.entries.entries()) {
    await writeFile(outputDir, entry.outputPath, renderEntryPage(entry, archive, index));
  }

  return archive;
}
