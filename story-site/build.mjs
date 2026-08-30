#!/usr/bin/env node
// SPDX-License-Identifier: GPL-3.0-or-later

import path from 'node:path';
import process from 'node:process';
import { buildSite } from './lib.mjs';

function parseArguments(argv) {
  const options = new Map();
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index];
    if (!token.startsWith('--')) throw new Error(`Unexpected argument: ${token}`);
    const value = argv[index + 1];
    if (!value || value.startsWith('--')) throw new Error(`Missing value for ${token}`);
    options.set(token.slice(2), value);
    index += 1;
  }
  return options;
}

try {
  const args = parseArguments(process.argv.slice(2));
  const rootDir = path.resolve(args.get('root') ?? '..');
  const outputDir = path.resolve(args.get('out') ?? 'dist');
  const archive = await buildSite({
    rootDir,
    outputDir,
    sourceRepository: args.get('repository'),
    sourceRevision: args.get('revision'),
  });

  process.stdout.write([
    `Built ${archive.entries.length} published witnesses.`,
    `Held ${archive.boundary.length} boundary records.`,
    `Archive digest: ${archive.archiveDigest}`,
    `Output: ${outputDir}`,
  ].join('\n') + '\n');
} catch (error) {
  process.stderr.write(`${error.stack ?? error.message}\n`);
  process.exitCode = 1;
}
