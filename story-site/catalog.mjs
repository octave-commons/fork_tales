// SPDX-License-Identifier: GPL-3.0-or-later

import path from 'node:path';

export const siteConfig = Object.freeze({
  title: 'Fork Tales',
  subtitle: 'An archaeological reader',
  sourceRepository: 'octave-commons/fork_tales',
  sourceBranch: 'main',
  creativeLicense: 'CC BY-SA 4.0',
  siteLicense: 'GPL-3.0-or-later',
});

export const collections = Object.freeze([
  {
    id: 'narrative',
    label: 'Gates of Truth',
    description: 'Numbered chapter witnesses, presented in source order without repairing gaps.',
  },
  {
    id: 'assemblies',
    label: 'Manuscript assemblies',
    description: 'Compiled manuscript witnesses. Assembly does not imply canonical completeness.',
  },
  {
    id: 'characters',
    label: 'Characters',
    description: 'Character bibles, identity records, and persona witnesses.',
  },
  {
    id: 'world',
    label: 'World',
    description: 'World bibles and setting records selected from the mixed repository.',
  },
  {
    id: 'myths',
    label: 'Myths',
    description: 'Mythic records kept distinct from manuscript chapters and operational analysis.',
  },
  {
    id: 'color-of-consequence',
    label: 'Color of Consequence',
    description: 'A related creative stratum recovered beside Gates of Truth.',
  },
  {
    id: 'songs',
    label: 'Songs & artifacts',
    description: 'Lyrics, choir records, dialogue fragments, and selected creative artifacts.',
  },
]);

const markdown = (value) => value.toLowerCase().endsWith('.md');
const artifact = (value) => /\.(?:ustx|mid|midi)$/i.test(value);

const includeRules = Object.freeze([
  {
    id: 'chapter-witness',
    collection: 'narrative',
    status: 'chapter-witness',
    kind: 'markdown',
    match: (value) => value.startsWith('narrative/') && markdown(value),
  },
  {
    id: 'manuscript-full-witness',
    collection: 'assemblies',
    status: 'assembly-witness',
    kind: 'markdown',
    match: (value) => value === 'MANUSCRIPT_FULL.md',
  },
  {
    id: 'gates-doc-witness',
    collection: 'assemblies',
    status: 'assembly-witness',
    kind: 'markdown',
    match: (value) => value === 'docs/gates_of_truth.md',
  },
  {
    id: 'character-witness',
    collection: 'characters',
    status: 'reference-witness',
    kind: 'markdown',
    match: (value) => value.startsWith('world_building/characters/') && markdown(value),
  },
  {
    id: 'world-bible-witness',
    collection: 'world',
    status: 'reference-witness',
    kind: 'markdown',
    match: (value) => value.startsWith('world_building/bible/') && markdown(value),
  },
  {
    id: 'myth-witness',
    collection: 'myths',
    status: 'myth-witness',
    kind: 'markdown',
    match: (value) => value.startsWith('world_building/myth/') && markdown(value),
  },
  {
    id: 'color-of-consequence-witness',
    collection: 'color-of-consequence',
    status: 'related-work-witness',
    kind: 'markdown',
    match: (value) => value.startsWith('world_building/color_of_consequence/') && markdown(value),
  },
  {
    id: 'song-witness',
    collection: 'songs',
    status: 'creative-witness',
    kind: 'markdown',
    match: (value) => value.startsWith('world_building/songs/') && markdown(value),
  },
  {
    id: 'song-project-reference',
    collection: 'songs',
    status: 'project-reference',
    kind: 'artifact',
    match: (value) => value.startsWith('world_building/songs/') && artifact(value),
  },
  {
    id: 'live-choir-witness',
    collection: 'songs',
    status: 'creative-witness',
    kind: 'markdown',
    match: (value) => value === 'LIVE_CHOIR.md',
  },
  {
    id: 'selected-doc-artifact',
    collection: 'songs',
    status: 'creative-witness',
    kind: 'markdown',
    match: (value) => {
      if (!value.startsWith('docs/') || !markdown(value)) return false;
      const name = path.posix.basename(value);
      return /^(?:artifact|dialog|new_lyrics)[_-]/i.test(name) || /_announcement(?:_|\.)/i.test(name);
    },
  },
]);

const holdRules = Object.freeze([
  {
    id: 'generated-analysis',
    disposition: 'held-out',
    reason: 'Generated analysis and unreviewed suggestions are evidence, not published canon.',
    match: (value) => value.startsWith('world_building/analysis/'),
  },
  {
    id: 'process-metadata',
    disposition: 'held-out',
    reason: 'Process metadata belongs to the archaeological record, not the public reading sequence.',
    match: (value) => value.startsWith('world_building/meta/'),
  },
  {
    id: 'workbench-notes',
    disposition: 'review-needed',
    reason: 'Workbench notes require human classification before publication.',
    match: (value) => value.startsWith('world_building/notes/'),
  },
  {
    id: 'unclassified-world-material',
    disposition: 'review-needed',
    reason: 'Miscellaneous world material remains visible in the boundary ledger until classified.',
    match: (value) => value.startsWith('world_building/misc/'),
  },
  {
    id: 'mixed-docs',
    disposition: 'review-needed',
    reason: 'The docs tree mixes creative and technical records; only explicit creative witnesses publish.',
    match: (value) => value.startsWith('docs/'),
  },
  {
    id: 'external-audio',
    disposition: 'external',
    reason: 'Large audio remains outside the static build; the archive records the boundary without copying bytes.',
    match: (value) => value.startsWith('narrative_audio_v3/'),
  },
  {
    id: 'unclassified-world-building',
    disposition: 'review-needed',
    reason: 'Unmatched world-building material is retained for review rather than silently published.',
    match: (value) => value.startsWith('world_building/'),
  },
]);

export const archaeologyAnchors = Object.freeze([
  {
    id: 'reconstitution',
    label: 'Story reconstitution',
    repository: 'octave-commons/fork_tales',
    revision: 'be1fb21380533c186157dc5a8a63fbfe7b69a791',
    path: '',
    relation: 'A single import commit reconstituted Gates of Truth and Color of Consequence artifacts.',
  },
  {
    id: 'devel-lore',
    label: 'Separated lore in devel',
    repository: 'riatzukiza/devel',
    revision: '80a95e5638f4ee95e182ebf0a22f4735ab55964f',
    path: 'Lore/fork-tales',
    relation: 'The superproject preserved a corpus organized as characters, creative works, events, and world records.',
  },
  {
    id: 'gates-of-aker-reader',
    label: 'Recovered Fork Tales reader',
    repository: 'octave-commons/gates-of-aker',
    revision: '8578d5d4f767addf0876043a0a5964cea793acba',
    path: 'web/src/components/ForkTalesPanel.tsx',
    relation: 'The exact precedent survives as a chapter-history reader with status, source paths, selection, and adjacent continuation controls. This site keeps the reading grammar while removing public mutation.',
  },
  {
    id: 'world-console',
    label: 'Current world console',
    repository: 'octave-commons/fork_tales',
    revision: 'c6357ebf6126114792f343301e18f5ca10c4c016',
    path: 'part64/frontend',
    relation: 'The current React interface is a simulation/operator console, distinct from the recovered public reader.',
  },
  {
    id: 'code-witness',
    label: 'Code-only witness',
    repository: 'riatzukiza/fork_tales',
    revision: '7d31f048878f2c372fda8397672e6c55e0bdb617',
    path: '',
    relation: 'A descendant fork preserves the runtime and dashboard without the story trees.',
  },
  {
    id: 'static-shell',
    label: 'Older static interface lineage',
    repository: 'riatzukiza/riatzukiza.github.io',
    revision: 'c61d2e9b7aa23ee4e533167ee0e87e7c43055fc3',
    path: '',
    relation: 'An older static-site shell demonstrates publication intent but contains no Fork Tales corpus.',
  },
]);

export function normalizeSourcePath(value) {
  return value.replaceAll('\\', '/').replace(/^\.\//, '');
}

export function classifyPath(input) {
  const value = normalizeSourcePath(input);
  const include = includeRules.find((rule) => rule.match(value));
  if (include) {
    return {
      action: 'include',
      rule: include.id,
      collection: include.collection,
      status: include.status,
      kind: include.kind,
    };
  }

  const hold = holdRules.find((rule) => rule.match(value));
  if (hold) {
    return {
      action: 'hold',
      rule: hold.id,
      disposition: hold.disposition,
      reason: hold.reason,
    };
  }

  return { action: 'ignore' };
}

export const candidateRoots = Object.freeze([
  'narrative',
  'world_building',
  'docs',
  'narrative_audio_v3',
]);

export const candidateRootFiles = Object.freeze([
  'MANUSCRIPT_FULL.md',
  'LIVE_CHOIR.md',
]);
