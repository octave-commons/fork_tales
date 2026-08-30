# Fork Tales story site

A deterministic, provenance-aware static reader for the creative strata currently mixed into `octave-commons/fork_tales`.

The site is intentionally a **projection before extraction**:

- source files remain untouched;
- an explicit classifier decides what publishes;
- held-out and review-needed paths remain visible in `archive.json`;
- chapter gaps and byte-identical witnesses are reported, not repaired;
- every page links back to its repository path, revision, and SHA-256 digest;
- large media is not copied into the Pages artifact.

This makes the eventual corpus-repository split reviewable and reversible.

## Build

Requires Node.js 22 or later and no third-party packages.

```sh
npm --prefix story-site test
SOURCE_REF="$(git rev-parse HEAD)" npm --prefix story-site run build
```

The generated site is written to `story-site/dist/`.

To preview it locally:

```sh
python -m http.server 4173 --directory story-site/dist
```

## Publication boundary

Published by default:

- `narrative/**/*.md`
- `MANUSCRIPT_FULL.md`
- `docs/gates_of_truth.md`
- selected `world_building/{bible,characters,myth,color_of_consequence,songs}/**`
- `LIVE_CHOIR.md`
- explicitly named creative artifacts in `docs/`

Held or review-needed by default:

- generated `world_building/analysis/`
- process metadata in `world_building/meta/`
- workbench material in `world_building/{notes,misc}/`
- unmatched mixed-purpose `docs/`
- large audio bytes

Change `catalog.mjs` to change the boundary. Do not broaden it by filename extension alone.

## Licensing

The generator, styles, scripts, tests, and workflow are released under GNU GPL v3 or later. Creative works retain the repository's CC BY-SA 4.0 terms.
