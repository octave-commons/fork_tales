# Fork Tales story extraction: archaeology record

## Question

What was the story corpus trying to become before code, generated analysis, simulation state, and creative work accumulated at one repository address?

## Recovered anchors

1. **Mixed source:** `octave-commons/fork_tales` contains the runtime and dashboard beside manuscript chapters, world-building records, songs, and audio references.
2. **Reconstitution commit:** `be1fb21380533c186157dc5a8a63fbfe7b69a791` imported Gates of Truth and Color of Consequence material in a large recovery operation. Some adjacent analysis records explicitly describe themselves as unreviewed suggestions.
3. **Separated lore:** `riatzukiza/devel` at `80a95e5638f4ee95e182ebf0a22f4735ab55964f` tracked Fork Tales, Gates of Aker, the separated `Lore/fork-tales` corpus, and a local-only Mythloom recovery lead. The surviving lore is organized as `characters/`, `creative/`, `events/`, and `world/`.
4. **Recovered reader:** `octave-commons/gates-of-aker` at `8578d5d4f767addf0876043a0a5964cea793acba` contains `web/src/components/ForkTalesPanel.tsx` and a public `/fork-tales` route. The panel exposes chapter status and history, selects and reads individual chapters, retains source paths, and sits beside preview/write-next-chapter controls.
5. **Code-only witness:** tree `7d31f048878f2c372fda8397672e6c55e0bdb617` in `riatzukiza/fork_tales` preserves runtime/dashboard strata without the manuscript and world-building trees.
6. **Current world interface:** `part64/frontend` is a substantial React/Vite simulation console. It is a different interface lineage from the recovered Gates of Aker reader.
7. **Older static shell:** `riatzukiza/riatzukiza.github.io` demonstrates an older publication surface, but no Fork Tales corpus was found there.

## Recovered intent

The public reading surface was not hypothetical. Gates of Aker already implemented the core interaction grammar: choose a chapter, read it with its status and source path visible, then move through chapter history.

That reader was coupled to a backend story engine and adjacent continuation controls. The recoverable design law is therefore not “restore the old application unchanged.” It is:

- preserve **choose → read → advance** as the public grammar;
- preserve source identity and archive damage as visible evidence;
- move generation, continuation, and corpus mutation to a separately authenticated studio/editor surface;
- let the public site consume a deterministic manifest rather than a neighboring mutable filesystem.

The corpus also wanted a stable address distinct from its interpreter/runtime. The website should therefore read a manifest of creative witnesses rather than import simulation behavior or infer canon from directory proximity.

## Damage retained as evidence

- Numbered chapter files contain sequence gaps.
- Several later chapter paths are byte-identical witnesses.
- `MANUSCRIPT_FULL.md` names only a subset of chapters despite its filename.
- The `docs/` and `world_building/` trees mix creative, operational, generated, and unresolved material.

The reader reports these conditions. It does not renumber chapters, collapse duplicates, elevate generated suggestions, or expose mutation controls as part of public reading.

## Extraction sequence

1. Publish a deterministic projection from an explicit manifest.
2. Review the inclusion/hold ledger.
3. Create a dedicated corpus repository with history-preserving moves.
4. Keep the interpreter/runtime and authenticated studio independent of the real corpus.
5. Move large media to the external byte store and retain content-addressed references in the corpus.
6. Publish text/catalog projections to the universal access plane without allowing unsupervised dual writes.

This change implements step 1 and records the evidence needed for step 2.
