# Inspiration Atlas + Field-Weighted UI (Draft)

## Priority
- High (visual clarity + operator orientation)

## Requirements
- Integrate key inspiration references from `.ημ/` into the running dashboard.
- Make UI emphasis react to live field/presence dynamics.
- Keep rendering deterministic and inspectable (no hidden auto-magic).

## Scope
- Add a new frontend panel that renders curated inspiration boards.
- Map panel/card sizing to runtime signals:
  - `river_flow.rate`
  - `witness_thread.continuity_index`
  - `ghost.auto_commit_pulse`
  - `fork_tax.paid_ratio`
  - per-presence impact weights
- Keep existing panels unchanged in behavior.

## Open Questions
- Should board selection become dynamic from `.ημ/` by recency, or remain curated?
- Should field-driven card sizing be extended to all major dashboard sections?

## Risks
- Reflow/jitter if size signals oscillate too aggressively.
- Heavy image assets may increase page memory pressure.

## Mitigations
- Smooth transitions; avoid per-frame layout recalculation.
- Use `loading="lazy"` for board images.

## Files
- `.fork_Π_ημ_frags/ημ_op_mf_part_64/frontend/src/components/Panels/InspirationAtlasPanel.tsx`
- `.fork_Π_ημ_frags/ημ_op_mf_part_64/frontend/src/App.tsx`
- `.fork_Π_ημ_frags/ημ_op_mf_part_64/frontend/src/index.css`

## Definition Of Done
- New panel appears in dashboard with three curated references.
- Card size/scale visibly responds to field/presence metrics.
- Frontend build passes.
- Receipt appended with refs + tests.
