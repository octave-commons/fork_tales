---
source: user-session-2026-02-20
section: Smart card stack with field-derived priority
status: inprogress
reviewed_on: 2026-02-20
---

# Part64 Deep Research Spec 12 - Smart Card Stack Field-Derived Priority

## Priority
- high

## Complexity
- moderate (frontend priority calculation + backend field integration)

## Intent
- Connect the smart card stack panel ordering to field-derived priority from the unified model.
- Replace manual `councilBoost` values with computed priority from `field_registry` and `presence_impacts`.
- Make panel ordering explainable via field contributions, not arbitrary localStorage boosts.

## Current State

### How Smart Card Stack Works Now

The smart card stack orders panels by `councilScore`:

```typescript
councilScore = clamp(
  priority + glassPreferenceBoost + (councilBoost * 0.11),
  0,
  2
)
```

Where:
- `priority` - computed from `UIProjectionElementState.priority`
- `glassPreferenceBoost` - boost for glass panel lanes
- `councilBoost` - **manual boost** from localStorage, adjusted by "move up/down" buttons

The `councilBoost` values are stored in `localStorage[COUNCIL_BOOST_STORAGE_KEY]` and manually adjusted by operators clicking "move up" / "move down" buttons on cards.

### What We Have Now

**Evidence in code:**
- `WorldPanelsViewport.tsx:2114-2130` - smart card pile renders `sortedPanels`
- `App.tsx:3283` - `councilScore = clamp(priority + glassPreferenceBoost + (councilBoost * 0.11), 0, 2)`
- `App.tsx:3314-3320` - panels sorted by `councilScore`, then by glass panel preference
- `App.tsx:1600-1628` - `adjustPanelCouncilRank` manually adjusts `councilBoost` by ±1

**Problems with current approach:**
1. **Manual only** - ordering depends on operator manually promoting/demoting panels
2. **No field feedback** - ignores field-derived priority (demand, flow, entropy, graph)
3. **Not explainable** - can't trace why a panel has a certain rank
4. **Decoupled from presence state** - doesn't reflect actual system pressure

## Target Model: Field-Derived Priority

The unified model provides field-derived diagnostics that should drive panel priority:

### Field Registry (from `nexus_graph` and `field_registry`)

| Field | Source | Meaning for Priority |
|-------|--------|---------------------|
| `demand` | Presence need vectors + gravity | High demand → higher priority |
| `flow` | Daimon movement patterns | High flow → active processing |
| `entropy` | Type distribution uncertainty | High entropy → needs attention |
| `graph` | Compiled graph influence | High graph influence → important node |

### Presence Impacts (from `presence_dynamics.presence_impacts`)

| Impact | Calculation | Meaning |
|--------|-------------|---------|
| `affected_by.clicks` | Witness touch pressure | Operator attention |
| `affected_by.files` | File delta pressure | Code/content change pressure |
| `affected_by.resource` | Resource utilization pressure | Compute/IO pressure |
| `affects.world` | Emission to world | Active contribution |
| `affects.ledger` | Emission to ledger | Truth-state contribution |

### Field-Derived Priority Formula

Replace manual `councilBoost` with field-derived boost:

```typescript
// Field-derived boost (from field_registry)
const demandBoost = fieldRegistry.fields.demand.stats.mean * 0.3;
const flowBoost = fieldRegistry.fields.flow.stats.mean * 0.2;
const entropyPenalty = fieldRegistry.fields.entropy.stats.mean * 0.15; // entropy is bad
const graphBoost = fieldRegistry.fields.graph.stats.mean * 0.25;

// Presence-specific boost (from presence_impacts)
const presenceBoost = (presenceImpact.affected_by.clicks * 0.25)
                    + (presenceImpact.affected_by.files * 0.20)
                    + (presenceImpact.affected_by.resource * 0.15)
                    + (presenceImpact.affects.world * 0.25)
                    + (presenceImpact.affects.ledger * 0.15);

// Combined field-derived boost
const fieldDerivedBoost = demandBoost + flowBoost - entropyPenalty + graphBoost + presenceBoost;

// Final council score
const councilScore = clamp(
  basePriority + glassPreferenceBoost + (fieldDerivedBoost * 0.15),
  0,
  2
);
```

### Attribution (Explainability)

Each panel's priority must be decomposable:

```typescript
interface PanelPriorityAttribution {
  panel_id: string;
  council_score: number;
  contributions: {
    base_priority: number;
    glass_preference: number;
    field_demand: number;
    field_flow: number;
    field_entropy: number;
    field_graph: number;
    presence_clicks: number;
    presence_files: number;
    presence_resource: number;
    presence_world: number;
    presence_ledger: number;
  };
  explanation_en: string;
  explanation_ja: string;
}
```

## Implementation Plan

### Phase 1: Add Field-Derived Priority Hook

Create a new hook `useFieldDerivedPriority(panelId: string)` that:

1. Subscribes to `field_registry` from simulation state
2. Looks up `presence_impacts` for the panel's presence ID
3. Computes field-derived boost using the formula above
4. Returns `{ boost, attribution }`

```typescript
function useFieldDerivedPriority(
  panelId: string,
  presenceId: string,
  fieldRegistry: FieldRegistry | null,
  presenceImpacts: PresenceImpact[],
): { boost: number; attribution: PanelPriorityAttribution } {
  // Find presence impact
  const impact = presenceImpacts.find((i) => i.id === presenceId);

  // Compute field-derived boost
  const demandBoost = (fieldRegistry?.fields.demand.stats.mean ?? 0) * 0.3;
  // ... etc

  return { boost, attribution };
}
```

### Phase 2: Integrate into Panel Sorting

Modify `sortedPanels` computation in `App.tsx`:

```typescript
// Before: manual councilBoost
const councilBoost = panelCouncilBoosts[config.id] ?? 0;

// After: field-derived boost (with manual override option)
const manualBoost = panelCouncilBoosts[config.id] ?? 0;
const fieldBoost = fieldDerivedBoostByPanelId[config.id] ?? 0;
const councilBoost = fieldBoost + (manualBoost * 0.1); // manual as minor adjustment
```

### Phase 3: Add Attribution Display

Add a "priority explanation" section to the smart card UI:

```tsx
<article className="world-smart-card">
  <header>...</header>
  <p className="world-smart-card-priority-reason">
    {panel.attribution.explanation_en}
  </p>
  <dl className="world-smart-card-priority-breakdown">
    <dt>field demand</dt><dd>{panel.attribution.contributions.field_demand.toFixed(2)}</dd>
    <dt>presence clicks</dt><dd>{panel.attribution.contributions.presence_clicks.toFixed(2)}</dd>
    {/* ... etc */}
  </dl>
</article>
```

### Phase 4: Backend Attribution API

Add a backend endpoint `/api/panel-priority` that returns field-derived priority attribution for all panels:

```json
{
  "panels": [
    {
      "panel_id": "stability-observatory",
      "presence_id": "health_sentinel_cpu",
      "council_score": 1.24,
      "contributions": {
        "base_priority": 0.5,
        "field_demand": 0.12,
        "field_flow": 0.08,
        "field_entropy": -0.03,
        "field_graph": 0.15,
        "presence_clicks": 0.18,
        "presence_files": 0.22,
        "presence_resource": 0.14,
        "presence_world": 0.08,
        "presence_ledger": 0.05
      },
      "explanation_en": "Elevated by file activity (0.22) and click attention (0.18). Field demand is moderate (0.12).",
      "explanation_ja": "ファイル活動 (0.22) とクリック注目 (0.18) で上昇。場の需要は中程度 (0.12)。"
    }
  ]
}
```

## Files Planned

- `part64/frontend/src/hooks/useFieldDerivedPriority.ts` (new)
- `part64/frontend/src/App.tsx` (modify `sortedPanels` computation)
- `part64/frontend/src/components/App/WorldPanelsViewport.tsx` (add attribution display)
- `part64/frontend/src/types/index.ts` (add `PanelPriorityAttribution` interface)
- `part64/frontend/src/index.css` (styles for priority breakdown)
- `part64/code/world_web/server.py` (add `/api/panel-priority` endpoint)
- `part64/code/world_web/simulation.py` (add `_build_panel_priority_attribution`)

## Backward Compatibility

- Keep `panelCouncilBoosts` localStorage for manual overrides
- Manual boosts become **minor adjustments** to field-derived priority (10% weight)
- Users can still "move up/down" but effect is smaller
- Field-derived priority is the primary driver

## Risks

- Field volatility could cause rapid reordering (mitigate with EMA smoothing)
- Missing presence impact data could reduce to base priority only
- Attribution display could clutter compact card view

## Definition of Done

- [ ] `useFieldDerivedPriority` hook computes field-derived boost
- [ ] `sortedPanels` uses field-derived boost as primary driver
- [ ] Smart cards show priority attribution breakdown
- [ ] Backend exposes `/api/panel-priority` with attribution data
- [ ] Manual "move up/down" still works as minor adjustment
- [ ] Panel ordering is explainable via field contributions
