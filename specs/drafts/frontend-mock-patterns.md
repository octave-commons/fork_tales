# Specification: Frontend Mock and Fallback Patterns

## Overview
This document catalogs the "Deterministic Hallucination" and structural mock patterns identified within the `part64/frontend` runtime. These patterns are used to maintain UI coherence and ensure the interface remains functional even when the underlying simulation state is sparse or asymmetrical.

## Findings

### 1. Canonical Typo: `symetry`
- **Location**: 
  - `part64/frontend/src/app/worldPanelLayout.ts`
  - `part64/frontend/src/App.tsx`
  - `part64/frontend/src/components/Panels/Chat.tsx`
- **Nature**: The string `symetry` (missing the second 'm') is used as a functional key for panel anchors and presence IDs. It has effectively become a "canonical" identifier through repetition.
- **Impact**: Any attempt to "fix" this typo requires a coordinated update across multiple files to avoid breaking the UI-to-Anchor bindings.

### 2. WebRTC Loopback (Presence Call Simulator)
- **Location**: `part64/frontend/src/components/Panels/PresenceCallDeck.tsx`
- **Nature**: Instead of connecting to a remote signaling server, the frontend establishes both `outbound` and `inbound` `RTCPeerConnection` instances locally. It pipes them together to mix `mix.wav` (field music) and Spoken Presence replies (TTS) into a single "call" stream.
- **Impact**: Mocks a complex network service using purely client-side logic to achieve the "Presence Call" experience.

### 3. Deterministic Hallucination (Spatial Positioning)
- **Location**: `part64/frontend/src/App.tsx`, `part64/frontend/src/app/worldPanelLayout.ts`
- **Nature**: The frontend uses a `stableUnitHash` function to generate coordinates (x, y) for entities that the backend has not yet positioned.
- **Logic**: `x: clamp(0.14 + (stableUnitHash(\`${museId}|x\`) * 0.72), 0.08, 0.92)`.
- **Impact**: Ensures that Muses and Nexus nodes appear in consistent, "hallucinated" locations, preventing UI jitter while waiting for simulation data.

### 4. Tiered Identity Fallbacks
- **Location**: `part64/frontend/src/components/Panels/Chat.tsx`, `part64/frontend/src/components/Panels/PresenceCallDeck.tsx`
- **Nature**: Hardcoded identity arrays (`FALLBACK_MUSES`, `FALLBACK_PRESENCES`) act as the final tier in a resolution cascade (Runtime -> World -> Manifest -> Fallback).
- **Impact**: Prevents "Empty World" scenarios during initial boot or simulation drift.

### 5. Hardware-to-Nexus Synthesis
- **Location**: `part64/frontend/src/App.tsx` (`buildMuseSurroundingNodes`)
- **Nature**: Telemetry from the `resource_heartbeat` (utilization % of GPU/CPU) is transformed into virtual "Nexus Nodes" (e.g., `device:gpu0`).
- **Impact**: Mocks the physical hardware as part of the digital simulation, allowing the AI to "perceive" its own compute environment.

## Risks
- **Typo Fragility**: The `symetry` identifier is a "magic string" that could easily be broken by a naive refactor.
- **Loopback Resource Leak**: Local WebRTC peer connections can be resource-heavy if not properly closed (though the `closeCallSession` callback appears robust).
- **Coordinate Conflict**: If the backend begins providing coordinates that conflict with the frontend's "hallucinated" positions, entities may "jump" unexpectedly.

## Sub-tasks
- [ ] **Typo Audit**: Decide whether to rename `symetry` to `symmetry` or document it as a "Lore-Accurate Glitch."
- [ ] **Utility Consolidation**: Move `stableUnitHash` and positioning logic into a shared spatial utility file.
- [ ] **Hardware Schema**: Formalize the conversion of heartbeat metrics to Nexus nodes to ensure consistent AI perception.

## Definition of Done
- [x] Identification of all major mock/hallucination patterns.
- [ ] Decision on `symetry` typo status.
- [ ] Refactor plan for spatial "hallucination" logic.
