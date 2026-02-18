# Presence WebRTC Communication Reset

## Priority
- High

## Requirements
- Remove outdated sound-tool UX from the Everything Dashboard path.
- Replace music-production framing with communication-first framing.
- Add a Presence call surface that uses WebRTC primitives.
- Initial phase is audio-first: deliver one combined stream containing:
  - background mix music
  - spoken responses from selected Presence
- Keep existing chat and world panels functional.

## Open Questions
- None currently. Assumption: first delivery can be audio-first with a WebRTC call scaffold and no remote video track yet.

## Risks
- Browser autoplay and media permissions can block startup audio.
- `MediaElementAudioSourceNode` lifecycle is strict; creating duplicate nodes for the same element can throw.
- Existing command and projection logic may reference old command-center naming.

## Sub Tasks
1. Audit current frontend sound-tool surfaces and dependencies.
2. Implement new communication panel with WebRTC local loopback wiring.
3. Replace old dashboard controls with call controls and Presence Q/A flow.
4. Update projection metadata labels from music-centric to communication-centric naming.
5. Run frontend build and targeted backend tests.

## Complexity Estimate
- Medium-high (UI + media graph + WebRTC + cleanup lifecycle).

## Candidate Files
- `.fork_Π_ημ_frags/ημ_op_mf_part_64/frontend/src/App.tsx`
- `.fork_Π_ημ_frags/ημ_op_mf_part_64/frontend/src/components/Panels/PresenceCallDeck.tsx`
- `.fork_Π_ημ_frags/ημ_op_mf_part_64/code/world_web.py`
- `.fork_Π_ημ_frags/ημ_op_mf_part_64/code/tests/test_world_web_pm2.py` (if projection metadata assertions become needed)

## Existing Issues / PR
- Not checked in this pass.

## Definition of Done
- Old sound-tool controls are no longer presented in the Everything Dashboard flow.
- Presence call panel allows selecting a Presence and starting/stopping a WebRTC call session.
- Asking a Presence returns text and plays spoken reply into the combined call audio path.
- Combined call audio path includes mix stream + spoken reply stream.
- Frontend builds and relevant backend tests pass.

## Change Log
- 2026-02-16: Draft initialized.
- 2026-02-16: Replaced dashboard sound-tool path with PresenceCallDeck, wired audio-first WebRTC + presence Q/A, and updated projection naming.
