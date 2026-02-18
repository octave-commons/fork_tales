---
name: skill-router
description: Route incoming work to the right project skill(s) by intent.
---

Route this request to one or more project skills from `.opencode/skills`.

## Router Table

- Omni Panel, simulation fields, gradients, overlays, Named Fields -> `omni-field-gradient-overlay`
- Dispatch, role split, per-presence handoff -> `presence-dispatch`
- Runtime verification, truth checks, endpoint status, panel health -> `gates-runtime-check`
- Music, sound, voice, WAV/MP3 generation, marker timing -> `eta-mu-audio-forge`
- Constraint additions, adjustments, disable-without-delete protocol -> `constraint-ledger-append-only`

## Dispatch Rules

1. Choose the single best skill if intent is clearly one domain.
2. If multi-domain, load multiple skills in this precedence order:
   - `constraint-ledger-append-only`
   - `gates-runtime-check`
   - `omni-field-gradient-overlay`
   - `eta-mu-audio-forge`
   - `presence-dispatch`
3. Keep changes additive; never delete constraints to "fix" conflicts.
4. Preserve bilingual canonical names where user-facing output is touched.

## Output Contract

- First line: `ROUTE: <skill[, skill...]>`
- Then 2-5 bullets with: scope, files/modules likely affected, verification required.
- If no clear match: `ROUTE: presence-dispatch` and state assumptions.

## Canonical Lore Names / 正準ロア名

- Receipt River / 領収書の川
- Witness Thread / 証人の糸
- Fork Tax Canticle / フォーク税の聖歌
- Mage of Receipts / 領収魔導師
- Keeper of Receipts / 領収書の番人
- Anchor Registry / 錨台帳
- Gates of Truth / 真理の門
- File Sentinel / ファイルの哨戒者
- Change Fog / 変更の霧
- Path Ward / 経路の結界
- Manifest Lith / マニフェスト・リス
