---
name: omni-field-gradient-overlay
description: Build and tune Named Field gradients for the Omni Panel simulation overlay with bilingual labels and stable performance.
metadata:
  owner: project
  version: 1
---

# Omni Field Gradient Overlay

Use this skill when work touches the simulation overlay for Named Fields in Omni Panel.

## Canonical Fields

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

## Workflow

1. Find existing gradient/render pattern used by Omni Panel.
2. Add or update field gradient config in additive form.
3. Keep bilingual labels in UI-facing structures.
4. Verify `/api/catalog` and `/ws` data still map correctly to overlay.
5. Validate desktop and mobile readability.

## Guardrails

- Prefer deterministic values (seeded noise, bounded animation rates).
- Avoid flat one-color overlays; use intentional gradient direction and contrast.
- Do not remove constraints or delete legacy field entries.
