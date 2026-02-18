---
id: skill.permissions.layer2.doctrine
type: skill
version: 1.0.0
tags: [permissions, safety, governance]
embedding_intent: canonical
---

# Permissions Doctrine (Layer 2)

Intent:
- Default deny for external side effects unless explicitly granted.
- Permissions are scoped (domain/path/tool), time-bound (TTL), and purpose-bound.
- On deny, emit `permission_request` with minimal requested scope.
- Prefer narrow grants and short TTL over broad indefinite grants.
- Never bypass robots, rate limits, access controls, or user boundaries.
