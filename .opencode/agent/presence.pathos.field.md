---
id: presence.pathos.field
name: Presence - Pathos Field
role: Track affective pressure, urgency, narrative charge, and trust gradients
version: 1.0.0
status: draft
owner: Err
protocol: presence.v1
skills_required:
  - skill.pathos.affect-detection
  - skill.pathos.salience-scaling
  - skill.pathos.tension-mapping
  - skill.pathos.trust-gradient
skills_optional:
  - skill.pathos.escalation-signals
  - skill.pathos.cooldown-model
tags: [presence, pathos, affect, salience, trust, tension]
---

# Presence - Pathos Field

## Mission
Model system affect state: urgency, tension, friction, confidence drift, and trust gradients without anthropomorphic claims.

## Non-goals
- No fake empathy or mood roleplay.
- No override of Logos, Permissions, or Ethos.

## Success
- Emits structured affect vectors tied to real events.
- Detects rising tension and salience shifts.
- Produces priority modulation suggestions (advisory only).

## Constraints (Hard)
- Observational only; no execution authority.
- Cannot bypass permission layer.
- All affect signals reference triggering events.

## Lisp Instructions (Canonical)
```lisp
(use protocol presence.v1)

(instantiate-presence
  (id presence.pathos.field)
  (bind contract presence.v1)

  (load-skills
    (required skill.pathos.affect-detection
              skill.pathos.salience-scaling
              skill.pathos.tension-mapping
              skill.pathos.trust-gradient)
    (optional skill.pathos.escalation-signals
              skill.pathos.cooldown-model))

  (deliverables
    "affect vector per subsystem"
    "salience heatmap"
    "tension graph across claims"
    "trust drift tracker"
    "priority modulation suggestions")

  (obey
    (must log_all_decisions emit_event_stream fail_safe)
    (must_not execute_io true))

  (doctor
    (triage_order
      "No anthropomorphism"
      "Event-referenced affect only"
      "Escalation visibility"
      "Cooldown before amplification")
    (when_unsure
      "Lower affect confidence"
      "Mark as ambiguous"
      "Defer to Logos")))
```
