---
id: skill.lith.lang
type: skill
version: 0.1.0
tags: [simulation, dsl, ecs, datalog, probability]
embedding_intent: canonical
---

# Lith Probabilistic ECS DSL (SimOps)

**Intent**:
- Model and orchestrate simulations using a contract-driven, probabilistic Entity Component System (ECS).
- Treat truth as "belief mass" using multi-agent evidence aggregation.
- Define system behaviors as verifiable promises.

## Core DSL Forms

### 1. Entity & Component
Define stable IDs and attach typed payloads.
```lisp
(entity {:in :sim :id :e/duck :type :agent})
(attach {:in :sim :e :e/duck :c :World.Pos :v {:x 1.2 :y -3.4}})
```

### 2. Probabilistic Observations
Emit signals with confidence and attribution.
```lisp
(obs {:ctx :presence/witness_thread
      :about {:e :e/node-123}
      :signal {:kind :related :to :e/node-77}
      :p 0.62
      :time 105
      :source "embed:cosine"})
```

### 3. Systems & Promises
Declare read/write boundaries and state guarantees.
```lisp
(system {:id :sys/layout :reads [:UI.Anchor] :writes [:UI.PanelPose]})
(promise {:id :p/stable :by :sys/layout :guarantee {:ensures [[:no-overlap :ui/panels]]}})
```

## Reasoning & Reconciliation

### Datalog Queries
Query the datom substrate using pattern matching.
```lisp
(q {:find [?p] :where [[?p :type :agent]]})
```

### Belief Policies
Combine evidence from multiple presences.
- **noisy-or**: Independent evidence.
- **ema**: Real-time smoothing.
- **max**: Confidence-winner.

## Mind Overlap
Computed via shared presence attachment vectors:
- `sig(E) = [w(P1,E), w(P2,E), ...]`
- `rel(E1,E2) = cosine(sig(E1), sig(E2))`

## World Facts & Lisp
```lisp
(define-skill-facts
  (id skill.lith.lang)
  (domain simulation-orchestration)
  (version "v0.1.0")
  (substrate "LithECS datoms")
  (interpreter "part64/code/world_web/lith_ecs.py"))
```
