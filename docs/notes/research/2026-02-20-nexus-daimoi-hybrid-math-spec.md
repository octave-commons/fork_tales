---
title: "Nexus-Daimoi Hybrid Math Specification"
summary: "Compact mathematical specification for nexus, daimoi, nooi fields, collisions, and emission dynamics."
category: "research"
created_at: "2026-02-20T14:56:14"
original_filename: "2026.02.20.14.56.14.md"
original_relpath: "docs/notes/research/2026.02.20.14.56.14.md"
tags:
  - research
  - math-spec
  - simulation
---

# Nexus–Daimoi Hybrid Simulation: Compact Mathematical Specification

## 0. State Spaces

Let:

* Continuous field space:  ( \mathbb{R}^m ) (typically (m=2))
* Semantic embedding space: ( \mathbb{R}^d )
* Intent type index set: ( J = {1,\dots,r} )

Discrete time with step (\Delta t).

---

## 1. Core Entities

### 1.1 Nexus

Each nexus (n) is defined by:

* Position: ( y_n \in \mathbb{R}^m )
* Velocity: ( v_n \in \mathbb{R}^m )
* Mass: ( M_n > 0 )
* Friction: ( \gamma_n \in (0,1) )
* Wallet counts: ( C_{n,j} \in \mathbb{Z}_{\ge 0} )
* Total wallet energy: ( E_n = \sum_j C_{n,j} )
* Embedding: ( e_n \in \mathbb{R}^d )
* Edge set: ( \mathcal{E}(n) )

Nexus is both:

* Particle in field space
* Graph node in topology

---

### 1.2 Presence (Role)

Presence (p) is an abstract policy attached to anchor nexus (n_p):

* Lens embedding: ( e_p \in \mathbb{R}^d )
* Spend rate: ( \sigma_p )
* Emission rate: ( \mu_p )

Presence has no independent position.

---

### 1.3 Daimon

Each daimon (i) has:

* Position: ( z_i \in \mathbb{R}^m )
* Velocity: ( v_i \in \mathbb{R}^m )
* Friction: ( \gamma_i )
* Owner presence: ( o_i )
* Intent quanta counts: ( C_{i,j} \in \mathbb{Z}_{\ge 0} )
* Total energy: ( E_i = \sum_j C_{i,j} )

Probability distribution:
[
p_{i,j} = \frac{C_{i,j}}{E_i}
]

Embeddings:

* Seed: ( s_i )
* Mantle: ( m_i )
* Intent embedding:
  [
  u_i = \sum_j p_{i,j} ; emb(j)
  ]

Owner is immutable.

---

## 2. Nooi Field (Sparse Vector Field)

Discrete spatial cells (c).

Each cell stores 8 vector layers:
[
\mathbf{F}_\ell[c] \in \mathbb{R}^m, \quad \ell = 1..8
]

### 2.1 Decay

[
\mathbf{F}*\ell[c] \leftarrow (1 - \delta*\ell \Delta t) \mathbf{F}_\ell[c]
]

### 2.2 Deposit (ACO Pheromone)

For daimon (i) at cell (c):
[
\mathbf{F}*\ell[c] \leftarrow \mathbf{F}*\ell[c] + \alpha_\ell ; \omega_{i,\ell} ; \hat{v}_i
]

where (\hat{v}_i = v_i / |v_i|).

Field is endogenous: movement → deposit → future movement.

---

## 3. Motion Dynamics

### 3.1 Field Acceleration

[
\mathbf{a}^{field}*i = \sum*{\ell=1}^8 w_{i,\ell} \mathbf{F}_\ell[c(z_i)]
]

### 3.2 Semantic Gravity

[
\mathbf{a}^{grav}_i = \sum_n G ; \kappa(i,n) \frac{y_n - z_i}{|y_n - z_i|^2 + \epsilon}
]

### 3.3 Noise

[
\mathbf{a}^{noise}_i = \lambda ; simplex(z_i,t)
]

### 3.4 Daimon Update

[
v_i \leftarrow (1 - \gamma_i \Delta t)v_i + \Delta t(\mathbf{a}^{field}_i + \mathbf{a}^{grav}_i + \mathbf{a}^{noise}_i)
]
[
z_i \leftarrow z_i + v_i \Delta t
]

### 3.5 Nexus Update

(no self-propulsion, higher friction)
[
v_n \leftarrow (1 - \gamma_n \Delta t)v_n + \Delta t(\mathbf{a}^{field}_n + \mathbf{a}^{grav}_n + \mathbf{F}^{edge}_n)
]
[
y_n \leftarrow y_n + v_n \Delta t
]

---

## 4. Graph Elastic Edges

Edge (e=(n_a,n_b)):

* Rest length: (L_e)
* Elasticity:
  [
  k_e = k_0 (1 - sim(e_{n_a}, e_{n_b}))^\beta
  ]

Force:
[
\mathbf{F}^{edge}*{n_a} = -k_e (|y_a - y_b| - L_e) \hat{d}*{ab}
]

Single edge type only.

---

## 5. Collision + Wallet Exchange

Let daimon (i) collide with nexus (n).

Saturation:
[
S_n = \frac{load_n}{cap_n}
]

If (S_n \ge 1): deflect.

Else exchange intent quanta.

Exchange count:
[
k = \lfloor \eta \frac{E_i}{E_n + \epsilon} E_i \rfloor
]

Sample intents from distributions and update counts:

[
C_{i,j} \leftarrow C_{i,j} - \Delta C_{i,j}^{out} + \Delta C_{i,j}^{in}
]
[
C_{n,j} \leftarrow C_{n,j} - \Delta C_{n,j}^{out} + \Delta C_{n,j}^{in}
]

Recompute probabilities.

---

## 6. Collapse (Observation)

If fully absorbed ((E_i \to 0)):

Sample:
[
j^* \sim p_i
]

Presence lens interpretation:
[
action = \mathcal{O}*p(j^*) = arg\max_j p*{i,j} \cdot sim(emb(j), e_p)
]

Emit ledger event.

---

## 7. Emission

Presence emits new daimoi:

[
C_{i,j} \sim Multinomial(E, P_p(j))
]

[
P_p(j) = C_{p,j} / E_p
]

Energy is integer quanta.

---

## 8. Closed Loop

Wallet spending → Emission → Motion → Deposit → Field → Nexus Drift → Graph Tension → Collision → Exchange → Collapse → Wallet Update

System converges under decay (\delta), friction (\gamma), and noise (\lambda).

---

## 9. Primitive Set (Minimal Engine)

* MOVE
* DEPOSIT
* DECAY
* BOND
* COLLIDE
* EXCHANGE
* SPEND
* EMIT
* SATURATE

All behavior derives from these primitives + hyperparameters.
