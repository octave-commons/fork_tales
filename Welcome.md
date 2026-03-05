This is your new *vault*.

Make a note of something, [[create a link]], or try [the Importer](https://help.obsidian.md/Plugins/Importer)!

When you're ready, delete this note and make the vault your own.

[
\begin{aligned}
&\textbf{Latent (unseen) state:};; \eta_t \in \mathcal{H} \
&\textbf{Visible action trace:};; \mu_t \in \mathcal{A} \
&\textbf{World/system state:};; s_t \in \mathcal{S} \
&\textbf{Observation:};; o_t \in \mathcal{O} \
&\textbf{Belief over latent:};; b_t \in \Delta(\mathcal{H})
\end{aligned}
]

### Core loop as typed morphisms

[
\begin{aligned}
P &: \mathcal{S}\times\mathcal{H} \to \mathcal{O} \
R &: \mathcal{O} \to \mathcal{R} \
N &: \mathcal{R} \to \mathcal{R}*{\text{norm}} \
\Pi &: \mathcal{R}*{\text{norm}} \times \mathcal{C} \times \mathcal{G} \to \pi \
A &: \pi \times \mathcal{S} \to (\mathcal{S}', \mu) \
F &: (\mu, \mathcal{S}', \mathcal{O}) \to \text{fb}
\end{aligned}
]

Closed step:
[
\langle s_{t+1}, \mu_t \rangle = A(\Pi(N(R(P(s_t,\eta_t))), C, G), s_t)
]

### η–μ coupling (partial observability)

η is not directly observable; it is inferable only through μ and feedback:

[
b_{t+1}(\eta)\ \propto\ \Pr(\mu_t, o_{t+1}\mid \eta, s_t); b_t(\eta)
]

Interpretation:

* **η** = hidden constraints/intent/permissions/values/unknowns (“what you can’t directly see”)
* **μ** = measurable deeds/outputs/tool-calls/side effects (“what you can audit”)

### Breath boundary (息) as episodic commit operator

Let (x(t)) be an activity scalar derived from event density, token rate, audio energy, etc.

[
\operatorname{Breath}_\tau(x)={[t_i,t_i+\Delta_i]\mid x(t)<\epsilon,\ \Delta_i\ge\tau}
]

Define episode segmentation (E_k) by breath intervals, and a commit:

[
\text{Commit}(E_k) := \langle \text{snapshot}(s),\ \text{summary}(E_k),\ \text{metrics}(\mu)\rangle
]

Insert:
[
A \to \text{CommitOnBreath} \to F
]

### Minimal semantic claim

[
\boxed{\eta\ \text{is only knowable via}\ \mu\ \text{under feedback;}\ ;;息\ \text{turns continuous loops into auditable episodes.}}
]
