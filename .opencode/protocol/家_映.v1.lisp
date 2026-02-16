(protocol 家_映.v1

  ;; =========================================================
  ;; 0) Axioms: א — The Silent Core (Immutable)
  ;; =========================================================

  (record aleph/state.v1
    (required (id :symbol)
              (axioms :list-of :string)
              (invariants :list-of :string)))

  (define א.core
    (aleph/state.v1
      (id א.0001)
      (axioms (list
        "Append-only event sourcing for all state."
        "Content-addressed identity for files/packages."
        "No destructive side-effects without permission gating."
        "Council voting required for process lifecycle actions (policy-defined)."
        "UI is a projection: derived from receipts + snapshots, never a source of truth."))
      (invariants (list
        "Deterministic replay from receipts."
        "Separation of 見/解/封 layers."
        "Partial order across ticks; causal edges preferred over wallclock."
        "Explainability for UI layout decisions."))))

  ;; =========================================================
  ;; 1) World Root: 家
  ;; =========================================================

  (record 家/state.v1
    (required (id :symbol) (ts :int) (root-path :string))
    (optional (notes :string)))

  ;; =========================================================
  ;; 2) Presence: 主
  ;; =========================================================

  (record 主.v1
    (required (id :symbol) (ts :int) (role :string)
              (scope :map)            ;; allowed namespaces/folders
              (perm-ref :symbol))     ;; points into 許
    (optional (trust :float)
              (field-affinity :map))) ;; f1..f8 weights

  ;; =========================================================
  ;; 3) Spirit Elements: 霊素 (Daimoi)
  ;; =========================================================

  (record 霊素.v1
    (required (id :symbol) (ts :int) (from :symbol)
              (field :symbol)         ;; f1..f8
              (op :symbol)            ;; impulse/attract/repel/damp/inject/bind/sever
              (vector :map)
              (magnitude :float))
    (optional (ttl :map) (confidence :float)
              (requires-perms :list-of :symbol)
              (ethos-tags :list-of :string)
              (provenance :map)))

  (record 霊素/applied.v1
    (required (霊素-id :symbol) (ts :int) (applied :bool) (reason :string)
              (snapshot-before :symbol) (snapshot-after :symbol))
    (optional (gates :map) (delta :map)))

  ;; =========================================================
  ;; 4) Fields: 場 (8-field vector state)
  ;; =========================================================

  (record 場/snapshot.v1
    (required (id :symbol) (ts :int)
              (vectors :map)          ;; f1..f8 -> {key->value}
              (applied-霊素 :list-of :symbol))
    (optional (merge-mode :symbol)    ;; wallclock/causal/hybrid
              (ticks :list-of :symbol)))

  (record 場/schema.v1
    (required (field :symbol) (name :string)
              (delta-keys :list-of :symbol)
              (interpretation :map)))

  ;; (Interface-only) field schemas for f1..f8 live here; keys are “knobs” for rendering + routing

  ;; =========================================================
  ;; 5) Coherence: 心
  ;; =========================================================

  (record 心/state.v1
    (required (id :symbol) (ts :int)
              (centroid :map)
              (tension :float)
              (drift :float)
              (entropy :float))
    (optional (dominant-perspective :symbol)))

  ;; =========================================================
  ;; 6) Breath: 息 (Concurrent ticks + causality)
  ;; =========================================================

  (record 息/tick.v1
    (required (id :symbol) (ts :int)
              (source :symbol) (phase :symbol) (seq :int))
    (optional (load :float)))

  (record 息/causal.v1
    (required (from-tick :symbol) (to-tick :symbol) (rel :symbol))
    (optional (channel :string) (evidence :string)))

  (record 息/perspective.v1
    (required (id :symbol) (name :string)
              (sources :list-of :symbol)
              (merge :symbol))          ;; event-time/causal-time/wallclock/hybrid
    (optional (weights :map) (window :map)))

  ;; =========================================================
  ;; 7) Knowledge Layers: 見 (η) / 解 (μ) / 封 (Π)
  ;; =========================================================

  (record 見/observation.v1
    (required (id :symbol) (ts :int) (source :symbol) (payload :string))
    (optional (entropy :float) (confidence :float) (refs :list-of :string)))

  (record 解/interpretation.v1
    (required (id :symbol) (ts :int) (from-見 :symbol)
              (model-version :string) (inputs-hash :string))
    (optional (claims :list-of :string)
              (nooi-nodes :list-of :symbol)
              (derived-霊素 :list-of :symbol)))

  (record 封/package.v1
    (required (id :string) (ts :int) (name :string) (kind :symbol)
              (inputs :list-of :map)
              (outputs :list-of :map))
    (optional (tags :list-of :string) (entrypoints :list-of :map)
              (provenance :map) (links :map)))

  ;; =========================================================
  ;; 8) Graphs: 結 (Nexus) / 脈 (Nooi)
  ;; =========================================================

  (record 結/resource.v1
    (required (id :symbol) (type :string))
    (optional (tags :list-of :string) (meta :map)))

  (record 結/edge.v1
    (required (from :symbol) (to :symbol) (rel :symbol) (weight :float) (ts :int))
    (optional (evidence :list-of :string)))

  (record 脈/node.v1
    (required (id :symbol) (kind :symbol) (label :string))
    (optional (embedding-ref :symbol) (confidence :float) (tags :list-of :string)))

  (record 脈/edge.v1
    (required (from :symbol) (to :symbol) (rel :symbol) (weight :float) (ts :int))
    (optional (evidence :list-of :string)))

  ;; =========================================================
  ;; 9) Governance: 許 (Permissions) / 議 (Council)
  ;; =========================================================

  (record 許/event.v1
    (required (id :symbol) (ts :int)
              (type :symbol)            ;; grant/deny/request/revoke
              (principal :symbol)
              (cap :symbol)
              (scope :map)
              (ttl :map)
              (reason :string)))

  (record 議/action.v1
    (required (id :symbol) (ts :int)
              (type :symbol)            ;; start/stop/restart/delete/remove/start-once/...
              (target :symbol)
              (reason :string)
              (requested-by :symbol))
    (optional (params :map) (risk :symbol) (ttl :map)))

  (record 議/membership.v1
    (required (ballot-id :symbol) (ts :int)
              (computed-from :symbol)    ;; nexus state ref
              (eligible-主 :list-of :symbol)
              (excluded :list-of :symbol)
              (reason :string)))

  (record 議/vote.v1
    (required (ballot-id :symbol) (ts :int)
              (voter :symbol)           ;; 主
              (stance :symbol)          ;; yes/no/abstain/veto
              (weight :float)
              (reason :string)))

  (record 議/warrant.v1
    (required (action-id :symbol) (ballot-id :symbol) (ts :int)
              (decision :symbol)        ;; approved/rejected
              (expires-ts :int)
              (tally :map)
              (votes :list-of :議/vote.v1)
              (rules-snapshot :map)
              (reason :string)))

  ;; =========================================================
  ;; 10) UI: 映 (Projection layer)
  ;; =========================================================

  ;; UI elements are Nexus resources (結/resource) with type ui.*
  ;; 映 records are derived layout/projection states.

  (record 映/element.v1
    (required (id :symbol)              ;; nexus.ui.*
              (kind :symbol)            ;; panel/widget/node/edge/list/chart/overlay/chat-lens
              (title :string))
    (optional (binds-to :list-of :symbol)
              (field-bindings :map)     ;; f1..f8 -> weight
              (presence :symbol)
              (tags :list-of :string)))

  ;; Derived visual state “mass model”
  (record 映/state.v1
    (required (element-id :symbol) (ts :int)
              (mass :float)             ;; 0..1 visual weight
              (priority :float)         ;; 0..1 z/attention ordering
              (area :float)             ;; 0..1 desired area fraction
              (opacity :float)
              (pulse :float))
    (optional (sources :list-of :symbol) ;; snapshots / 霊素 / ballots
              (explain :map)))

  ;; Global clamps to keep the UI usable
  (record 映/clamp.v1
    (required (min-area :float) (max-area :float)
              (max-pulse :float) (decay-half-life :map)))

  ;; UI events (pins override dynamics but are still append-only receipts)
  (record 映/event.v1
    (required (id :symbol) (ts :int)
              (type :symbol)             ;; pin/unpin/focus/freeze/unfreeze
              (element-id :symbol)
              (ttl :map)
              (reason :string))
    (optional (area :float)))

  ;; The computed layout snapshot
  (record 映/layout.v1
    (required (id :symbol) (ts :int)
              (perspective :symbol)      ;; 息/perspective id
              (elements :list-of :symbol)
              (rects :map)               ;; element-id -> {x y w h}
              (states :list-of :映/state.v1)
              (clamps :映/clamp.v1))
    (optional (notes :string)))

  ;; Chat as “field-bound lenses”
  (record 映/chat-session.v1
    (required (id :symbol) (ts :int)
              (presence :symbol)         ;; 主
              (lens-element :symbol)     ;; nexus.ui.chat.*
              (field-bindings :map)
              (memory-scope :symbol))    ;; local/shared/council/global
    (optional (tags :list-of :string) (status :symbol)))

  ;; Vector visualization contract (no lies)
  (record 映/vector-view.v1
    (required (id :symbol) (ts :int)
              (field-snapshot :symbol)   ;; 場/snapshot
              (mode :symbol)             ;; axes/pca/cluster/barycentric-slice
              (axes :map))               ;; e.g. {x f3.focus, y f6.curiosity}
    (optional (overlay-霊素 :list-of :symbol)
              (overlay-主 :list-of :symbol)
              (show-causality :bool)))

  ;; Concurrency visualization: swimlanes + causal threads
  (record 映/tick-view.v1
    (required (id :symbol) (ts :int)
              (sources :list-of :symbol)
              (window :map))
    (optional (show-causal :bool) (merge :symbol)))
)
