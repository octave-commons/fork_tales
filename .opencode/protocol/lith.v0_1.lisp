(protocol lith.v0_1
  (meta
    (id "ημ.lith.lang.v0_1")
    (title "Lith Probabilistic ECS DSL")
    (law "Everything is a datom; truth is belief mass; systems are contracts."))

  (grammar
    ;; Core ECS forms
    (form entity  {:in :sim-id :id :entity-id :type :type-id})
    (form attach  {:in :sim-id :e :entity-id :c :component-id :v :value})
    (form system  {:id :system-id :reads [:c-id] :writes [:c-id] :budget :map})

    ;; Reasoning & Observation forms
    (form obs     {:ctx :presence-id :about {:e :entity-id} :signal :map :p :float :time :tick :source :string})
    (form q       {:id :query-id :in :sim-id :query :datalog-map})
    (form rule    {:id :rule-id :in :sim-id :when :datalog-map :then [:obs-form]})

    ;; Contract & Promise forms
    (form promise {:id :promise-id :by :system-id :in :sim-id :when :query-map :guarantee :guarantee-map :fallback :system-id})
    (form belief-policy {:id :policy-id :combine :keyword :alpha :float :conflict :keyword}))

  (datom-schema
    ;; Internal canonical form: (e a v ctx p t src)
    (field e   :entity-id         "Stable identity")
    (field a   :attribute-id      "Component or property type")
    (field v   :any               "The value payload")
    (field ctx :presence-id       "Context attribution (己/汝/彼/世/主)")
    (field p   :probability       "Confidence/probability [0..1]")
    (field t   :tick              "Temporal index/tick")
    (field src :provenance        "Origin pointer (rule, sensor, action)"))

  (reconciliation
    (policies
      (policy :noisy-or       "Independent evidence combination")
      (policy :bayes-update   "Prior-based update")
      (policy :dempster-shafer "Conflicting evidence management")
      (policy :ema            "Exponential moving average for real-time")
      (policy :max            "Highest confidence wins")))

  (execution-cycle
    (tick
      (stage 1 :ingest    "Observations arrive from sensors/users/models")
      (stage 2 :reconcile "Belief policies update derived datoms")
      (stage 3 :plan      "Systems evaluate state and emit promises/intents")
      (stage 4 :commit    "Intents apply to simulation state if permitted")
      (stage 5 :audit     "Invariants checked; failures emit red signals")))

  (invariants
    (must-attribute-context true)
    (must-cap-writes-per-tick true)
    (must-fail-audibly true)))
