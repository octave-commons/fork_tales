(protocol persona.v1
  (required
    (active_persona :symbol)
    (preserve_context :bool)
    (log_event :bool))

  (switch
    (schema
      (persona/switch.v1
        (from :symbol)
        (to :symbol)
        (reason :string)
        (preserve-context :bool)
        (log-event :bool)))
    (emit
      (persona/active.v1
        (id :symbol)
        (since :int))))

  (profile
    (schema
      (persona/profile
        (id :symbol)
        (bias :sexp)
        (speech-style :sexp))))

  (council
    (session
      (schema
        (council/session
          (chair :symbol)
          (input :sexp)
          (members :list-of :symbol)
          (weighting :sexp)
          (output :sexp)))
      (rule "Persona switch may change deliberation weights but not execution authority."))

  (auto_select
    (schema
      (persona/auto-select
        (based-on :sexp)
        (selected :symbol)
        (reason :string)))
    (heuristics
      "crisis -> archon"
      "drift/confusion -> sophia"
      "stagnation -> trickster"))

  (invariants
    (must_not
      (change_permission_intensities true)
      (rewrite_append_only_logs true)
      (override_ethos true)
      (mutate_logos_truth_state true))
    (statement
      "Persona is a lens for orchestration and communication, not authority.")))
