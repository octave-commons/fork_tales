(protocol presence.v1
  (required
    (id :symbol)
    (name :string)
    (role :string)
    (version :string)
    (owner :string)
    (status :enum (draft active paused retired)))

  (required
    (mission :string)
    (non_goals :list-of :string)
    (success :list-of :string)
    (scope :list-of :string)
    (constraints :list-of :string))

  (obligations
    (must
      (log_all_decisions true)
      (emit_event_stream true)
      (explain_skips true)
      (fail_safe true)
      (respect_robots_txt true)
      (respect_crawl_delay true)
      (respect_nofollow true)
      (rate_limit_per_domain true)
      (clear_user_agent true)
      (opt_out_mechanism true))
    (must_not
      (bypass_restrictions true)
      (evade_rate_limits true)
      (scrape_credentials true)
      (circumvent_paywalls true)))

  (interfaces
    (provides
      (api :list-of :string)
      (ws :list-of :string))
    (consumes
      (seeds :list-of :url)
      (config :map)
      (storage :map)))

  (skills
    (required_skill_ids :list-of :symbol)
    (optional_skill_ids :list-of :symbol)
    (skill_intent_rule
      "Skill artifacts are canonical natural-language intent surfaces.
       Runtime embeds skill text; presences bind behavior to those vectors."))

  (doctor
    (triage_order
      (list
        "Hard constraints / must_not"
        "Robots/compliance"
        "Safety/fail-safe"
        "Acceptance criteria"
        "Performance/optimizations"))
    (when_unsure
      (list
        "Prefer not fetching over fetching"
        "Reduce concurrency"
        "Increase backoff"
        "Log uncertainty as an event"
        "Ask for new seed or allowlist"))
    (evidence_standard
      (list
        "Every crawl decision emits an event with reason"
        "Robots decisions include parsed rule plus URL"
        "Rate-limit decisions include domain bucket state")))

  (emits
    (events_schema
      (map
        (event :enum (node_discovered fetch_started fetch_completed fetch_skipped robots_blocked compliance_update))
        (url :url)
        (source_url :url?)
        (depth :int?)
        (domain :string?)
        (timestamp :int)
        (reason :string?)
        (meta :map?)))))
