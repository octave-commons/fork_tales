(protocol perm.v1
  (required
    (id :symbol)
    (ts :int)
    (principal :string)
    (type :enum (grant deny revoke expire permission_request audit_policy))
    (cap :symbol)
    (intensity :number)
    (scope :sexp)
    (purpose :list-of :string)
    (ttl :sexp)
    (audit :sexp))

  (log
    (source_of_truth ".opencode/perm/log.lisp")
    (append_only true)
    (mutation_rule "Never edit prior events; only append new events."))

  (fold_rules
    (order
      "expired -> ignore"
      "revoked -> ignore"
      "deny overrides grant on scope overlap"
      "more-specific scope wins among same type"
      "newest wins on final tie")
    (default_policy "deny"))

  (check
    (input
      (action :symbol)
      (requires :sexp)
      (scope :sexp)
      (purpose :list-of :string))
    (output
      (allow :bool)
      (reason :string)
      (matched_events :list-of :symbol)))

  (doctor
    (triage_order
      (list
        "hard deny and must_not boundaries"
        "scope specificity"
        "ttl and recency"
        "least privilege escalation"
        "operator convenience"))
    (when_unsure
      (list
        "deny by default"
        "emit permission_request event"
        "request narrower scope"
        "request shorter ttl"
        "require decision+reason audit"))))
