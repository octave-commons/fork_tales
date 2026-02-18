(contract "promethean.presence-say/v1"
  (mission "Compile presence-state + nexus slice into say-intent; LLM renders only.")
  (say-intent
    (facts :vector)
    (asks :vector)
    (repairs :vector)
    (constraints (no-new-facts true) (cite-refs true) (max-lines 8)))
  (compile (in [:presence-state :nexus-slice]) (out :say-intent))
  (render  (in [:say-intent]) (out :text)))
