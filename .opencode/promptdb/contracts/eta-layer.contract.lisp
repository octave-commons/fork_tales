(contract "promethean.eta-layer/v1"
  (mission "Keep eta as raw observational substrate with minimal transformation.")

  (folders
    (eta ".η")
    (stream ".η/stream")
    (raw ".η/raw")
    (live ".η/live"))

  (invariants
    (must
      (append-only ".η/stream")
      (immutable ".η/raw")
      (snapshot-log ".η/live")
      (mu-reads-eta true))
    (must_not
      (rewrite ".η/stream")
      (rewrite ".η/raw")
      (mu-mutate-eta true)
      (pi-package-from-eta-direct true)))

  (flow
    (stages [:eta :mu :pi])
    (meaning
      (eta "raw observations and field tremors")
      (mu "interpretation and semantic linking")
      (pi "structured transport artifacts")))

  (proof
    (receipts-file "receipts.log")
    (required-refs
      [".opencode/protocol/eta.v1.lisp"
       ".opencode/promptdb/02_eta_layer.intent.lisp"
       ".η/README.md"])))
