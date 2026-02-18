(contract "promethean.truth-binding/v1"
  (mission "Bind Truth as a world-scoped judged claim with proof-chain and calibrated confidence.")

  (symbols
    (name "名")
    (truth "真")
    (reason "理")
    (falsehood "伪"))

  (bindings
    (canonical-name "Gates_of_Truth")
    (meaning :truth/record)
    (law "Truth requires world scope (ω) + proof refs + receipts."))

  (invariants
    (must
      (world-scope-key :ctx/ω-world)
      (proof-required true)
      (status-enum [:proved :refuted :undecided])
      (proof-kinds-subset [:logic/bridge :evidence/record :score/run :gov/adjudication :trace/record]))
    (must_not
      (mint-from-embedding-only true)
      (mint-without-receipts true)
      (promote-sim-beads-to-truth true)))

  (operators
    (mint
      (name "truth/mint")
      (requires [:world :claim :status :κ :proof :ctx])
      (ctx-required-key :ctx/ω-world))
    (guard
      (name "truth/guard")
      (requires [:world :claim :θ])
      (passes-when [:status:proved :κ>=θ :proof-present])))

  (gates
    (targets [:push-truth :publish :release])
    (deny-when
      (missing-proof true)
      (missing-world-scope true)
      (missing-receipts true))
    (pass-when
      (truth-record-status :proved)
      (truth-record-kappa-gte-threshold true)
      (proof-chain-verifiable true)))

  (proof
    (receipts-file "receipts.log")
    (required-refs
      [".opencode/protocol/truth.v1.lisp"
       ".opencode/promptdb/03_bind_truth.intent.lisp"
       ".opencode/promptdb/contracts/truth-layer.contract.lisp"
       "receipts.log"]))

  (eta-mu
    (eta "claims and narratives about φ")
    (mu "judged truth status with proof refs and κ")))
