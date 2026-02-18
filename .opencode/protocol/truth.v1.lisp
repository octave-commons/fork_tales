(protocol truth.v1
  (sigil-registry
    (sigil
      (glyph "名")
      (id :meta/name)
      (arity 2)
      (surface (name sym value))
      (core (meta/name :sym sym :value value))
      (ascii "NAME"))
    (sigil
      (glyph "真")
      (id :truth/record)
      (arity 1)
      (surface
        (truth id
          :world world-id
          :claim φ
          :status status
          :κ κ
          :proof proof
          :ctx ctx))
      (core
        (truth/record
          :id id
          :world world-id
          :claim φ
          :status status
          :fact/κ κ
          :proof proof
          :ctx ctx))
      (ascii "TRUTH"))
    (sigil
      (glyph "理")
      (id :truth/reason)
      (arity 1)
      (surface (reason proof))
      (core (truth/reason :proof proof))
      (ascii "REASON"))
    (sigil
      (glyph "伪")
      (id :truth/false)
      (arity 1)
      (surface
        (false id
          :world world-id
          :claim φ
          :proof proof
          :κ κ
          :ctx ctx))
      (core
        (truth/false
          :id id
          :world world-id
          :claim φ
          :proof proof
          :fact/κ κ
          :ctx ctx))
      (ascii "FALSE")))

  (record meta/name.v1
    (required
      (id :symbol)
      (symbol :symbol)
      (meaning :symbol)
      (law :string))
    (optional
      (glyph :string)
      (aliases :list-of :string)
      (ctx :map)))

  (record truth/record.v1
    (required
      (id :symbol)
      (world :symbol)
      (claim :sexp)
      (status :enum (proved refuted undecided))
      (fact/κ :float)
      (proof :list-of :map))
    (optional
      (ctx :map)
      (minted-by :symbol)
      (minted-at :int)
      (notes :string)))

  (record truth/reason.v1
    (required
      (proof :list-of :map))
    (optional
      (logic :map)
      (ctx :map)))

  (record truth/false.v1
    (required
      (id :symbol)
      (world :symbol)
      (claim :sexp)
      (proof :list-of :map)
      (fact/κ :float))
    (optional
      (ctx :map)
      (notes :string)))

  (binding
    (name-of-truth
      (meta/name.v1
        (id gates_of_truth)
        (symbol Gates_of_Truth)
        (glyph "真")
        (meaning :truth/record)
        (aliases ["真理" "truth" "道"])
        (law "Truth must cite proof refs and be world-scoped by ω context."))))

  (invariants
    (must
      (truth-world-scoped true)
      (truth-proof-required true)
      (truth-status-calibrated true)
      (truth-proof-kinds-subset [:logic/bridge :evidence/record :score/run :gov/adjudication :trace/record]))
    (must_not
      (mint-truth-from-sim-bead true)
      (mint-truth-without-ctx-world true)
      (mint-truth-without-proof true))
    (promotion
      (rule "embeddings/beads may influence interpretation but cannot mint truth records directly")
      (rule "truth status changes require new proof refs and fresh receipt rows")))

  (operators
    (op truth/mint
      (input
        (world :symbol)
        (claim :sexp)
        (status :enum (proved refuted undecided))
        (κ :float)
        (proof :list-of :map)
        (ctx :map))
      (output :truth/record.v1)
      (requires
        (ctx-key :ctx/ω-world)
        (proof-non-empty true)
        (persist-flow :行→動→記)))

    (op truth/guard
      (input
        (θ :float)
        (claim :sexp)
        (world :symbol))
      (output :bool)
      (requires
        (status :proved)
        (κ-gte-threshold true)
        (same-world true)
        (same-claim true))))

  (judgment
    (truth?
      (input
        (world :symbol)
        (claim :sexp))
      (definition
        "truth?(W, φ) holds iff there exists truth/record.v1 with world=W, claim=φ, status=proved.")
      (evidence
        (requires-proof true)
        (requires-receipts true)
        (requires-world-scope true)))))
