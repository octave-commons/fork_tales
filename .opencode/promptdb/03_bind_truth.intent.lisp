(packet
  (v "opencode.packet/v1")
  (id "truth-binding:part64")
  (kind :intent)
  (title "Bind Truth Name + Operators + Invariants")
  (tags [:truth :eta-mu :proof :gates :receipts])

  (routing
    (target :eta-mu-world)
    (handler :orchestrate)
    (mode :apply))

  (slots
    (owner "Err")
    (dod "Truth protocol + contract exist, name binding is explicit, and receipts cite the truth artifacts.")
    (options ["apply now" "dry-run: protocol-only"])
    (evidence
      [".opencode/protocol/truth.v1.lisp"
       ".opencode/promptdb/contracts/truth-layer.contract.lisp"
       "receipts.log"]))

  (body
    (orchestrate
      (declare
        (truth
          (name "Gates_of_Truth")
          (glyphs ["名" "真" "理" "伪"])
          (operator-path ["truth/mint" "truth/guard"])
          (world-scope-key :ctx/ω-world)
          (status-enum [:proved :refuted :undecided])
          (proof-kinds [:logic/bridge :evidence/record :score/run :gov/adjudication :trace/record])))

      (contracts
        (create
          (file ".opencode/protocol/truth.v1.lisp")
          (form (protocol truth.v1)))
        (create
          (file ".opencode/promptdb/contracts/truth-layer.contract.lisp")
          (form (contract "promethean.truth-binding/v1"))))

      (receipts
        (append-only "receipts.log")
        (emit
          (receipt
            (kind :decision)
            (refs
              [".opencode/promptdb/03_bind_truth.intent.lisp"
               ".opencode/protocol/truth.v1.lisp"
               ".opencode/promptdb/contracts/truth-layer.contract.lisp"]))))))

  (expects
    (prints
      (apply
        ["truth-protocol-bound"
         "truth-contract-registered"
         "receipts-appended"])))))
