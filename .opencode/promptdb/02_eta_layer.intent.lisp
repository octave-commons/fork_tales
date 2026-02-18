(packet
  (v "opencode.packet/v1")
  (id "eta-layer:part66")
  (kind :intent)
  (title "Establish Eta Observational Layer")
  (tags [:eta :mu :pi :protocol :ingest :receipts])

  (routing
    (target :eta-mu-world)
    (handler :orchestrate)
    (mode :apply))

  (slots
    (owner "Err")
    (dod "eta directories exist; eta.v1 protocol exists; eta-layer contract exists; receipts capture eta adoption.")
    (options ["apply now" "dry-run: docs-and-structure"])
    (evidence
      [".opencode/protocol/eta.v1.lisp"
       ".opencode/promptdb/contracts/eta-layer.contract.lisp"
       ".η/README.md"
       "receipts.log"]))

  (body
    (orchestrate
      (declare
        (eta
          (root ".η")
          (subdirs ["stream" "raw" "live"])
          (mutation-policy :append-only))
        (flow
          (stages [:eta :mu :pi])
          (rule "mu may interpret eta but must not mutate eta")))

      (contracts
        (create
          (file ".opencode/protocol/eta.v1.lisp")
          (form (protocol eta.v1)))
        (create
          (file ".opencode/promptdb/contracts/eta-layer.contract.lisp")
          (form (contract "promethean.eta-layer/v1"))))

      (receipts
        (append-only "receipts.log")
        (emit
          (receipt
            (kind :decision)
            (refs
              [".opencode/promptdb/02_eta_layer.intent.lisp"
               ".opencode/promptdb/contracts/eta-layer.contract.lisp"
               ".opencode/protocol/eta.v1.lisp"
               ".η/README.md"])))))))
