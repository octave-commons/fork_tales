;; .opencode/promptdb/00_wire_world.intent.lisp
;; Executable (routable) intent packet.
;; DATA-ONLY: no eval. Agents interpret via meta-contract eta-mu.intent-lisp/v1.

(packet
  (v "opencode.packet/v1")
  (id "wire-world:part64")
  (kind :intent)
  (title "Wire PromptDB -> Simulation -> Interface")
  (tags [:eta-mu :promethean :nexus :daemoi :nooi :ui :receipts :push-truth])

  (routing
    (target :eta-mu-world)
    (handler :orchestrate)
    (mode :dry-run))

  (slots
    (owner "Err")
    (dod "PromptDB packets indexed as Nexus; Presences compiled to say-intent; UI panels + WS events; receipts.log v2")
    (options
      ["dry-run: generate contracts only"
       "apply: write files + add endpoints + wire frontend"
       "apply-min: backend only"])
    (evidence
      [".opencode/promptdb/" "receipts.log" "manifest.lith" "*.intent.lisp"]))

  (body
    (orchestrate
      (declare
        (promptdb (root ".opencode/promptdb")
                  (one-form-per-file true)
                  (data-only true)
                  (packet-header-required true))
        (truth (unit :pi-package+host)
               (manifest "manifest.lith")
               (intent-ext ".intent.lisp")
               (receipt-file "receipts.log")))

      (contracts
        (create
          (file ".opencode/promptdb/contracts/ui.contract.lisp")
          (form (contract "promethean.ui-panels/v1")))
        (create
          (file ".opencode/promptdb/contracts/presence-say.contract.lisp")
          (form (contract "promethean.presence-say/v1")))
        (create
          (file ".opencode/promptdb/contracts/receipts.v2.contract.lisp")
          (form (contract "promethean.receipts/v2"))))

      (tasks
        (task
          (id "backend.api.presence-say")
          (write
            (file "code/world_web.py")
            (adds ["POST /api/presence/say" "POST /api/drift/scan" "POST /api/push-truth/dry-run"]))
          (tests
            (add "code/tests/test_world_web_pm2.py" "presence say returns say-intent + rendered text")))

        (task
          (id "frontend.panels")
          (write
            (file "frontend/src/App.tsx")
            (adds ["/say <PresenceId>" "/drift" "/push-truth --dry-run"]))
          (tests
            (add "frontend" "build passes")))

        (task
          (id "promptdb.index")
          (write
            (file "code/world_web.py")
            (adds ["scan .opencode/promptdb one-form-per-file" "catalog includes packets" "ws emits catalog"]))
          (tests
            (add "code/tests/test_world_web_pm2.py" "promptdb scan populates catalog"))))))

  (expects
    (prints
      (dry-run
        ["files-to-write" "endpoints-to-add" "tests-to-add" "receipts-required" "gates-affected"])
      (apply
        ["files-written" "endpoints-added" "tests-added" "receipts-appended" "verification-commands"]))))
