;; .opencode/promptdb/01_validate_wire_world.intent.lisp
;; DATA-ONLY executable validation packet for the completion report.
;; Intent: run deterministic checks against canonical Part 64 runtime (8787),
;; verify endpoints + WS + PromptDB catalog + command paths, then append receipts.

(packet
  (v "opencode.packet/v1")
  (id "validate:wire-world:part64")
  (kind :intent)
  (title "Validate Wire-World Completion Report (Part 64 Canonical)")
  (tags [:ημ :validation :receipts :promptdb :ui :presence-say :drift :push-truth])

  (routing
    (target :eta-mu-world)
    (handler :validate)
    (mode :apply)) ; validation should emit receipts; keep side effects to receipts.log only

  (slots
    (owner "Err")
    (dod "All validation checks pass; receipts appended: :catalog, :test-run, :build, :decision (validation summary).")
    (options ["run-only-http+ws" "run-http-only" "run-frontend-build-only"])
    (evidence
      [".fork_Π_ημ_frags/ημ_op_mf_part_64/code/world_web.py"
       ".fork_Π_ημ_frags/ημ_op_mf_part_64/frontend/src/App.tsx"
       ".opencode/promptdb/00_wire_world.intent.lisp"
       ".opencode/promptdb/contracts/ui.contract.lisp"
       ".opencode/promptdb/contracts/presence-say.contract.lisp"
       ".opencode/promptdb/contracts/receipts.v2.contract.lisp"
       "receipts.log"]))

  (body
    (validate

      ;; --- Canonical runtime availability (hard requirement) ----------------
      (runtime
        (base-url "http://127.0.0.1:8787")
        (must
          (http-get "/" (expect (status 200)))
          (http-get "/api/catalog" (expect (status 200)))
          (ws-connect "ws://127.0.0.1:8787/ws" (expect (connected true)))))

      ;; --- Endpoint contract checks ----------------------------------------
      (http
        (must
          ;; /api/catalog includes PromptDB packets
          (http-get "/api/catalog"
            (expect
              (json-has-keys ["timestamp" "count" "items"])
              (json-items-any
                (where
                  (and
                    (= kind "packet") ; allow either "packet" or ":packet" depending on impl
                    (or (contains uri ".opencode/promptdb/")
                        (contains path ".opencode/promptdb/")))))))

          ;; /api/eta-mu-ledger still works
          (http-post "/api/eta-mu-ledger"
            (json {"text":"we should ship this\nproof in artifacts/audio/test.wav"})
            (expect (status 200) (json-has-keys ["rows" "jsonl"])))

          ;; /api/presence/say returns say_intent + rendered_text
          (http-post "/api/presence/say"
            (json {"presence_id":"MageOfReceipts" "text":"validate state -> say"})
            (expect
              (status 200)
              (json-has-keys ["say_intent" "rendered_text"])
              (json-path-has "say_intent.facts")
              (json-path-has "say_intent.asks")
              (json-path-has "say_intent.repairs")))

          ;; /api/drift/scan returns receipts parse + blocked gates summary
          (http-post "/api/drift/scan"
            (json {"scope":[".opencode/promptdb","receipts.log"]})
            (expect
              (status 200)
              (json-has-keys ["receipts_parse" "blocked_gates" "drifts"])))

          ;; /api/push-truth/dry-run returns gate status + needs + predicted drifts
          (http-post "/api/push-truth/dry-run"
            (json {"origin_rev":"HEAD"})
            (expect
              (status 200)
              (json-has-keys ["gate" "needs" "predicted_drifts" "artifacts"])))))

      ;; --- Frontend command path checks ------------------------------------
      ;; Minimal: build succeeds (already claimed). Optional: grep for commands.
      (frontend
        (must
          (exec
            (cwd ".fork_Π_ημ_frags/ημ_op_mf_part_64/frontend")
            (cmd ["npm" "run" "build"]))
          (file-contains
            ".fork_Π_ημ_frags/ημ_op_mf_part_64/frontend/src/App.tsx"
            ["/say" "/drift" "/push-truth" "/ledger"])))


      ;; --- Tests ------------------------------------------------------------
      (tests
        (must
          (exec
            (cwd ".fork_Π_ημ_frags/ημ_op_mf_part_64")
            (cmd ["python" "-m" "pytest" "code/tests/test_world_web_pm2.py"])))))

      ;; --- Receipts emission policy ----------------------------------------
      ;; Append validation receipts ONLY after all must-checks pass.
      (receipts
        (append-only "receipts.log")
        (format :receipt-line) ; per receipts.v2 contract
        (emit-on-pass
          (receipt
            (kind :catalog)
            (refs
              ["http://127.0.0.1:8787/api/catalog"
               ".opencode/promptdb/00_wire_world.intent.lisp"
               ".opencode/promptdb/contracts/ui.contract.lisp"]))
          (receipt
            (kind :test-run)
            (refs
              ["code/tests/test_world_web_pm2.py"
               "pytest:code/tests/test_world_web_pm2.py"]))
          (receipt
            (kind :build)
            (refs
              ["frontend/npm run build"
               ".fork_Π_ημ_frags/ημ_op_mf_part_64/frontend"]))
          (receipt
            (kind :decision)
            (refs
              ["validation:wire-world:part64"
               ".opencode/promptdb/01_validate_wire_world.intent.lisp"])
            (note "Validated Part64 wire-world report: runtime+ws+endpoints+catalog+tests+build"))))

      ;; --- Expected stdout summary (for chat) -------------------------------
      (expects
        (prints
          (apply
            ["checks:pass/fail"
             "http:verified"
             "ws:verified"
             "catalog:promptdb-items:count"
             "presence-say:shape:ok"
             "drift-scan:shape:ok"
             "push-truth-dry-run:shape:ok"
             "tests:pass"
             "frontend:build:pass"
             "receipts:appended:list"])))))
