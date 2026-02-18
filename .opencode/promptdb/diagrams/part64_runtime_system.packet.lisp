;; .opencode/promptdb/diagrams/part64_runtime_system.packet.lisp
;; Companion packet for the Part 64 Runtime System diagram + open questions.
;; DATA-ONLY. One top-level form.

(packet
  (v "opencode.packet/v1")
  (id "diagram:part64-runtime-system:v1")
  (kind :diagram)
  (title "Part 64 Runtime System — Map + Open Questions")
  (tags [:ημ :part64 :runtime :nexus :daemoi :nooi :catalog :ws :drift :receipts])

  ;; Optional: attach the generated image artifact by path/uri in your world.
  ;; Replace the uri with your canonical artifacts path if you store images elsewhere.
  (links
    (refs
      ["artifact:image:/mnt/data/A_detailed_flowchart-style_diagram_illustrates_the.png"
       "runtime:http://127.0.0.1:8787/"
       "runtime:http://127.0.0.1:8787/api/catalog"
       "runtime:ws://127.0.0.1:8787/ws"
       ".opencode/promptdb/00_wire_world.intent.lisp"
       ".opencode/promptdb/01_validate_wire_world.intent.lisp"
       ".opencode/promptdb/contracts/ui.contract.lisp"
       ".opencode/promptdb/contracts/presence-say.contract.lisp"
       ".opencode/promptdb/contracts/receipts.v2.contract.lisp"
       "receipts.log"]))

  (body
    (diagram
      (summary
        "Canonical Part 64 runtime: PromptDB packets index into catalog; presences compile to say-intent;
         drift/scan summarizes blocked gates from receipts + artifact presence; push-truth dry-run predicts needs/drifts;
         websocket broadcasts state. Receipts are the μ-proof boundary.")

      (components
        (component
          (id promptdb)
          (kind :store)
          (uri ".opencode/promptdb/")
          (contains
            ["*.intent.lisp" "*.contract.lisp" "manifest.lith" "receipts policy packets"])
          (invariants
            ["one-top-level-form" "data-only" "header-required:(v id kind title tags body)"]))

        (component
          (id world-web)
          (kind :service)
          (uri "code/world_web.py")
          (api
            (http
              ["/api/catalog"
               "/api/eta-mu-ledger"
               "/api/presence/say"
               "/api/drift/scan"
               "/api/push-truth/dry-run"])
            (ws ["/ws"]))
          (outputs
            ["catalog items include promptdb packet metadata"
             "say_intent + rendered_text"
             "blocked_gates + drifts + receipts_parse"
             "push-truth needs + predicted_drifts + artifacts"]))

        (component
          (id frontend)
          (kind :ui)
          (uri "frontend/")
          (commands
            ["/ledger ..."
             "/say <PresenceId> <text>"
             "/drift"
             "/push-truth --dry-run"])
          (invariants
            ["commands are pure dispatchers" "server is source-of-truth"]))

        (component
          (id receipts)
          (kind :log)
          (uri "receipts.log")
          (invariants
            ["append-only"
             "line-format per receipts.v2"
             "refs must include intent + contract ids for enforcement"])))

      (flows
        (flow (from promptdb) (to world-web) (via :index) (notes "collect_promptdb_packets → catalog"))
        (flow (from receipts) (to world-web) (via :scan) (notes "/api/drift/scan parses + summarizes"))
        (flow (from world-web) (to frontend) (via :http) (notes "slash commands call endpoints"))
        (flow (from world-web) (to frontend) (via :ws) (notes "catalog/drift/presence updates")))

      ;; These are the open questions represented on the diagram.
      ;; Keep them as IDs so they can be referenced by other packets/presences.
      (open-questions
        (q (id q.task-queue) (text "How should tasks be queued (persisted, deduped, replayed)?"))
        (q (id q.drift-sampling) (text "How do we sample drifts: periodic scan, fs watchers, ws events, or hybrid?"))
        (q (id q.refresh-rate) (text "How often refresh PromptDB → catalog index; what’s the debounce policy?"))
        (q (id q.prediction-verification) (text "How do we verify predicted drifts/needs (test oracle, receipts oracle)?"))
        (q (id q.proofs-required) (text "What proofs are required for :push-truth beyond zip+host+manifest?")))

      ;; A small, practical TODO list that can become tasks later.
      (next-actions
        (todo "Define task queue semantics (in-memory vs file-backed) and emit receipts for enqueue/dequeue.")
        (todo "Add file watcher Daemon (or polling) for promptdb refresh; expose refresh stats in /api/catalog.")
        (todo "Make push-truth proof schema explicit: required refs + hashes + host handle in manifest.lith.")
        (todo "Wire Presence:KeeperOfContracts to speak when any open-questions remain unresolved for a gate.")))))
