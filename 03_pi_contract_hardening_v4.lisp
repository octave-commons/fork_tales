;; Π Contract Hardening v4
;; Next logical step: cryptographic attestation + capability negotiation + deterministic replay.
;; This does NOT require trusted hardware; it improves tamper-evidence and drift control.

(contract Π_v4

  ;; ============================================================
  ;; 0. KEY IDEA
  ;; ============================================================
  ;; Π export is accepted only if:
  ;;  - each shard signs its export
  ;;  - capabilities are declared + versioned
  ;;  - replay metadata exists to reproduce key derivations
  ;;  - promotion gates are enforced deterministically


  ;; ============================================================
  ;; 1. ATTESTATION (SIGNATURES)
  ;; ============================================================
  (attestation
    ;; Each shard/agent has an identity keypair.
    ;; Sign the manifest hash (not the whole zip) to keep it stable.

    (keyring
      (require
        (key :brain   (type ed25519) (kid :string))
        (key :senses  (type ed25519) (kid :string))
        (key :memory  (type ed25519) (kid :string))
        (key :voice   (type ed25519) (kid :string))
        (key :router  (type ed25519) (kid :string)))
      (rotation
        (policy "allow, but must publish new kid + revoke old")
        (require-revocation-log true)))

    (signed
      (object "manifest.json")
      (hash-alg "sha256")
      (signatures
        ;; one signature per required agent
        (sig (agent :brain)  (kid) (sig64))
        (sig (agent :senses) (kid) (sig64))
        (sig (agent :memory) (kid) (sig64))
        (sig (agent :voice)  (kid) (sig64))
        (sig (agent :router) (kid) (sig64)))

      (reject-if
        (missing-signature true)
        (unknown-kid true)
        (sig-verify-fails true))))


  ;; ============================================================
  ;; 2. CAPABILITY NEGOTIATION (VERSIONED SKILL SET)
  ;; ============================================================
  (capabilities
    ;; Exporting shard declares what it can do, with versions.
    ;; Receiver declares required minima.

    (declare
      (cap (name :intent-parse)      (ver "1.x") (schema-hash :sha256))
      (cap (name :promptdb-compile)  (ver "1.x") (schema-hash :sha256))
      (cap (name :state-snapshot)    (ver "2.x") (schema-hash :sha256))
      (cap (name :runtime-verify)    (ver "2.x") (schema-hash :sha256))
      (cap (name :diff-report)       (ver "1.x") (schema-hash :sha256))
      (cap (name :vector-manifest)   (ver "1.x") (schema-hash :sha256))
      (cap (name :replay-metadata)   (ver "1.x") (schema-hash :sha256)))

    (negotiate
      ;; Receiver requirements (minimum versions)
      (require
        (min :state-snapshot  "2.0")
        (min :runtime-verify  "2.0")
        (min :vector-manifest "1.0")
        (min :replay-metadata "1.0"))

      (reject-if
        (capability-missing true)
        (version-too-low true)
        (schema-hash-mismatch true))))


  ;; ============================================================
  ;; 3. DETERMINISTIC REPLAY METADATA
  ;; ============================================================
  (replay
    ;; Enough information to reproduce *derivations* used to generate the export.
    ;; Not to recreate the entire world, but to make the export checkable.

    (require
      (clock
        (now-iso8601)
        (timezone "America/Phoenix")
        (monotonic-ns :int))

      (inputs
        (export-scope :enum (full delta promptdb-only services-only))
        (baseline-Π-sha :sha256?)
        (seed
          ;; if any seeded selection occurred, record the seed source
          (method :enum (sha256_text_datehour fixed none))
          (value :string?))
        (parameters
          (temperatures per-agent)
          (top_p per-agent)
          (max_tokens per-agent)))

      (tool-trace
        ;; Minimal, privacy-safe trace: list of tool calls + hashes of inputs/outputs.
        ;; Allows verifying that a claim corresponds to some concrete action.
        (events jsonl)
        (redactions
          (secrets true)
          (pii true)
          (raw-payloads false))))

    (reject-if
      (missing-replay-metadata true)
      (tool-trace-missing true)))


  ;; ============================================================
  ;; 4. REQUEST CONTRACTS (UPGRADED)
  ;; ============================================================
  (requests

    (request :Π-export
      (requires
        (manifest signed)
        (capabilities declared)
        (replay metadata)
        (runtime-verification fresh<60s)
        (vector-manifest counts+checksums)
        (promptdb integrity)
        (errors explicit))
      (reject-if
        (ws-untested true)
        (inventory!=embedded true)
        (missing-signature true)
        (capability-missing true)
        (missing-replay-metadata true)))

    (request :Π-promote
      (requires
        (Π-sha)
        (verification-report)
        (promotion-decision :enum (allow sandbox reject))
        (reasons :list))
      (invariant
        "Promotion is a receiver-side act only; exporter cannot self-promote.")))


  ;; ============================================================
  ;; 5. PROMOTION GATES (TIGHTENED)
  ;; ============================================================
  (promotion

    (allow-only-if
      (attestation-verified true)
      (capabilities-negotiated true)
      (replay-present true)
      (manifest-matches true)
      (runtime-verification-passed true)
      (critical-errors none))

    (sandbox-if
      (ws-timeout true)
      (container-restart-loop true)
      (promptdb-partial true)
      (baseline-unknown true)
      (tool-trace-incomplete true)))


  ;; ============================================================
  ;; 6. FILES REQUIRED INSIDE Π ZIP
  ;; ============================================================
  (payload
    (must-include
      "manifest.json"
      "manifest.sig.json"      ;; signatures + kids
      "capabilities.json"
      "replay.json"
      "runtime_verification.json"
      "errors.jsonl"
      "world_state.sexp"
      "promptdb_files.json"
      "promptdb_prompts.jsonl" ;; or a declared external manifest
      "vector_manifest.json"   ;; counts + checksums
      "services_pm2.json"
      "services_docker.jsonl"))


  ;; ============================================================
  ;; 7. RECEIVER OUTPUT (STANDARDIZED)
  ;; ============================================================
  (receiver-report
    (emit
      (verification
        (outer-sha256)
        (manifest-sha256)
        (sig-verified per-agent)
        (capability-check)
        (replay-check)
        (runtime-check)
        (vector-check)
        (promptdb-check)
        (errors-summary))
      (decision (allow|sandbox|reject))
      (reasons :list)
      (next-actions :list)))

  (invariants
    "Attestation signs manifest hash, not the zip bytes."
    "Capabilities are versioned + schema-hashed."
    "Replay metadata is mandatory for checkability."
    "Exporter cannot self-promote."
    "Canon and Evolve lanes never merge implicitly.")


  ;; ============================================================
  ;; 8. OPEN QUESTIONS (MUST BE ANSWERED OR EXPLICITLY DEFERRED)
  ;; ============================================================
  (open-questions

    (question :authority-model
      (ask "Which shard is authoritative for conflict resolution?")
      (require-answer :enum (brain router external-human quorum unknown))
      (reject-if (unanswered true)))

    (question :vector-source-of-truth
      (ask "Is Chroma authoritative, or is vector state reconstructible from files/logs?")
      (require-answer :enum (authoritative reconstructible hybrid unknown)))

    (question :promptdb-canonicality
      (ask "Is promptdb content canonical in-repo, in-db, or generated?")
      (require-answer :enum (repo db generated hybrid unknown)))

    (question :websocket-criticality
      (ask "Is /ws required for promotion or advisory only?")
      (require-answer :enum (required advisory experimental unknown)))

    (question :container-failure-policy
      (ask "Does a restarting ML container block promotion?")
      (require-answer :enum (block sandbox ignore unknown)))

    (question :git-requirement
      (ask "Must a valid git commit exist for Π promotion?")
      (require-answer :enum (required advisory none unknown)))

    (question :schema-evolution
      (ask "How are schema changes versioned and migrated?")
      (require-answer :enum (semver strict-hash rolling manual unknown)))

    (question :key-rotation-policy
      (ask "How are attestation keys rotated and revoked?")
      (require-answer :enum (manual quorum time-based hardware-backed unknown)))

    (question :replay-depth
      (ask "How deep must deterministic replay go (derivations only vs full state rebuild)?")
      (require-answer :enum (derivations-only full-replay bounded-depth unknown)))

    (question :promotion-audit-log
      (ask "Where is the canonical promotion log stored?")
      (require-answer :enum (append-only-file vector-store git-history external-ledger unknown)))

    (policy
      "Every open question must have one of: explicit answer OR explicit 'unknown' with rationale."
      "Unknown is allowed; silent omission is not."))

)
