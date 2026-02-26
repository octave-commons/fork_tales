;; SPDX-License-Identifier: GPL-3.0-or-later
;; This file is part of Fork Tales.
;; Copyright (C) 2024-2025 Fork Tales Contributors
;;
;; This program is free software: you can redistribute it and/or modify
;; it under the terms of the GNU General Public License as published by
;; the Free Software Foundation, either version 3 of the License, or
;; (at your option) any later version.
;;
;; This program is distributed in the hope that it will be useful,
;; but WITHOUT ANY WARRANTY; without even the implied warranty of
;; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;; GNU General Public License for more details.
;;
;; You should have received a copy of the GNU General Public License
;; along with this program.  If not, see <https://www.gnu.org/licenses/>.

;; Π Contract Strengthening v3
;; Purpose: Define REQUIRED capabilities the exporting shard must declare
;; before Π promotion is allowed.

(contract Π_v3

  ;; ============================================================
  ;; 1. REQUIRED AGENT DECLARATIONS
  ;; ============================================================
  (agents
    (require
      (agent :brain
        (role "simulation + decision logic")
        (heartbeat-required true)
        (state-export (goals open-loops decision-field)))

      (agent :senses
        (role "watchers + websocket + file observers")
        (must-expose (event-queue-depth last-activity-ts)))

      (agent :memory
        (role "vector + log persistence")
        (must-expose
          (vector-count)
          (collection-names)
          (embedding-model)
          (write-ahead-log-pointer)))

      (agent :voice
        (role "tts or outward synthesis")
        (must-expose (model-id status last-inference-ts)))

      (agent :router
        (role "prompt routing + command dispatch")
        (must-expose (active-routes route-table version))))

    (forbid
      (agent-without-heartbeat true)))


  ;; ============================================================
  ;; 2. REQUIRED SKILLS (EXPLICIT CAPABILITIES)
  ;; ============================================================
  (skills
    (must-implement

      (skill :intent-parse
        (input-schema (text context))
        (output-schema (intent confidence tags)))

      (skill :promptdb-compile
        (input-schema (fragment-id))
        (output-schema (compiled-prompt sha256)))

      (skill :state-snapshot
        (input-schema (scope))
        (output-schema (sexp sha256 manifest)))

      (skill :runtime-verify
        (input-schema (endpoints))
        (output-schema (status latency error?)))

      (skill :diff-report
        (input-schema (baseline-sha candidate-sha))
        (output-schema (delta summary))))

    (must-report
      (missing-skill-list)))


  ;; ============================================================
  ;; 3. REQUEST CONTRACTS (MANDATORY FOR NEXT DELIVERY)
  ;; ============================================================
  (requests

    (request :Π-export
      (requires
        (world-state full)
        (promptdb full-or-manifest)
        (services snapshot)
        (runtime-verification fresh<60s)
        (vector-manifest counts+checksums)
        (git (branch commit dirty?)))
      (reject-if
        (missing-field true)
        (stale-runtime-check true)
        (manifest-mismatch true)))

    (request :Π-delta
      (requires
        (baseline-sha)
        (changed-files sha256)
        (service-diff)
        (promptdb-diff))
      (reject-if
        (baseline-unknown true)))

    (request :agent-health
      (requires
        (heartbeat age<30s)
        (cpu mem uptime)
        (error-count last-200))
      (reject-if
        (heartbeat-stale true)))

    (request :promptdb-integrity
      (requires
        (fragment-count)
        (embedded-count)
        (rules-count)
        (sha256-per-fragment))
      (reject-if
        (inventory!=embedded true))))


  ;; ============================================================
  ;; 4. PROMOTION GATES
  ;; ============================================================
  (promotion
    (allow-only-if
      (all-agents-declared true)
      (all-required-skills-present true)
      (runtime-verification-passed true)
      (vector-store-manifest-present true)
      (no-unresolved-errors critical-only)))

    (sandbox-if
      (ws-timeout true)
      (container-restart-loop true)
      (promptdb-partial true)))


  ;; ============================================================
  ;; 5. META-INVARIANTS
  ;; ============================================================
  (invariants
    "Π is immutable once promoted."
    "Missing fields must be explicit records, not silent omission."
    "All diffs reference prior Π by sha."
    "No agent may claim health without a recent heartbeat."
    "Canon and Evolve lanes must never merge implicitly."))

