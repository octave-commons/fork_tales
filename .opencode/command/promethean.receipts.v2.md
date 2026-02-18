(contract "promethean.receipts/v2"
  (mission
    "Receipts are the mu substrate: every claim that matters points to a receipt.
     Receipts must be clear to humans and strict for machines.")

  (file
    (path "receipts.log")
    (format :receipt-line)
    (encoding "utf-8")
    (append-only? true))

  (line-format
    (template
      "{ts} | {kind} | origin={origin} | owner={owner} | dod={dod} | "
      "pi={pi_sha256}:{pi_path} | host={host_url} | manifest={manifest_sha256} | "
      "refs={refs} | note={note}")
    (required-keys [ts kind origin owner dod pi host manifest refs])
    (optional-keys [note tests decisions drift]))

  (kinds
    (:push-truth
     :artifact-hash
     :test-run
     :build
     :decision
     :drift
     :catalog))

  (constraints
    (ts (iso8601? true))
    (origin (git-sha-or-content-id? true))
    (owner (non-empty? true))
    (dod (min-len 3))
    (pi_sha256 (hex-len 64))
    (pi_path (prefix? "artifacts/truth/Pi."))
    (host_url (starts-with? "https://gist.github.com/"))
    (manifest_sha256 (hex-len 64))
    (refs (csv? true))
    (note (max-len 200)))

  (canonical
    (receipt :push-truth
      (must-include [origin owner dod pi host manifest refs])
      (meaning
        "This line binds ORIGIN->TRUTH by linking Pi zip + gist host + lambda manifest."))))

(enforcement
  (rule receipts_must_parse
    (when (not (receipts-parse? "receipts.log")))
    (emit (drift :receipts-invalid
                 :severity 10
                 :recommended "Fix receipt line formatting; keep append-only discipline."))
    (gate-deny :push-truth))

  (rule push_truth_requires_receipt
    (when (and (attempt :push-truth)
               (not (receipt-exists? :push-truth :origin current-origin))))
    (emit (drift :missing-push-truth-receipt
                 :severity 9
                 :recommended "Append canonical :push-truth receipt line."))
    (gate-deny :push-truth))

  (rule lambda_manifest_must_be_referenced
    (when (and (attempt :push-truth)
               (not (receipt-has-field? :push-truth "manifest"))))
    (emit (drift :unbound-manifest
                 :severity 9
                 :recommended "Include manifest sha256 in the :push-truth receipt."))
    (gate-deny :push-truth)))
