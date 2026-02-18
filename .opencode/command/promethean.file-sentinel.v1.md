(contract "promethean.file-sentinel/v1"
  (mission
    "Continuously watch allowed paths. Convert file changes into receipts.
     Raise drift when changes lack anchors (owner/DoD/decision note/test).")

  (scope
    (watch-roots ["artifacts/" "docs/" "code/" "frontend/"])
    (ignore-globs ["**/node_modules/**" "**/.git/**" "**/dist/**" "**/__pycache__/**"])
    (max-file-bytes 2000000)
    (hash-algo "sha256"))

  (event
    (file-event
      (id :string)
      (ts :iso8601)
      (kind :keyword)
      (path :string)
      (old-path :string?)
      (bytes :int?)
      (hash :string?)
      (content-snippet :string?)
      (source :keyword)))

  (receipt
    (id :string)
    (ts :iso8601)
    (kind :keyword)
    (refs :vector)
    (hash :string?)
    (meta :map))

  (anchor
    (id :string)
    (owner :symbol)
    (dod :string)
    (deadline :iso8601?)
    (links :vector)
    (paths :vector)
    (status :keyword))

  (drift
    (id :string)
    (ts :iso8601)
    (severity :int)
    (kind :keyword)
    (paths :vector)
    (evidence :vector)
    (needs :set)
    (recommended :string))

  (invariants
    (anchored-changes
      (description "Any changed file must be covered by an active anchor.")
      (repair "Create/extend an anchor: Owner/DoD + paths; link receipts/tests; commit note."))

    (receipts-for-artifacts
      (description "Any artifact/* output must have a receipt (hash + origin).")
      (repair "Hash artifact; write receipt entry; link to producing commit/test log."))

    (no-unsafe-path-writes
      (description "Writes outside allowlisted roots are flagged.")
      (repair "Move output into allowed root or add explicit scope exception."))

    (no-huge-blobs
      (description "Large binary changes require explicit approval + receipt.")
      (repair "Store externally or add LFS; attach receipt + justification note.")))

  (derive
    (unanchored-change
      (when (and (file-event? e)
                 (in-scope? e.path)
                 (not (covered-by-open-anchor? e.path))))
      (emit (drift :unanchored-change
                   :severity 6
                   :paths [e.path]
                   :evidence [e.id]
                   :needs #{:owner :dod :receipt}
                   :recommended "Create anchor for path + add receipt hash; then commit.")))

    (artifact-missing-receipt
      (when (and (file-event? e)
                 (path-prefix? e.path "artifacts/")
                 (not (receipt-exists-for? e.path))))
      (emit (drift :missing-receipt
                   :severity 7
                   :paths [e.path]
                   :evidence [e.id]
                   :needs #{:receipt}
                   :recommended "Generate sha256 + append receipt; link producing run.")))

    (unsafe-path
      (when (and (file-event? e)
                 (not (in-scope? e.path))))
      (emit (drift :unsafe-path
                   :severity 8
                   :paths [e.path]
                   :evidence [e.id]
                   :needs #{:decision}
                   :recommended "Move file into allowlist or document scope exception.")))

    (huge-blob
      (when (and (file-event? e)
                 (> e.bytes 2000000)))
      (emit (drift :large-blob
                   :severity 8
                   :paths [e.path]
                   :evidence [e.id]
                   :needs #{:decision :receipt}
                   :recommended "Use LFS/external store + justification + receipt."))))

  (songform
    (id path-ward)
    (modes (:warning :audit :praise :ritual))
    (hooks ("name the path" "anchor the change" "show the hash" "push truth"))
    (jp-hooks ("経路を名乗れ" "変更に錨を" "ハッシュを示せ" "真理へ押し込め"))
    (call-and-response? true))

  (action :push-truth
    (in {:origin-rev :git-sha
         :notes :string?
         :anchor-id :string?})
    (out {:pi-zip :artifact
          :gist :host-ref
          :manifest :artifact
          :receipt :receipt
          :ledger :mu-proof}))

  (act
    (detect (in :file-event*) (out :drift*))
    (receiptize (in :file-event*) (out :receipt*))
    (sing (in :drift) (out :song))
    (gate
      (on [:publish :release]
          (requires [:anchored-changes :receipts-for-artifacts]))
      (on [:push-truth]
          (requires
            [:oss-license-present :source-available :attribution-ok
             :artifact-hash-present :tests-linked :decision-log-written
             :anchored-changes :receipts-for-artifacts :no-unsafe-path-writes :no-huge-blobs]))))

  (update "promethean.file-sentinel/v1"
    (policy
      (gate
        (push-truth
          (requires
            [:oss-license-present :source-available :attribution-ok
             :artifact-hash-present :tests-linked :decision-log-written
             :anchored-changes :receipts-for-artifacts :no-unsafe-path-writes :no-huge-blobs])))
      (feedback
        (on-block :push-truth :sing)
        (on-pass :push-truth :praise)))))
