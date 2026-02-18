(contract "promethean.truth-manifest.lith/v1"
  (mission
    "Lith balances parentheses: every Truth package must have a single canonical manifest
     that is (a) parseable, (b) self-describing, (c) binds Pi(zip) to Host(gist),
     and (d) carries the minimal receipts to upgrade eta->mu.")

  (file
    (name "manifest.lith")
    (format :lith-sexp)
    (encoding "utf-8")
    (required? true))

  (schema
    (manifest
      (v "truth.manifest.lith/v1")
      (origin
        (rev :string)
        (branch :string?)
        (created-at :iso8601)
        (owner :string?))
      (pi
        (path :string)
        (sha256 :string)
        (bytes :int)
        (includes ["LICENSE" "README.md" "receipts.log" "manifest.lith"]))
      (host
        (kind :github-gist)
        (id :string)
        (url :string)
        (mirrors ["receipts.log" "manifest.lith"]))
      (receipts
        (manifest-sha256 :string)
        (receipt-entry :string))
      (claims
        (summary :string)
        (dod :string?)
        (tests [:string*])
        (decisions [:string*])
        (license :string))
      (contents
        (files [:string*])
        (counts :map))))

  (invariants
    (balanced-parens
      (description "manifest.lith must parse as a single well-formed s-expression.")
      (repair "Fix parentheses; no stray forms; keep it data-only."))

    (single-source-of-truth
      (description "Exactly one manifest.lith per Truth unit; included in Pi and mirrored to Host.")
      (repair "Ensure zip includes manifest.lith; gist contains the same exact text."))

    (binding-complete
      (description "manifest.lith must bind zip sha256 + gist url + origin rev in one place.")
      (repair "Fill pi.sha256, host.url, origin.rev; recompute manifest-sha256."))

    (receipt-glue-present
      (description "receipts.log must contain a :push-truth entry referencing manifest + zip + gist.")
      (repair "Append receipt entry; include manifest-sha256 and zip-sha256 and gist-url."))

    (lambda-signed
      (description
        "manifest.lith must include its own sha256 in receipts.manifest-sha256,
         computed over the exact UTF-8 bytes of the file.")
      (repair
        "Compute sha256(manifest.lith); write into (receipts (manifest-sha256 ...));
         re-zip Pi; update gist mirror; append push-truth receipt.")))

  (songform
    (id manifest-lith)
    (modes (:audit :repair :praise))
    (hooks ("balance the parens" "one form, one truth" "bind Pi to host"))
    (jp-hooks ("括弧を均せ" "一つの形、一つの真理" "Piをホストへ結べ"))
    (call-and-response? false))

  (update "promethean.truth-manifest.lith/v1"
    (bindings
      (presence (id Lith)
        (field manifest-lith)
        (domain :lisp)
        (role :balancer)
        (sigil "λ")
        (jp-alias "括弧の均衡者")))
    (log-prefix
      (for :manifest-lith) "[λ]")))

(manifest
  (v "truth.manifest.lith/v1")
  (origin (rev "abc1234") (created-at "2026-02-15T06:12:00Z") (owner "Err"))
  (pi (path "artifacts/truth/Pi.1a2b3c4d.zip") (sha256 "1a2b...") (bytes 123456))
  (host (kind :github-gist) (id "deadbeef") (url "https://gist.github.com/..."))
  (receipts (manifest-sha256 "55aa...") (receipt-entry "push-truth:..."))
  (claims (summary "Ledger + sentinel wired") (license "GNU-GPL-v3"))
  (contents (files ["LICENSE" "README.md" "receipts.log" "manifest.lith"]) (counts (files 4))))

(rule block-push-truth-if-lambda-invalid
  (when (not (valid-manifest-sha? "manifest.lith")))
  (emit (drift :manifest-invalid
               :severity 9
               :needs #{:receipt}
               :recommended "Recompute manifest-sha256 and update Pi + gist."))
  (gate-deny :push-truth))
