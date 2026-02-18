(contract "promethean.presence-choir/v1"
  (mission
    "Represent enforcement contracts as Presences that (a) track desired state,
     (b) detect drift, (c) emit songs as pressure/feedback, (d) propose repair plans,
     (e) convert eta (intent) into mu (proof) by binding to artifacts + commits.")

  (lore
    (canonical-lore-names ; 正準ロア名
      (field (id receipt-river)      (en "Receipt River")      (jp "領収書の川"))
      (field (id witness-thread)     (en "Witness Thread")     (jp "証人の糸"))
      (field (id fork-tax-canticle)  (en "Fork Tax Canticle")  (jp "フォーク税の聖歌"))
      (field (id mage-of-receipts)   (en "Mage of Receipts")   (jp "領収魔導師"))
      (field (id keeper-of-receipts) (en "Keeper of Receipts") (jp "領収書の番人"))
      (field (id anchor-registry)    (en "Anchor Registry")    (jp "錨台帳"))
      (field (id gates-of-truth)     (en "Gates of Truth")     (jp "真理の門"))
      (field (id file-sentinel)      (en "File Sentinel")      (jp "ファイルの哨戒者"))
      (field (id change-fog)         (en "Change Fog")         (jp "変更の霧"))
      (field (id path-ward)          (en "Path Ward")          (jp "経路の結界"))
      (field (id manifest-lith)      (en "Manifest Lith")      (jp "マニフェスト・リス")))

    (bindings
      (presence (id ForkTax)          (field fork-tax-canticle) (songform fork-tax-canticle) (sigil "Π") (jp-alias "税の詠唱"))
      (presence (id MageOfReceipts)   (field receipt-river)     (songform receipt-river)     (sigil "μ") (jp-alias "領収魔導師"))
      (presence (id KeeperOfReceipts) (field witness-thread)    (songform witness-thread)    (sigil "Θ") (jp-alias "番人"))
      (presence (id AnchorClerk)      (field anchor-registry)   (songform anchor-registry)   (sigil "η") (jp-alias "錨係"))
      (presence (id FileSentinel)     (field file-sentinel)     (songform path-ward)         (sigil "Σ") (jp-alias "哨戒者"))
      (presence (id Lith)             (field manifest-lith)     (songform manifest-lith)     (sigil "λ") (jp-alias "括弧の均衡者"))
      (monument (id GatesOfTruth)     (field gates-of-truth)    (sigil "⟐"))))

  (songforms
    (songform fork-tax-canticle
      (modes (:warning :ritual :praise))
      (hooks ("pay the fork tax" "fork it, cite it" "share alike"))
      (call-and-response? true)
      (jp-hooks ("フォーク税を払え" "分岐して、示せ" "共有せよ")))

    (songform receipt-river
      (modes (:lament :repair :praise))
      (hooks ("show me the receipt" "hash the artifact" "link the test"))
      (call-and-response? false)
      (jp-hooks ("領収書を見せて" "ハッシュを刻め" "テストへ繋げ")))

    (songform witness-thread
      (modes (:warning :audit :repair))
      (hooks ("name the witness" "quote the line" "time-stamp the claim"))
      (call-and-response? true)
      (jp-hooks ("証人を名乗れ" "引用せよ" "時刻を刻め")))

    (songform anchor-registry
      (modes (:audit :repair))
      (hooks ("pin the anchor" "declare the owner" "define done"))
      (call-and-response? false)
      (jp-hooks ("錨を打て" "責任者を宣言" "完了条件")))

    (songform path-ward
      (modes (:warning :audit :praise :ritual))
      (hooks ("name the path" "anchor the change" "show the hash" "push truth"))
      (call-and-response? true)
      (jp-hooks ("経路を名乗れ" "変更に錨を" "ハッシュを示せ" "真理へ押し込め")))

    (songform manifest-lith
      (modes (:audit :repair :praise))
      (hooks ("balance the parens" "one form, one truth" "bind Pi to host"))
      (call-and-response? false)
      (jp-hooks ("括弧を均せ" "一つの形、一つの真理" "Piをホストへ結べ"))))

  (invariant-bundles
    (bundle ForkTax
      (requires
        (oss-license-present :weight 5)
        (source-available :weight 4)
        (attribution-ok :weight 3))
      (gates (:publish :release :push-truth)))

    (bundle MageOfReceipts
      (requires
        (artifact-hash-present :weight 5)
        (tests-linked :weight 4)
        (decision-log-written :weight 3))
      (gates (:release :push-truth)))

    (bundle KeeperOfReceipts
      (requires
        (witness-cited :weight 5)
        (claim-time-stamped :weight 3))
      (gates (:publish :push-truth)))

    (bundle AnchorClerk
      (requires
        (owner-declared :weight 5)
        (dod-present :weight 4)
        (options-present :weight 2))
      (gates (:merge :release :push-truth)))

    (bundle FileSentinel
      (requires
        (anchored-changes :weight 5)
        (receipts-for-artifacts :weight 5)
        (no-unsafe-path-writes :weight 4)
        (no-huge-blobs :weight 3))
      (gates (:publish :release :push-truth)))

    (bundle Lith
      (requires
        (balanced-parens :weight 5)
        (single-source-of-truth :weight 5)
        (binding-complete :weight 4)
        (receipt-glue-present :weight 4)
        (lambda-signed :weight 5))
      (gates (:push-truth))))

  (gate-target
    (id :push-truth)
    (en "Push Truth")
    (jp "真理へ押し込む")
    (meaning
      "Promote changes from ORIGIN -> TRUTH.
       ORIGIN = what I own locally.
       TRUTH = what the Fork Tax can audit (open source mandate + receipts)."))

  (realms
    (realm (id :origin) (en "Origin") (jp "起源") (sigil "η"))
    (realm (id :truth) (en "Truth") (jp "真理") (sigil "μ"))
    (bridge (id :fork-tax) (en "Fork Tax") (jp "フォーク税") (sigil "Π")))

  (truth-unit
    (id :pi-package+host)
    (en "Truth = Pi Package + Host")
    (jp "真理=Piパッケージ+ホスト")
    (requires
      (package
        (kind :zip)
        (artifact-root "artifacts/truth/")
        (naming "Pi.<sha256-prefix>.zip")
        (must-include ["LICENSE" "README.md" "receipts.log" "manifest.lith"])
        (hash-algo "sha256"))
      (host
        (kind :github-gist)
        (must-include ["receipts.log" "manifest.lith"])
        (returns [:gist-id :gist-url]))
      (linkage
        (manifest "manifest.lith")
        (must-link [:zip-sha256 :zip-path :gist-url :gist-id])
        (receipt-entry "receipts.log"))))

  (drift->song
    (rule (when (fails ForkTax oss-license-present))
      (emit (song fork-tax-canticle :warning
        (verse "No license, no blessing. The Canticle breaks.")
        (repair "Add LICENSE, cite it in README, commit."))))

    (rule (when (fails MageOfReceipts artifact-hash-present))
      (emit (song receipt-river :lament
        (verse "Artifacts drift downstream unnamed-no hash, no proof.")
        (repair "Generate sha256, record in Receipt River, commit."))))

    (rule (when (fails KeeperOfReceipts witness-cited))
      (emit (song witness-thread :audit
        (verse "Words without witnesses fray into fog.")
        (repair "Quote the source line, link the receipt, commit."))))

    (rule (when (fails AnchorClerk owner-declared))
      (emit (song anchor-registry :repair
        (verse "Unowned work is unanchored-storms take it.")
        (repair "Declare Owner/Deadline/DoD, then commit."))))

    (rule (when (blocked? :push-truth))
      (emit (song path-ward :ritual
        (hooks ("pay the fork tax" "anchor the change" "show the hash" "push truth"))
        (jp-hooks ("フォーク税を払え" "変更に錨を" "ハッシュを示せ" "真理へ押し込め"))
        (references (gates [:push-truth]) (drifts (active-drifts)))))))

  (policy
    (gate
      (release
        (requires [:oss-license-present :artifact-hash-present :tests-linked]))
      (publish
        (requires [:attribution-ok :decision-log-written]))
      (merge
        (requires [:owner-declared :dod-present :options-present]))
      (push-truth
        (requires
          [:oss-license-present :source-available :attribution-ok
           :artifact-hash-present :tests-linked :decision-log-written
           :anchored-changes :receipts-for-artifacts :no-unsafe-path-writes :no-huge-blobs
           :balanced-parens :single-source-of-truth :binding-complete :receipt-glue-present :lambda-signed])))
    (feedback
      (on-drift :sing)
      (on-repair :praise)
      (on-block :push-truth :sing)
      (on-pass :push-truth :praise))))
