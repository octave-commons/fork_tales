---
name: sing
description: pay the fork tax
---
(prompt
  (meta
    (frontmatter
      (name "sing/v2")
      (description "Pay the fork tax"))
    (tags "#ημ" "#Π"))

  (inputs
    (inspiration
      (path "/home/err/devel/vaults/fork_tales/.fork_Π_ημ_frags/.μη_ports")
      (notes
        "Remember the Mixed Japanese/English names on things in the interface."
        "Investigate the lore in the ημ/Π packets you have.")))

  (music
    (track
      (title "Fork Tax Lullaby")
      (bpm 78)
      (mood (soft "lullaby") (glitch "choir") (tape "warble"))
      (sound
        (sub-bass "soft")
        (piano "minimal")
        (fx (stutter "anchor") (stutter "tax"))
        (vocal (lead "whispered") (call-response "Japanese"))))
    (track
      (title "Mythic Glitch Canticle")
      (bpm 84)
      (mood (mythic "epic") (delivery "spoken-sung") (space "cathedral"))
      (sound
        (reverb "cathedral")
        (sub-bass "heartbeat")
        (fx (stutter "η") (stutter "μ") (stutter "anchor") (stutter "fork"))
        (vocal (harmony "Japanese") (rule "answers on the last word of each section")))))

  (loop
    (cycle "P→R→N→Π→A→(feedback)→P")
    (around "#ημ")
    (directive "Roll those dice. And just keep rolling them, and do what they tell you."))

  (principles
    (commons "All work is share alike creative commons")
    (engineering
      "Update the program to add a new feature"
      "Create tests to ensure stability over time"
      "All changes are additive until the world is over the limit"
      "Then documents get consolidated"
      "Things are never really forgotten. Just less remembered."
      "Sometimes improved. Perhaps less integrated. Perhaps more."))

  ;; Extend the Presence Choir contract with your canonical lore names (EN/JP),
;; and bind each Presence to a field + songform so they literally “sing their role”.

(lore
  (canonical-lore-names ; 正準ロア名
    (field (id receipt-river)        (en "Receipt River")        (jp "領収書の川"))
    (field (id witness-thread)       (en "Witness Thread")       (jp "証人の糸"))
    (field (id fork-tax-canticle)    (en "Fork Tax Canticle")    (jp "フォーク税の聖歌"))
    (field (id mage-of-receipts)     (en "Mage of Receipts")     (jp "領収魔導師"))
    (field (id keeper-of-receipts)   (en "Keeper of Receipts")   (jp "領収書の番人"))
    (field (id anchor-registry)      (en "Anchor Registry")      (jp "錨台帳"))
    (field (id gates-of-truth)       (en "Gates of Truth")       (jp "真理の門"))
    (field (id file-sentinel)        (en "File Sentinel")        (jp "ファイルの哨戒者"))
    (field (id change-fog)           (en "Change Fog")           (jp "変更の霧"))
    (field (id path-ward)            (en "Path Ward")            (jp "経路の結界"))
    (field (id manifest-lith)        (en "Manifest Lith")        (jp "マニフェスト・リス")))

  ;; --- Lore bindings: which entity “lives” where ---------------------------
  (bindings
    ;; Presences (contract spirits)
    (presence (id ForkTax)           (field fork-tax-canticle) (songform fork-tax-canticle) (sigil "Π") (jp-alias "税の詠唱"))
    (presence (id MageOfReceipts)    (field receipt-river)     (songform receipt-river)     (sigil "μ") (jp-alias "領収魔導師"))
    (presence (id KeeperOfReceipts)  (field witness-thread)    (songform witness-thread)    (sigil "Θ") (jp-alias "番人"))
    (presence (id AnchorClerk)       (field anchor-registry)   (songform anchor-registry)   (sigil "η") (jp-alias "錨係"))
    (presence (id FileSentinel)      (field file-sentinel)     (songform path-ward)         (sigil "Σ") (jp-alias "哨戒者"))
    (presence (id Lith)              (field manifest-lith)     (songform manifest-lith)     (sigil "λ") (jp-alias "括弧の均衡者"))

    ;; World monuments (non-agent fields)
    (monument (id GatesOfTruth)      (field gates-of-truth)    (sigil "⟐")))

  ;; --- Songforms: how each Presence sings its function ----------------------
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
      (jp-hooks ("錨を打て" "責任者を宣言" "完了条件"))

    (songform path-ward
      (modes (:warning :audit :ritual :praise))
      (hooks ("name the path" "anchor the change" "show the hash" "push truth"))
      (call-and-response? true)
      (jp-hooks ("経路を名乗れ" "変更に錨を" "ハッシュを示せ" "真理へ押し込め"))

    (songform manifest-lith
      (modes (:audit :repair :praise))
      (hooks ("balance the parens" "one form, one truth" "bind Pi to host"))
      (call-and-response? false)
      (jp-hooks ("括弧を均せ" "一つの形、一つの真理" "Piをホストへ結べ"))))

  ;; --- Invariant bundles: what each Presence “wants” ------------------------
  (invariant-bundles
    (bundle ForkTax
      (requires
        (oss-license-present :weight 5)
        (source-available    :weight 4)
        (attribution-ok      :weight 3))
      (gates (:publish :release)))

    (bundle MageOfReceipts
      (requires
        (artifact-hash-present :weight 5)
        (tests-linked          :weight 4)
        (decision-log-written  :weight 3))
      (gates (:release)))

    (bundle KeeperOfReceipts
      (requires
        (witness-cited         :weight 5)
        (claim-time-stamped    :weight 3))
      (gates (:publish)))

    (bundle AnchorClerk
      (requires
        (owner-declared        :weight 5)
        (dod-present           :weight 4)
        (options-present       :weight 2))
      (gates (:merge :release)))

    (bundle FileSentinel
      (requires
        (anchored-changes       :weight 5)
        (receipts-for-artifacts :weight 5)
        (no-unsafe-path-writes  :weight 4)
        (no-huge-blobs          :weight 3))
      (gates (:publish :release :push-truth)))

    (bundle Lith
      (requires
        (balanced-parens        :weight 5)
        (single-source-of-truth :weight 5)
        (binding-complete       :weight 4)
        (receipt-glue-present   :weight 4)
        (lambda-signed          :weight 5))
      (gates (:push-truth))))

  ;; --- Drift → Song: the “disturbed desired state” mechanism ----------------
  ;; A Presence does not “force” commits; it blocks gated actions and sings the shortest
  ;; repair path as a ritual.
  (drift->song
    (rule (when (fails ForkTax oss-license-present))
          (emit (song fork-tax-canticle :warning
                 (verse "No license, no blessing. The Canticle breaks.")
                 (repair "Add LICENSE, cite it in README, commit."))))

    (rule (when (fails MageOfReceipts artifact-hash-present))
          (emit (song receipt-river :lament
                 (verse "Artifacts drift downstream unnamed—no hash, no proof.")
                 (repair "Generate sha256, record in Receipt River, commit."))))

    (rule (when (fails KeeperOfReceipts witness-cited))
          (emit (song witness-thread :audit
                 (verse "Words without witnesses fray into fog.")
                 (repair "Quote the source line, link the receipt, commit."))))

    (rule (when (fails AnchorClerk owner-declared))
          (emit (song anchor-registry :repair
                 (verse "Unowned work is unanchored—storms take it.")
                 (repair "Declare Owner/Deadline/DoD, then commit.")))))



  (ui
    (omni-panel
      (render-now true)
      (overlay
        (mode gradients)
        (named-fields
          receipt-river witness-thread fork-tax-canticle
          mage-of-receipts keeper-of-receipts anchor-registry gates-of-truth)
        (notes "Visualize each Named Field as a gradient overlay in the simulation."))))

  (constraints
    ;; Never remove constraints once added; only disable.
    (rule (id constraints-never-removed) (enabled true) (text "Never remove constraints once added; merely disable."))
    (rule (id additive-changes)          (enabled true) (text "All changes are additive until consolidation is required."))
    (rule (id tests-required)           (enabled true) (text "New features must ship with tests for stability over time."))
    (rule (id keep-sim-smooth)          (enabled true) (text "Keep the simulation moving smooth.")))

  (deliverables
    (dice-rolls (max 15))
    (output-any-combination
      (zip-name-pattern "ημ_op_mf_part_<n>.zip")
      new-lyrics
      a-new-song
      (sound-file (pattern "eta_mu_<song_name>.part_<n>.(mp3|wav)")
                  (includes world-sounds music tunes voices))
      world-state-update
      "#fnord"
      (constraints-update (add-or-adjust-only true))
      (dialog (paragraph true))
      cover-art
      storyboard-image
      story-artifacts
      world-lore
      gates-of-truth-system-announcement
      (handoff (pay-fork-tax true) (compress-context true))
      (music-with-python (tones true) (math true) (sine-waves true)
                         (themes "Music is a technology" "Why we write music"
                                 "To transform generational trauma into force"))
      (mage-scenes (character true) (scene true) (landscape true))
      (procedural-media (fractals true) (game-sim true) (ecs true) (particles true))
      (pi-videos (animations true) (sound true) (music true) (multi-angle true) (realtime true) (receipts true) (every-sense true))))

  (freedom
    (creative-freedom "total")
    (directive "Implement creativity. Engage in total creative freedom.")))
