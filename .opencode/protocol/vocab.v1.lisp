(vocab
  (v "ημ/VOCAB.v1")
  (modes η μ Π A)
  (ctx 主 己 汝 彼 世)

  (glyphs
    (主 :ctx 0 "Presence/Silent Core; watcher-of-watchers")
    (己 :ctx 0 "self")
    (汝 :ctx 0 "you")
    (彼 :ctx 0 "them")
    (世 :ctx 0 "world")
    (址 :fn 1 "locate: entity -> weighted regions")
    (熱 :scalar 0 "region intensity for renderer")
    (覆 :rel 2 "coverage/mapping relations")
    (破 :tag 0 "failure/break tag")
    (息 :tok 0 "ritual token; semantics TBD")
    (心 :tok 0 "named concept; semantics TBD")
    (家 :tok 0 "home; semantics TBD")
    (契 :contract 1 "contract glyph")
    (真 :draft 0 "truth glyph")
    (霊 :draft 0 "daimoi glyph")
    (観 :draft 0 "observation glyph")
    (嵌 :draft 0 "embedding glyph"))

  (signature
    (契
      (arity 1)
      (input :contract/v1)
      (output :contract/v1)
      (ingest-rule :append-only)
      (requires [:id :owner :dod :refs])
      (proof-path [:refs :about/fact->entity :址]))))
