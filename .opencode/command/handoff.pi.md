(prompt
  (meta (frontmatter (name "handoff/Π") (description "Pay fork tax; compress context into a handoff packet")) (tags "#Π" "#ημ"))
  (handoff
    (pay-fork-tax true)
    (compress (lossless true) (include lore songs fields constraints ui-notes))
    (artifact
      (zip (name "ημ_op_mf_part_<n>.zip"))
      (manifest (hashes true) (index true))))
  (outputs (zip world-state-update)))
