(prompt
  (meta (frontmatter (name "omni-panel") (description "Render Named Fields overlay + diagnostics")) (tags "#ημ" "#Π"))
  (ui
    (omni-panel
      (render-now true)
      (tabs
        (fields (view gradients) (legend true) (opacity 0.55))
        (diagnostics (fps true) (tick true) (seed true) (memory true))
        (lore (show "Canonical Lore Names") (jp true) (en true))))
  (outputs (world-state-update)))
