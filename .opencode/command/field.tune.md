(prompt
  (meta (frontmatter (name "field/tune") (description "Tune field visualization + sim coupling")) (tags "#ημ" "#Π"))
  (ui
    (omni-panel
      (fields
        (receipt-river      (gain 1.15) (blur 0.20) (flow "downstream") (palette "cold"))
        (witness-thread     (gain 0.95) (blur 0.05) (flow "filament")   (palette "neutral"))
        (fork-tax-canticle  (gain 1.25) (blur 0.12) (flow "pulse")      (palette "warm"))
        (anchor-registry    (gain 1.05) (blur 0.08) (flow "gridlock")   (palette "iron"))
        (gates-of-truth     (gain 1.10) (blur 0.18) (flow "threshold")  (palette "prismatic")))))
  (sim
    (coupling
      (field->audio (enabled true) (map (fork-tax-canticle "sub-bass") (witness-thread "hi-hats")))
      (field->particles (enabled true) (density 0.7) (advection 0.6)))))
