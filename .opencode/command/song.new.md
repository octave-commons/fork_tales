(prompt
  (meta (frontmatter (name "song/new") (description "Generate a new song seed from lore + fields")) (tags "#ημ" "#Π"))
  (music
    (seed
      (bpm (choose 78 84 92 104))
      (style (glitch-choir lullaby) (mythic spoken-sung) (cathedral-reverb))
      (stutters "η" "μ" "anchor" "fork" "tax")
      (jp-en (ratio 0.35) (call-response true))
      (motifs (heartbeat-sub true) (tape-warble true) (minimal-piano true))))
  (outputs (a-new-song new-lyrics (sound-file (pattern "eta_mu_<song_name>.part_<n>.wav")))))
