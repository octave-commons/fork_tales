(prompt
  (meta (frontmatter (name "roll") (description "Roll dice and choose deliverables")) (tags "#ημ"))
  (loop (dice (max 15)) (policy "any-combination"))
  (selection
    (prefer (new-lyrics a-new-song world-state-update "#fnord"))
    (avoid (remove-constraints)))
