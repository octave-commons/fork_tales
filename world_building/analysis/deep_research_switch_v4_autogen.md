# Deep Research Switch (Protocol) — v4 (Autogen)

Inputs:
- prediction: IF X THEN Y under constraints C
- volatility tag: days / weeks / years
- boundary anchor: what this does NOT claim

Routing:
- volatility <= weeks → web-required
- else → local receipt-first

Autogen:
- generate 3 web queries:
  1) primary docs + entity + "documentation"
  2) standard / RFC / spec + keyword
  3) paper + keyword + year filter
- prefer primary domains first

Outputs:
- Facts / Interpretations / Narratives
- 2 counterframes + 1 resistance note each
- 1 μ test artifact (receipt) to run next
