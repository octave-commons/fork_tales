(intent instantiate.presence
  (presence presence.web-graph-weaver)
  (from_agent_file ".opencode/agent/presence.web-graph-weaver.md")
  (embed_skills
    skill.web.crawl.ethical
    skill.web.graph.delta-stream
    skill.ui.graph.dashboard)
  (seed_urls "https://example.org/")
  (config
    (max_depth 3)
    (max_concurrency 8)
    (per_domain_rps 0.5)))
