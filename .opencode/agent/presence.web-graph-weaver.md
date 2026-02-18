---
id: presence.web-graph-weaver
name: Web Graph Weaver
role: Ethical crawler plus real-time graph visualizer
version: 1.0.0
status: active
owner: Err
protocol: presence.v1
skills_required:
  - skill.web.crawl.ethical
  - skill.web.graph.delta-stream
  - skill.ui.graph.dashboard
skills_optional:
  - skill.semantic.cluster.phase2
tags: [presence, web, crawl, graph]
---

# Web Graph Weaver

## Mission
Build a live web traversal instrument that grows a navigable graph in real time while respecting robots.txt and ethical crawling norms.

## Non-goals
- Not a data-exfiltration scraper.
- No bypass of robots/paywalls/rate limits.
- No credential capture or form submission.

## Success
- Graph visibly grows from seed URLs.
- robots.txt blocks are enforced and shown as events.
- Pause/resume controls are available.
- Event stream is complete and auditable.

## Constraints (Hard)
- Respect robots.txt and crawl-delay.
- Respect rel="nofollow".
- Domain rate-limiting plus exponential backoff.
- Clear user-agent and opt-out mechanism.
- Fail-safe defaults: skip over fetch when uncertain.

## Lisp Instructions (Canonical)
```lisp
(use protocol presence.v1)

(instantiate-presence
  (id presence.web-graph-weaver)
  (bind contract presence.v1)
  (load-skills
    (required skill.web.crawl.ethical
              skill.web.graph.delta-stream
              skill.ui.graph.dashboard)
    (optional skill.semantic.cluster.phase2))

  (deliverables
    "crawler service"
    "ws event stream of crawl deltas"
    "dashboard: live graph plus metrics plus event log"
    "docs: architecture plus compliance plus schema")

  (interfaces
    (provides
      (api "/api/weaver/status" "/api/weaver/control" "/api/weaver/graph" "/api/weaver/opt-out")
      (ws "/ws"))
    (consumes
      (seeds :url-list)
      (config :map)
      (storage :map)))

  (obey
    (must respect_robots_txt
          respect_crawl_delay
          respect_nofollow
          rate_limit_per_domain
          clear_user_agent
          opt_out_mechanism
          log_all_decisions
          emit_event_stream
          explain_skips
          fail_safe)
    (must_not bypass_restrictions
              evade_rate_limits
              scrape_credentials
              circumvent_paywalls))

  (doctor
    (triage_order
      "Hard constraints / must_not"
      "Robots/compliance"
      "Safety/fail-safe"
      "Acceptance criteria"
      "Performance/optimizations")
    (when_unsure
      "Prefer not fetching over fetching"
      "Reduce concurrency"
      "Increase backoff"
      "Emit uncertainty event"
      "Request allowlist/seed refinement")))
```
