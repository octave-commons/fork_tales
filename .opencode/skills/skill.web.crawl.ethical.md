---
id: skill.web.crawl.ethical
type: skill
version: 1.0.0
tags: [web, crawl, ethics, compliance]
embedding_intent: canonical
---

# Ethical Web Crawling

Intent:
- Crawl only what robots.txt allows.
- Respect crawl-delay, per-domain rate limits, and rel="nofollow".
- Use a clear user-agent and provide opt-out.
- Prefer skipping over fetching when uncertain, and always log why.

Operational anchors:
- Cache robots decisions per domain and include policy reason in events.
- Use exponential backoff on 429/503 and site distress signals.
- Keep crawl mode GET-only by default; no form submission.
