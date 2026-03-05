const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");

const {
  extractArxivIdFromUrl,
  isArxivSearchUrl,
  parseArxivSearchSeed,
  buildArxivApiQueryUrl,
  extractArxivAbsUrlsFromApiFeed,
  extractFeedEntries,
  extractFeedEntryLinks,
  looksLikeFeedDocument,
  canonicalArxivAbsUrlFromId,
  canonicalArxivPdfUrlFromId,
  canonicalWikipediaArticleUrl,
  extractSemanticReferences,
  classifyKnowledgeUrl,
  normalizeAnalysisSummary,
  parseAuthHeader,
  llmAuthHeaders,
  loadWorldWatchlistSeeds,
  parseWorldWatchlistSeeds,
  GraphStore,
  WebGraphWeaver,
} = require("../web_graph_weaver.js");

function findReference(rows, edgeKind, targetUrl) {
  return rows.find(
    (row) => row.edge_kind === edgeKind && row.url === targetUrl,
  );
}

test("extractArxivIdFromUrl supports canonical and legacy IDs", () => {
  assert.equal(
    extractArxivIdFromUrl("https://arxiv.org/abs/2401.12345v2"),
    "2401.12345",
  );
  assert.equal(
    extractArxivIdFromUrl("https://arxiv.org/pdf/2401.12345v2.pdf"),
    "2401.12345",
  );
  assert.equal(
    extractArxivIdFromUrl("https://arxiv.org/abs/cs/9901001v1"),
    "cs/9901001",
  );
  assert.equal(
    extractArxivIdFromUrl("https://example.com/abs/2401.12345"),
    "",
  );
});

test("arXiv search seed helpers normalize query into API query", () => {
  const searchUrl = "https://arxiv.org/search/?query=graph+neural+network&searchtype=all&size=12&order=-announced_date_first";
  assert.equal(isArxivSearchUrl(searchUrl), true);
  const seed = parseArxivSearchSeed(searchUrl);
  assert.ok(seed);
  assert.equal(seed.searchQuery, "all:graph neural network");
  assert.equal(seed.maxResults, 12);
  assert.equal(seed.sortBy, "submittedDate");

  const apiUrl = buildArxivApiQueryUrl(seed);
  assert.ok(apiUrl.includes("export.arxiv.org/api/query"));
  assert.ok(apiUrl.includes("search_query=all%3Agraph+neural+network"));
});

test("extractArxivAbsUrlsFromApiFeed parses canonical arXiv abs URLs", () => {
  const feed = `
    <feed>
      <entry>
        <id>http://arxiv.org/abs/2401.12345v2</id>
      </entry>
      <entry>
        <id>http://arxiv.org/abs/cs/9901001v1</id>
      </entry>
    </feed>
  `;
  const rows = extractArxivAbsUrlsFromApiFeed(feed, 10);
  assert.deepEqual(rows, [
    "https://arxiv.org/abs/2401.12345",
    "https://arxiv.org/abs/cs/9901001",
  ]);
});

test("extractFeedEntryLinks parses RSS, Atom, and jsonfeed entry URLs", () => {
  const rss = `
    <rss version="2.0">
      <channel>
        <item><link>https://news.ycombinator.com/item?id=1</link></item>
        <item><link>https://example.org/advisory</link></item>
      </channel>
    </rss>
  `;
  assert.deepEqual(extractFeedEntryLinks(rss, "https://hnrss.org/frontpage", 10), [
    "https://news.ycombinator.com/item?id=1",
    "https://example.org/advisory",
  ]);

  const atom = `
    <feed xmlns="http://www.w3.org/2005/Atom">
      <title>sample</title>
      <entry>
        <link rel="self" href="https://hnrss.org/frontpage" />
        <link rel="alternate" href="https://news.ycombinator.com/item?id=2" />
      </entry>
      <entry>
        <id>https://example.net/post/7</id>
      </entry>
    </feed>
  `;
  assert.deepEqual(extractFeedEntryLinks(atom, "https://hnrss.org/frontpage", 10), [
    "https://news.ycombinator.com/item?id=2",
    "https://example.net/post/7",
  ]);

  const jsonFeed = JSON.stringify({
    version: "https://jsonfeed.org/version/1.1",
    items: [
      { id: "1", url: "https://example.com/a" },
      { id: "2", external_url: "https://example.com/b" },
    ],
  });
  assert.deepEqual(extractFeedEntryLinks(jsonFeed, "https://example.com/feed", 10), [
    "https://example.com/a",
    "https://example.com/b",
  ]);
});

test("extractFeedEntries preserves feed metadata for RSS and JSON feed", () => {
  const rss = `
    <rss version="2.0">
      <channel>
        <item>
          <title>Transit chokepoint warning</title>
          <link>https://example.net/report/42</link>
          <description>Shipping lane disruption risk elevated.</description>
          <pubDate>Tue, 04 Mar 2026 06:12:00 GMT</pubDate>
        </item>
      </channel>
    </rss>
  `;
  const rssRows = extractFeedEntries(rss, "https://example.net/feed", 10);
  assert.equal(rssRows.length, 1);
  assert.equal(rssRows[0].url, "https://example.net/report/42");
  assert.equal(rssRows[0].title, "Transit chokepoint warning");
  assert.ok(String(rssRows[0].summary || "").includes("Shipping lane disruption"));
  assert.equal(rssRows[0].sourceKind, "feed:rss");

  const jsonFeed = JSON.stringify({
    version: "https://jsonfeed.org/version/1.1",
    items: [
      {
        id: "item-1",
        url: "https://alerts.example.org/post/7",
        title: "Port outage",
        content_text: "Port telemetry unavailable for 2 hours.",
        date_published: "2026-03-04T08:00:00Z",
      },
    ],
  });
  const jsonRows = extractFeedEntries(jsonFeed, "https://alerts.example.org/feed", 10);
  assert.equal(jsonRows.length, 1);
  assert.equal(jsonRows[0].url, "https://alerts.example.org/post/7");
  assert.equal(jsonRows[0].title, "Port outage");
  assert.equal(jsonRows[0].sourceKind, "feed:json");
  assert.ok(String(jsonRows[0].summary || "").includes("telemetry unavailable"));
});

test("looksLikeFeedDocument detects RSS and jsonfeed payloads", () => {
  assert.equal(
    looksLikeFeedDocument("application/rss+xml", "<rss><channel><item /></channel></rss>"),
    true,
  );
  assert.equal(
    looksLikeFeedDocument(
      "application/feed+json",
      '{"version":"https://jsonfeed.org/version/1.1","items":[]}',
    ),
    true,
  );
  assert.equal(
    looksLikeFeedDocument("text/html", "<html><body>hi</body></html>"),
    false,
  );
});

test("canonical URL helpers normalize arXiv and Wikipedia resources", () => {
  assert.equal(
    canonicalArxivAbsUrlFromId("2401.12345v2"),
    "https://arxiv.org/abs/2401.12345",
  );
  assert.equal(
    canonicalArxivPdfUrlFromId("2401.12345v2"),
    "https://arxiv.org/pdf/2401.12345.pdf",
  );
  assert.equal(
    canonicalWikipediaArticleUrl("https://en.m.wikipedia.org/wiki/Graph_theory#history"),
    "https://en.wikipedia.org/wiki/Graph_theory",
  );
  assert.equal(
    canonicalWikipediaArticleUrl("https://en.wikipedia.org/wiki/File:Graph.svg"),
    "",
  );
});

test("extractSemanticReferences maps arXiv citations, PDF edges, and wiki cross references", () => {
  const html = `
    <html>
      <body>
        <a href="/pdf/2401.12345v2.pdf">PDF</a>
        <a href="https://arxiv.org/abs/2402.54321">Reference</a>
        <a href="https://en.wikipedia.org/wiki/Graph_neural_network">Wikipedia</a>
        <p>Related work includes arXiv:2403.11111v3 and arXiv:2401.12345.</p>
      </body>
    </html>
  `;
  const payload = extractSemanticReferences(
    "https://arxiv.org/abs/2401.12345v2",
    html,
  );

  assert.equal(payload.source_kind, "arxiv_abs");
  assert.ok(
    findReference(
      payload.references,
      "paper_pdf",
      "https://arxiv.org/pdf/2401.12345.pdf",
    ),
  );
  assert.ok(
    findReference(
      payload.references,
      "citation",
      "https://arxiv.org/abs/2402.54321",
    ),
  );
  assert.ok(
    findReference(
      payload.references,
      "citation",
      "https://arxiv.org/abs/2403.11111",
    ),
  );
  assert.ok(
    findReference(
      payload.references,
      "cross_reference",
      "https://en.wikipedia.org/wiki/Graph_neural_network",
    ),
  );
  assert.equal(
    findReference(
      payload.references,
      "citation",
      "https://arxiv.org/abs/2401.12345",
    ),
    undefined,
  );
});

test("extractSemanticReferences maps Wikipedia internal links and arXiv cross references", () => {
  const html = `
    <html>
      <body>
        <a href="/wiki/Graph_theory">Graph theory</a>
        <a href="https://arxiv.org/abs/2402.54321">arXiv link</a>
        <a href="/wiki/File:Graph.svg">ignore file page</a>
        <a rel="nofollow" href="https://arxiv.org/abs/2404.00001">nofollow arXiv</a>
      </body>
    </html>
  `;
  const payload = extractSemanticReferences(
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    html,
  );

  assert.equal(payload.source_kind, "wikipedia_article");
  assert.ok(
    findReference(
      payload.references,
      "wiki_reference",
      "https://en.wikipedia.org/wiki/Graph_theory",
    ),
  );
  assert.ok(
    findReference(
      payload.references,
      "cross_reference",
      "https://arxiv.org/abs/2402.54321",
    ),
  );

  const nofollowCrossRef = findReference(
    payload.references,
    "cross_reference",
    "https://arxiv.org/abs/2404.00001",
  );
  assert.ok(nofollowCrossRef);
  assert.equal(nofollowCrossRef.nofollow, true);
  assert.equal(nofollowCrossRef.enqueue, false);
});

test("classifyKnowledgeUrl recognizes arXiv and Wikipedia pages", () => {
  assert.equal(classifyKnowledgeUrl("https://arxiv.org/abs/2401.12345"), "arxiv_abs");
  assert.equal(classifyKnowledgeUrl("https://arxiv.org/pdf/2401.12345.pdf"), "arxiv_pdf");
  assert.equal(
    classifyKnowledgeUrl("https://en.wikipedia.org/wiki/Graph_theory"),
    "wikipedia_article",
  );
  assert.equal(classifyKnowledgeUrl("https://example.com"), "other");
});

test("normalizeAnalysisSummary removes prompt echo and placeholders", () => {
  const noisy =
    "**Summary:** the page text in 2 concise bullets. FocusIntent: <what should a graph crawler learn from this page>.";
  const normalized = normalizeAnalysisSummary(
    noisy,
    "CISA advisory references a known exploited vulnerability and mitigation steps.",
  );

  assert.match(normalized, /^- .+\n- .+\nFocusIntent: .+/s);
  assert.equal(
    normalized.toLowerCase().includes("the page text in 2 concise bullets"),
    false,
  );
  assert.equal(
    normalized.includes("<what should a graph crawler learn from this page>"),
    false,
  );
});

test("llm auth header helpers normalize auth env shapes", () => {
  assert.deepEqual(parseAuthHeader("Authorization: Bearer abc"), {
    Authorization: "Bearer abc",
  });
  assert.deepEqual(parseAuthHeader("Bearer xyz"), {
    Authorization: "Bearer xyz",
  });

  const keys = [
    "WEAVER_LLM_AUTH_HEADER",
    "WEAVER_LLM_BEARER_TOKEN",
    "WEAVER_LLM_API_KEY",
    "WEAVER_LLM_API_KEY_HEADER",
    "TEXT_GENERATION_AUTH_HEADER",
    "TEXT_GENERATION_BEARER_TOKEN",
    "TEXT_GENERATION_API_KEY",
    "TEXT_GENERATION_API_KEY_HEADER",
  ];
  const restore = {};
  for (const key of keys) {
    restore[key] = process.env[key];
    delete process.env[key];
  }

  try {
    process.env.TEXT_GENERATION_API_KEY = "token-text";
    assert.deepEqual(llmAuthHeaders(), { "X-API-Key": "token-text" });

    process.env.TEXT_GENERATION_API_KEY_HEADER = "Authorization";
    assert.deepEqual(llmAuthHeaders(), { Authorization: "token-text" });

    process.env.WEAVER_LLM_BEARER_TOKEN = "bearer-local";
    assert.deepEqual(llmAuthHeaders(), {
      Authorization: "Bearer bearer-local",
    });

    process.env.WEAVER_LLM_AUTH_HEADER = "X-Token: weaver-direct";
    assert.deepEqual(llmAuthHeaders(), { "X-Token": "weaver-direct" });
  } finally {
    for (const key of keys) {
      if (restore[key] === undefined) {
        delete process.env[key];
      } else {
        process.env[key] = restore[key];
      }
    }
  }
});

test("world watchlist parser returns normalized maritime seed rows", () => {
  const rows = parseWorldWatchlistSeeds({
    enabled: true,
    domains: [
      {
        id: "hormuz",
        seed_urls: [
          {
            url: "https://www.ukmto.org/advisory/003-26",
            kind: "maritime:ukmto_advisory",
          },
          {
            url: "https://www.maritime.dot.gov/msci/2026-001-persian-gulf-strait-hormuz-and-gulf-oman-iranian-illegal-boarding-detention-seizure",
            kind: "maritime:marad_advisory",
          },
        ],
      },
    ],
  });
  assert.equal(Array.isArray(rows), true);
  assert.equal(rows.length, 2);
  assert.equal(rows[0].kind.startsWith("maritime:"), true);
  assert.equal(rows[1].kind.startsWith("maritime:"), true);
});

test("world watchlist parser preserves feed seed metadata", () => {
  const rows = parseWorldWatchlistSeeds({
    enabled: true,
    domains: [
      {
        id: "hacker_news",
        seed_urls: [
          {
            url: "https://hnrss.org/frontpage",
            kind: "feed:rss",
            title: "Hacker News Frontpage RSS",
            source_type: "rss",
          },
        ],
      },
    ],
  });
  assert.equal(rows.length, 1);
  assert.equal(rows[0].url, "https://hnrss.org/frontpage");
  assert.equal(rows[0].kind, "feed:rss");
  assert.equal(rows[0].source_type, "rss");
  assert.equal(rows[0].domain_id, "hacker_news");
});

test("weaver start auto-loads world watchlist seeds", () => {
  const rows = loadWorldWatchlistSeeds();
  assert.equal(Array.isArray(rows), true);
  assert.equal(rows.length >= 1, true);

  const weaver = new WebGraphWeaver();
  try {
    const outcome = weaver.start({ seeds: [] });
    assert.equal(outcome.ok, true);
    assert.equal(Array.isArray(outcome.world_watch_seeds), true);
    assert.equal(outcome.world_watch_seeds.length >= 1, true);
  } finally {
    weaver.shutdown();
  }
});

test("WebGraphWeaver bootstrap seed selection skips local and opt-out hosts", () => {
  const weaver = new WebGraphWeaver();
  try {
    weaver.graph.upsertUrl("https://example.com/security/advisory", 0, null);
    weaver.graph.setUrlStatus("https://example.com/security/advisory", {
      status: "discovered",
      activation_potential: 1.25,
    });
    weaver.graph.upsertUrl("http://127.0.0.1:8899/seed", 0, null);
    weaver.graph.setUrlStatus("http://127.0.0.1:8899/seed", {
      status: "discovered",
      activation_potential: 4,
    });
    weaver.graph.upsertUrl("https://localhost/internal", 0, null);
    weaver.graph.upsertUrl("https://blocked.example/private", 0, null);
    weaver.optOutDomains.add("blocked.example");

    const seeds = weaver._collectBootstrapGraphSeeds(16);
    assert.equal(
      seeds.includes("https://example.com/security/advisory"),
      true,
    );
    assert.equal(
      seeds.some((row) => String(row).includes("127.0.0.1")),
      false,
    );
    assert.equal(
      seeds.some((row) => String(row).includes("localhost")),
      false,
    );
    assert.equal(
      seeds.some((row) => String(row).includes("blocked.example")),
      false,
    );
  } finally {
    weaver.shutdown();
  }
});

test("WebGraphWeaver autostart bootstrap runs without manual control action", () => {
  const weaver = new WebGraphWeaver();
  try {
    const graphSeed = "https://example.com/autostart";
    weaver.graph.upsertUrl(graphSeed, 0, null);
    weaver.graph.setUrlStatus(graphSeed, {
      status: "discovered",
      activation_potential: 1.0,
    });
    weaver.autoStartCrawl = true;
    weaver.autoStartGraphSeedLimit = 8;

    const outcome = weaver._autostartCrawlOnBoot();
    assert.equal(outcome.ok, true);
    assert.equal(weaver.running, true);
    assert.equal(Array.isArray(outcome.requested_seeds), true);
    assert.equal(outcome.requested_seeds.includes(graphSeed), true);
    assert.equal(
      weaver.recentEvents.some(
        (row) =>
          row
          && row.event === "crawl_state"
          && row.state === "running"
          && row.start_reason === "bootstrap_autostart",
      ),
      true,
    );
  } finally {
    weaver.shutdown();
  }
});

test("WebGraphWeaver autostart raises low restored runtime limits to defaults", () => {
  const weaver = new WebGraphWeaver();
  try {
    const graphSeed = "https://example.com/autostart-limits";
    weaver.graph.upsertUrl(graphSeed, 0, null);
    weaver.graph.setUrlStatus(graphSeed, {
      status: "discovered",
      activation_potential: 1.0,
    });

    weaver.currentMaxDepth = 1;
    weaver.currentMaxNodes = 100;
    weaver.currentConcurrency = 1;
    weaver.currentMaxRequestsPerHost = 1;
    weaver.entityCount = 1;
    weaver._bootstrapEntities();

    weaver.autoStartCrawl = true;
    const outcome = weaver._autostartCrawlOnBoot();
    assert.equal(outcome.ok, true);
    assert.equal(weaver.currentMaxDepth >= 12, true);
    assert.equal(weaver.currentConcurrency >= 32, true);
    assert.equal(weaver.currentMaxRequestsPerHost >= 16, true);
    assert.equal(weaver.currentMaxNodes >= 2_000_000, true);
    assert.equal(weaver.entities.length >= 32, true);
  } finally {
    weaver.shutdown();
  }
});

test("WebGraphWeaver accepts high runtime limits beyond legacy caps", () => {
  const weaver = new WebGraphWeaver();
  try {
    const outcome = weaver.start({
      seeds: ["https://example.com/high-limits"],
      maxDepth: 48,
      maxNodes: 1_500_000,
      concurrency: 128,
      maxPerHost: 80,
      entityCount: 128,
      startReason: "test_high_limits",
    });
    assert.equal(outcome.ok, true);
    assert.equal(weaver.currentMaxDepth, 48);
    assert.equal(weaver.currentConcurrency, 128);
    assert.equal(weaver.currentMaxRequestsPerHost, 80);
    assert.equal(weaver.entities.length, 128);
    assert.equal(weaver.currentMaxNodes >= 1_500_000, true);
    assert.equal(
      weaver.recentEvents.some(
        (row) =>
          row
          && row.event === "crawl_state"
          && row.state === "running"
          && row.start_reason === "test_high_limits",
      ),
      true,
    );
  } finally {
    weaver.shutdown();
  }
});

test("registerInteraction enqueues URL after activation threshold", () => {
  const weaver = new WebGraphWeaver();
  try {
    const url = "https://example.com/research";
    const first = weaver.registerInteraction({
      url,
      delta: 0.3,
      source: "test",
    });
    assert.equal(first.ok, true);
    assert.equal(first.interaction.enqueued, false);

    const second = weaver.registerInteraction({
      url,
      delta: 0.9,
      source: "test",
    });
    assert.equal(second.ok, true);
    assert.equal(second.interaction.enqueued, true);
    assert.equal(weaver.frontier.has(url), true);
  } finally {
    weaver.shutdown();
  }
});

test("enqueueUrl enforces node cooldown window", () => {
  const weaver = new WebGraphWeaver();
  try {
    const url = "https://example.com/cooldown";
    weaver.graph.upsertUrl(url, 0, null);
    weaver.graph.setUrlStatus(url, {
      cooldown_until: Date.now() + 10 * 60 * 1000,
    });
    const outcome = weaver.enqueueUrl(url, null, 0, "manual");
    assert.equal(outcome.ok, false);
    assert.equal(outcome.reason, "cooldown_active");
    assert.ok(Number(outcome.retry_in_ms) > 0);
  } finally {
    weaver.shutdown();
  }
});

test("_processItem respects per-host concurrency cap", async () => {
  const weaver = new WebGraphWeaver();
  try {
    const url = "https://example.com/host-cap";
    weaver.currentMaxRequestsPerHost = 1;
    const hostState = weaver._domainState("example.com");
    hostState.active = 1;
    await weaver._processItem({
      url,
      sourceUrl: null,
      depth: 0,
      readyAt: Date.now(),
      priority: 1,
    });
    assert.equal(weaver.stats.host_concurrency_waits >= 1, true);
    assert.equal(weaver.frontier.has(url), true);
  } finally {
    weaver.shutdown();
  }
});

test("_processItem handles arXiv search via API without crawling /search HTML", async () => {
  const weaver = new WebGraphWeaver();
  const originalFetch = global.fetch;
  try {
    global.fetch = async (url) => {
      const target = String(url);
      assert.ok(target.includes("export.arxiv.org/api/query"));
      return {
        ok: true,
        status: 200,
        text: async () => `
          <feed>
            <entry><id>http://arxiv.org/abs/2401.12345v2</id></entry>
            <entry><id>http://arxiv.org/abs/2402.54321v1</id></entry>
          </feed>
        `,
      };
    };

    await weaver._processItem({
      url: "https://arxiv.org/search/?query=graph+learning&searchtype=all&size=5",
      sourceUrl: null,
      depth: 0,
      readyAt: Date.now(),
      priority: 1,
    });

    assert.equal(weaver.frontier.has("https://arxiv.org/abs/2401.12345"), true);
    assert.equal(weaver.frontier.has("https://arxiv.org/abs/2402.54321"), true);
  } finally {
    global.fetch = originalFetch;
    weaver.shutdown();
  }
});

test("_processItem expands RSS feed links into crawler frontier", async () => {
  const weaver = new WebGraphWeaver();
  const originalFetch = global.fetch;
  try {
    weaver._policyFor = async () => ({
      allow: [],
      disallow: [],
      crawlDelayMs: null,
    });

    global.fetch = async (url) => {
      const target = String(url);
      assert.equal(target, "https://hnrss.org/frontpage");
      return {
        ok: true,
        status: 200,
        url: target,
        headers: {
          get: (name) => {
            if (String(name).toLowerCase() === "content-type") {
              return "application/rss+xml; charset=utf-8";
            }
            return "";
          },
        },
        text: async () => `
          <rss version="2.0">
            <channel>
              <item>
                <title>HN entry</title>
                <link>https://news.ycombinator.com/item?id=123</link>
              </item>
              <item>
                <title>Security advisory</title>
                <description>Critical advisory for exposed edge gateway.</description>
                <link>https://example.org/security/advisory</link>
              </item>
            </channel>
          </rss>
        `,
        arrayBuffer: async () => Buffer.from(""),
      };
    };

    await weaver._processItem({
      url: "https://hnrss.org/frontpage",
      sourceUrl: null,
      depth: 0,
      readyAt: Date.now(),
      priority: 1,
    });

    assert.equal(weaver.frontier.has("https://news.ycombinator.com/item?id=123"), true);
    assert.equal(weaver.frontier.has("https://example.org/security/advisory"), true);
    const edges = weaver.graph.getOutgoingUrlEdges("https://hnrss.org/frontpage");
    assert.equal(
      edges.some((row) => row.target === "url:https://news.ycombinator.com/item?id=123"),
      true,
    );
    const advisoryNode = weaver.graph.getUrlNode("https://example.org/security/advisory");
    assert.equal(Boolean(advisoryNode?.feed_entry), true);
    assert.equal(String(advisoryNode?.feed_entry_source_kind || ""), "feed:rss");
    assert.ok(
      String(advisoryNode?.analysis_summary || "").toLowerCase().includes("critical advisory"),
    );
  } finally {
    global.fetch = originalFetch;
    weaver.shutdown();
  }
});

test("entityControl configure adjusts count and host limit", () => {
  const weaver = new WebGraphWeaver();
  try {
    const result = weaver.entityControl({
      action: "configure",
      count: 3,
      maxPerHost: 4,
      nodeCooldownMs: 90_000,
      activationThreshold: 0.75,
    });
    assert.equal(result.ok, true);
    assert.equal(weaver.entities.length, 3);
    assert.equal(weaver.currentMaxRequestsPerHost, 4);
    assert.equal(weaver.nodeCooldownMs, 90_000);
    assert.equal(weaver.activationThreshold, 0.75);
  } finally {
    weaver.shutdown();
  }
});

test("GraphStore restoreSnapshot hydrates graph and drops orphan edges", () => {
  const graph = new GraphStore({ maxUrlNodes: 100 });
  const counts = graph.restoreSnapshot({
    nodes: [
      { id: "domain:example.com", kind: "domain", domain: "example.com", label: "example.com" },
      { id: "url:https://example.com/", kind: "url", url: "https://example.com/", label: "https://example.com/" },
    ],
    edges: [
      {
        id: "edge:ok",
        kind: "domain_membership",
        source: "domain:example.com",
        target: "url:https://example.com/",
      },
      {
        id: "edge:orphan",
        kind: "hyperlink",
        source: "url:https://missing.example/",
        target: "url:https://example.com/",
      },
    ],
  });

  assert.equal(counts.nodes_total, 2);
  assert.equal(counts.edges_total, 1);
  assert.equal(counts.url_nodes_total, 1);
  assert.equal(graph.nodes.size, 2);
  assert.equal(graph.edges.size, 1);
  assert.equal(graph.edges.has("edge:ok"), true);
  assert.equal(graph.edges.has("edge:orphan"), false);
});

test("WebGraphWeaver _restoreFromSnapshot hydrates persisted status and graph", () => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "weaver-restore-"));
  const snapshotPath = path.join(tmpDir, "snapshot.json");
  fs.writeFileSync(snapshotPath, JSON.stringify({
    generated_at: "2026-03-03T00:00:00Z",
    status: {
      state: "paused",
      config: {
        max_depth: 4,
        max_nodes: 200,
        concurrency: 3,
        max_requests_per_host: 2,
        node_cooldown_ms: 4500,
        activation_threshold: 1.4,
        default_delay_ms: 900,
      },
      metrics: {
        discovered: 11,
        fetched: 7,
        skipped: 2,
        robots_blocked: 1,
        duplicate_content: 1,
        errors: 0,
        semantic_edges: 4,
        citation_edges: 1,
        wiki_reference_edges: 1,
        cross_reference_edges: 1,
        paper_pdf_edges: 1,
        host_concurrency_waits: 3,
        cooldown_blocked: 2,
        interactions: 5,
        activation_enqueues: 2,
        entity_moves: 3,
        entity_visits: 4,
        llm_analysis_success: 6,
        llm_analysis_fail: 1,
        average_fetch_ms: 55,
      },
      opt_out_domains: ["blocked.example"],
      entities: {
        enabled: true,
        paused: true,
        count: 1,
        entities: [
          {
            id: "entity:1",
            label: "crawler-1",
            state: "cooldown",
            current_url: "https://example.com/",
            visits: 9,
          },
        ],
      },
    },
    graph: {
      nodes: [
        { id: "domain:example.com", kind: "domain", label: "example.com", domain: "example.com" },
        { id: "url:https://example.com/", kind: "url", label: "https://example.com/", url: "https://example.com/", domain: "example.com" },
      ],
      edges: [
        {
          id: "edge:ok",
          kind: "domain_membership",
          source: "domain:example.com",
          target: "url:https://example.com/",
        },
      ],
    },
    events: [
      { event: "crawl_state", timestamp: 10, state: "paused" },
      { event: "graph_delta", timestamp: 11, reason: "restore" },
    ],
  }), "utf-8");

  const weaver = new WebGraphWeaver();
  try {
    const restored = weaver._restoreFromSnapshot(snapshotPath);
    assert.equal(restored, true);
    assert.equal(weaver.running, true);
    assert.equal(weaver.paused, true);
    assert.equal(weaver.currentMaxDepth, 4);
    assert.equal(weaver.currentMaxNodes >= 200, true);
    assert.equal(weaver.currentConcurrency, 3);
    assert.equal(weaver.currentMaxRequestsPerHost, 2);
    assert.equal(weaver.nodeCooldownMs, 4500);
    assert.equal(weaver.activationThreshold, 1.4);
    assert.equal(weaver.stats.fetched, 7);
    assert.equal(weaver.stats.duplicates, 1);
    assert.equal(weaver.graph.nodes.size, 2);
    assert.equal(weaver.graph.edges.size, 1);
    assert.equal(weaver.optOutDomains.has("blocked.example"), true);
    assert.equal(weaver.recentEvents.length >= 2, true);
    assert.equal(weaver.entities.length, 1);
  } finally {
    weaver.shutdown();
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});

test("WebGraphWeaver _persistSnapshot does not overwrite non-empty snapshot with empty graph", () => {
  const weaver = new WebGraphWeaver();
  const writeFileSyncSpy = fs.writeFileSync;
  const renameSyncSpy = fs.renameSync;
  let writeCount = 0;
  fs.writeFileSync = (...args) => {
    writeCount += 1;
    return writeFileSyncSpy(...args);
  };
  fs.renameSync = (...args) => renameSyncSpy(...args);

  try {
    weaver.lastSnapshotGraphCounts = {
      nodes_total: 12,
      edges_total: 25,
      url_nodes_total: 8,
    };
    weaver._persistSnapshot();
    assert.equal(writeCount, 0);

    weaver.graph.upsertDomain("example.com");
    weaver.graph.upsertUrl("https://example.com/", 0, null);
    weaver._persistSnapshot();
    assert.equal(writeCount, 1);
  } finally {
    weaver.shutdown();
    fs.writeFileSync = writeFileSyncSpy;
    fs.renameSync = renameSyncSpy;
  }
});

test("WebGraphWeaver _restoreFromSnapshot tolerates trailing snapshot corruption", () => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "weaver-restore-corrupt-"));
  const snapshotPath = path.join(tmpDir, "snapshot.json");
  const validPayload = {
    generated_at: "2026-03-03T00:00:00Z",
    status: {
      state: "stopped",
      config: {
        max_depth: 3,
        max_nodes: 100,
        concurrency: 2,
        max_requests_per_host: 2,
      },
      metrics: {
        fetched: 1,
      },
      entities: {
        enabled: true,
        paused: false,
        count: 1,
        entities: [],
      },
    },
    graph: {
      nodes: [
        {
          id: "domain:example.com",
          kind: "domain",
          label: "example.com",
          domain: "example.com",
        },
      ],
      edges: [],
    },
    events: [],
  };
  fs.writeFileSync(
    snapshotPath,
    `${JSON.stringify(validPayload)}\nTHIS_IS_TRAILING_CORRUPTION`,
    "utf-8",
  );

  const weaver = new WebGraphWeaver();
  try {
    const restored = weaver._restoreFromSnapshot(snapshotPath);
    assert.equal(restored, true);
    assert.equal(weaver.graph.nodes.size, 1);
  } finally {
    weaver.shutdown();
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});

test("WebGraphWeaver _restoreFromSnapshot keeps headroom above restored URL count", () => {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), "weaver-restore-headroom-"));
  const snapshotPath = path.join(tmpDir, "snapshot.json");
  fs.writeFileSync(snapshotPath, JSON.stringify({
    generated_at: "2026-03-04T00:00:00Z",
    status: {
      state: "stopped",
      config: {
        max_depth: 3,
        max_nodes: 2,
        concurrency: 2,
        max_requests_per_host: 2,
      },
      metrics: {
        fetched: 1,
      },
      entities: {
        enabled: true,
        paused: false,
        count: 1,
        entities: [],
      },
    },
    graph: {
      nodes: [
        { id: "domain:example.com", kind: "domain", label: "example.com", domain: "example.com" },
        { id: "url:https://example.com/a", kind: "url", label: "https://example.com/a", url: "https://example.com/a", domain: "example.com" },
        { id: "url:https://example.com/b", kind: "url", label: "https://example.com/b", url: "https://example.com/b", domain: "example.com" },
        { id: "url:https://example.com/c", kind: "url", label: "https://example.com/c", url: "https://example.com/c", domain: "example.com" },
      ],
      edges: [],
    },
    events: [],
  }), "utf-8");

  const weaver = new WebGraphWeaver();
  try {
    const restored = weaver._restoreFromSnapshot(snapshotPath);
    assert.equal(restored, true);
    assert.equal(weaver.graph.urlNodeCount, 3);
    assert.equal(weaver.currentMaxNodes > weaver.graph.urlNodeCount, true);
  } finally {
    weaver.shutdown();
    fs.rmSync(tmpDir, { recursive: true, force: true });
  }
});

test("WebGraphWeaver _handleMaxNodesReached autogrows node cap when work remains", () => {
  const weaver = new WebGraphWeaver();
  try {
    weaver.running = true;
    weaver.paused = false;
    weaver.currentMaxNodes = 128;
    weaver.graph.maxUrlNodes = 128;
    weaver.graph.urlNodeCount = 128;
    weaver.frontier.push({
      url: "https://example.com/pending",
      sourceUrl: null,
      depth: 0,
      readyAt: Date.now(),
      priority: 0,
    });

    weaver._handleMaxNodesReached();

    assert.equal(weaver.running, true);
    assert.equal(weaver.currentMaxNodes > 128, true);
    assert.equal(weaver.graph.maxUrlNodes, weaver.currentMaxNodes);
    assert.equal(
      weaver.recentEvents.some((row) =>
        row && row.event === "crawl_state" && row.reason === "max_nodes_autogrow"
      ),
      true,
    );
  } finally {
    weaver.shutdown();
  }
});

test("WebGraphWeaver _handleMaxNodesReached stops when no work remains", () => {
  const weaver = new WebGraphWeaver();
  try {
    weaver.running = true;
    weaver.paused = false;
    weaver.currentMaxNodes = 256;
    weaver.graph.maxUrlNodes = 256;
    weaver.graph.urlNodeCount = 256;

    weaver._handleMaxNodesReached();

    assert.equal(weaver.running, false);
    assert.equal(weaver.paused, false);
    assert.equal(
      weaver.recentEvents.some((row) =>
        row && row.event === "crawl_state" && row.reason === "max_nodes_reached"
      ),
      true,
    );
  } finally {
    weaver.shutdown();
  }
});
