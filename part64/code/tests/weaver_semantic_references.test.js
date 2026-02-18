const test = require("node:test");
const assert = require("node:assert/strict");

const {
  extractArxivIdFromUrl,
  canonicalArxivAbsUrlFromId,
  canonicalArxivPdfUrlFromId,
  canonicalWikipediaArticleUrl,
  extractSemanticReferences,
  classifyKnowledgeUrl,
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
