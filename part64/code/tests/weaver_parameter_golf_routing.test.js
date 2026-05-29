const test = require("node:test");
const assert = require("node:assert/strict");

const modulePath = require.resolve("../web_graph_weaver.js");
delete require.cache[modulePath];
process.env.WEAVER_RESEARCH_MODE = "parameter-golf";

const { WebGraphWeaver } = require(modulePath);

test("Parameter Golf enqueue redirects robots-blocked patch diffs to canonical PR bridge URLs", () => {
  const weaver = new WebGraphWeaver();
  try {
    const patchUrl = "https://patch-diff.githubusercontent.com/raw/openai/parameter-golf/pull/276.patch";
    const sourceUrl = "https://parameter-golf.github.io/data/non_record_16m.json";
    weaver.graph.upsertUrl(sourceUrl, 0, null);
    const outcome = weaver.enqueueUrl(patchUrl, sourceUrl, 1, "hyperlink_discovered");

    assert.equal(outcome.ok, true);
    assert.equal(outcome.reason, "policy_redirect");
    assert.equal(outcome.redirected_from, patchUrl);
    assert.equal(outcome.url, "https://github.com/openai/parameter-golf/pull/276");
    assert.equal(weaver.optOutDomains.has("patch-diff.githubusercontent.com"), true);

    const blockedNode = weaver.graph.getUrlNode(patchUrl);
    assert.equal(blockedNode?.status, "blocked");
    assert.equal(blockedNode?.compliance, "policy_blocked");
    assert.equal(blockedNode?.preferred_bridge_url, "https://github.com/openai/parameter-golf/pull/276");

    const bridgedNode = weaver.graph.getUrlNode("https://github.com/openai/parameter-golf/pull/276");
    assert.equal(bridgedNode?.status, "queued");
  } finally {
    weaver.shutdown();
  }
});

test("Parameter Golf bootstrap seed selection skips policy-blocked patch diff URLs", () => {
  const weaver = new WebGraphWeaver();
  try {
    const blockedPatch = "https://patch-diff.githubusercontent.com/raw/openai/parameter-golf/pull/120.patch";
    const rawSubmission = "https://raw.githubusercontent.com/openai/parameter-golf/main/records/track_non_record_16mb/example/submission.json";
    weaver.graph.upsertUrl(blockedPatch, 0, null);
    weaver.graph.setUrlStatus(blockedPatch, {
      status: "discovered",
      activation_potential: 2.0,
    });
    weaver.graph.upsertUrl(rawSubmission, 0, null);
    weaver.graph.setUrlStatus(rawSubmission, {
      status: "discovered",
      activation_potential: 1.0,
    });

    const seeds = weaver._collectBootstrapGraphSeeds(8);
    assert.equal(seeds.includes(blockedPatch), false);
    assert.equal(seeds.includes(rawSubmission), true);
  } finally {
    weaver.shutdown();
  }
});

test("Parameter Golf candidate routing prefers raw follow-ons over repo navigation noise", () => {
  const weaver = new WebGraphWeaver();
  try {
    const sourceUrl = "https://github.com/openai/parameter-golf/pull/276";
    const rawSubmission = "https://raw.githubusercontent.com/openai/parameter-golf/main/records/track_non_record_16mb/example/submission.json";
    const repoBlob = "https://github.com/openai/parameter-golf/blob/main/records/track_non_record_16mb/example/README.md";

    weaver.graph.upsertUrl(sourceUrl, 0, null);
    weaver.graph.setUrlStatus(sourceUrl, {
      status: "fetched",
      activation_potential: 1.2,
    });
    weaver.graph.upsertUrl(rawSubmission, 1, sourceUrl);
    weaver.graph.setUrlStatus(rawSubmission, {
      status: "discovered",
      activation_potential: 1.1,
    });
    weaver.graph.upsertUrl(repoBlob, 1, sourceUrl);
    weaver.graph.setUrlStatus(repoBlob, {
      status: "discovered",
      activation_potential: 1.1,
    });
    weaver.graph.upsertEdge("hyperlink", `url:${sourceUrl}`, `url:${rawSubmission}`);
    weaver.graph.upsertEdge("hyperlink", `url:${sourceUrl}`, `url:${repoBlob}`);

    const entity = {
      id: "entity:test",
      current_url: sourceUrl,
    };
    const candidates = weaver._candidateTargetsForEntity(entity);

    assert.equal(candidates.length >= 2, true);
    assert.equal(candidates[0]?.url, rawSubmission);
    assert.equal(candidates.find((row) => row.url === repoBlob)?.score < candidates[0].score, true);
  } finally {
    weaver.shutdown();
  }
});