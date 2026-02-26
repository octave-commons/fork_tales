#!/usr/bin/env node
// SPDX-License-Identifier: GPL-3.0-or-later
// This file is part of Fork Tales.
// Copyright (C) 2024-2025 Fork Tales Contributors
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";

const RECORD_ID = "promethean.fork-tax-git/v1";

const DEFAULTS = Object.freeze({
  maxDirtyMinutes: 120,
  maxUnpushedCommits: 5,
  maxUntrackedItems: 40,
  owner: "Err",
  dod: "checkpoint committed and push-ready",
  push: false,
  includeUntracked: false,
  message: "",
  mode: "audit",
  repoRoot: process.cwd(),
});

function parseNumber(value, fallback) {
  const parsed = Number.parseInt(String(value), 10);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function parseArgs(argv) {
  const options = { ...DEFAULTS };
  for (let idx = 0; idx < argv.length; idx += 1) {
    const token = argv[idx];
    if (token === "--audit") {
      options.mode = "audit";
      continue;
    }
    if (token === "--cycle") {
      options.mode = "cycle";
      continue;
    }
    if (token === "--push") {
      options.push = true;
      continue;
    }
    if (token === "--include-untracked") {
      options.includeUntracked = true;
      continue;
    }
    if (token === "--owner" && argv[idx + 1]) {
      options.owner = argv[idx + 1];
      idx += 1;
      continue;
    }
    if (token === "--dod" && argv[idx + 1]) {
      options.dod = argv[idx + 1];
      idx += 1;
      continue;
    }
    if (token === "--message" && argv[idx + 1]) {
      options.message = argv[idx + 1];
      idx += 1;
      continue;
    }
    if (token === "--repo" && argv[idx + 1]) {
      options.repoRoot = path.resolve(argv[idx + 1]);
      idx += 1;
      continue;
    }
    if (token === "--max-dirty-minutes" && argv[idx + 1]) {
      options.maxDirtyMinutes = parseNumber(argv[idx + 1], options.maxDirtyMinutes);
      idx += 1;
      continue;
    }
    if (token === "--max-unpushed-commits" && argv[idx + 1]) {
      options.maxUnpushedCommits = parseNumber(argv[idx + 1], options.maxUnpushedCommits);
      idx += 1;
      continue;
    }
    if (token === "--max-untracked-items" && argv[idx + 1]) {
      options.maxUntrackedItems = parseNumber(argv[idx + 1], options.maxUntrackedItems);
      idx += 1;
      continue;
    }
    if (token === "--test") {
      options.mode = "test";
      continue;
    }
    throw new Error(`Unknown argument: ${token}`);
  }
  return options;
}

function runGit(repoRoot, args, { allowFail = false } = {}) {
  try {
    return execFileSync("git", args, {
      cwd: repoRoot,
      encoding: "utf8",
      stdio: ["ignore", "pipe", "pipe"],
    }).trimEnd();
  } catch (error) {
    if (allowFail) {
      return null;
    }
    const stdout = String(error?.stdout ?? "").trim();
    const stderr = String(error?.stderr ?? "").trim();
    const details = [stdout, stderr].filter(Boolean).join("\n");
    throw new Error(`git ${args.join(" ")} failed${details ? `: ${details}` : ""}`);
  }
}

function safeField(value) {
  return String(value ?? "")
    .replace(/[|\n\r]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function parseStatus(output) {
  const lines = String(output ?? "")
    .split(/\r?\n/)
    .filter(Boolean);
  const header = lines[0]?.startsWith("## ") ? lines[0].slice(3) : "HEAD";

  let branch = header;
  let upstream = null;
  let ahead = 0;
  let behind = 0;

  if (header.startsWith("No commits yet on ")) {
    branch = header.slice("No commits yet on ".length).trim();
  } else {
    let mainPart = header;
    let trackPart = "";
    const bracketIdx = header.indexOf(" [");
    if (bracketIdx >= 0) {
      mainPart = header.slice(0, bracketIdx);
      trackPart = header.slice(bracketIdx + 2, header.endsWith("]") ? -1 : undefined);
    }
    if (mainPart.includes("...")) {
      const [localBranch, tracked] = mainPart.split("...", 2);
      branch = localBranch || "HEAD";
      upstream = tracked || null;
    } else {
      branch = mainPart || "HEAD";
    }
    if (trackPart) {
      const aheadMatch = trackPart.match(/ahead\s+(\d+)/);
      const behindMatch = trackPart.match(/behind\s+(\d+)/);
      ahead = aheadMatch ? Number.parseInt(aheadMatch[1], 10) : 0;
      behind = behindMatch ? Number.parseInt(behindMatch[1], 10) : 0;
    }
  }

  const staged = [];
  const unstaged = [];
  const untracked = [];
  const tracked = [];

  for (const line of lines.slice(1)) {
    if (line.length < 3) continue;
    const status = line.slice(0, 2);
    const filePath = line.slice(3).trim();
    if (!filePath) continue;
    if (status === "??") {
      untracked.push(filePath);
      continue;
    }
    const x = status[0];
    const y = status[1];
    tracked.push(filePath);
    if (x !== " ") {
      staged.push(filePath);
    }
    if (y !== " ") {
      unstaged.push(filePath);
    }
  }

  return {
    branch: branch.trim(),
    upstream,
    ahead,
    behind,
    staged,
    unstaged,
    untracked,
    tracked,
    dirty: tracked.length > 0 || untracked.length > 0,
  };
}

function minutesSince(epochSeconds) {
  if (!Number.isFinite(epochSeconds)) return null;
  const nowSeconds = Math.floor(Date.now() / 1000);
  return Math.max(0, Math.floor((nowSeconds - epochSeconds) / 60));
}

function collectGitState(repoRoot) {
  const repoCheck = runGit(repoRoot, ["rev-parse", "--is-inside-work-tree"], {
    allowFail: true,
  });
  if (repoCheck !== "true") {
    throw new Error(`Not a git repository: ${repoRoot}`);
  }

  const normalizedRoot = runGit(repoRoot, ["rev-parse", "--show-toplevel"]) || repoRoot;
  const statusOutput = runGit(normalizedRoot, ["status", "--porcelain", "--branch"]) || "";
  const parsedStatus = parseStatus(statusOutput);
  const commitEpochRaw = runGit(normalizedRoot, ["log", "-1", "--format=%ct"], {
    allowFail: true,
  });
  const commitEpoch = commitEpochRaw ? Number.parseInt(commitEpochRaw, 10) : Number.NaN;
  const remoteOrigin = runGit(normalizedRoot, ["config", "--get", "remote.origin.url"], {
    allowFail: true,
  });

  const inferredUpstream =
    parsedStatus.upstream ||
    runGit(normalizedRoot, ["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], {
      allowFail: true,
    });

  return {
    repoRoot: normalizedRoot,
    branch: parsedStatus.branch,
    upstream: inferredUpstream,
    ahead: parsedStatus.ahead,
    behind: parsedStatus.behind,
    dirty: parsedStatus.dirty,
    trackedChanges: parsedStatus.tracked.length,
    stagedChanges: parsedStatus.staged.length,
    unstagedChanges: parsedStatus.unstaged.length,
    untrackedChanges: parsedStatus.untracked.length,
    stagedPaths: parsedStatus.staged,
    unstagedPaths: parsedStatus.unstaged,
    untrackedPaths: parsedStatus.untracked,
    commitEpoch: Number.isFinite(commitEpoch) ? commitEpoch : null,
    minutesSinceCommit: minutesSince(commitEpoch),
    remoteOrigin,
    receiptsPresent: fs.existsSync(path.join(normalizedRoot, "receipts.log")),
  };
}

function rankStatus(current, incoming) {
  const order = { pass: 0, watch: 1, block: 2 };
  return order[incoming] > order[current] ? incoming : current;
}

function makePresence(id, status, facts, asks, repairs, refs) {
  return {
    id,
    status,
    say_intent: {
      facts,
      asks,
      repairs,
      constraints: {
        no_new_facts: true,
        cite_refs: true,
        max_lines: 8,
      },
    },
    refs,
  };
}

function unique(items) {
  return [...new Set(items.filter(Boolean))];
}

function evaluatePresences(state, options) {
  const refs = [
    "contracts/contract_fork_tax_git_v1.mjs",
    ".opencode/command/promethean.fork-tax-git.v1.md",
    ".opencode/promptdb/contracts/receipts.v2.contract.lisp",
  ];

  const presences = [];

  {
    let status = "pass";
    const facts = [
      `branch=${state.branch}`,
      `minutes_since_commit=${state.minutesSinceCommit ?? "none"}`,
      `ahead=${state.ahead}`,
      `behind=${state.behind}`,
    ];
    const asks = [];
    const repairs = [];

    if (state.minutesSinceCommit === null) {
      status = rankStatus(status, "block");
      asks.push("Initialize commit lineage on this branch.");
      repairs.push("Run a first checkpoint commit before editing further.");
    }

    if (state.dirty && (state.minutesSinceCommit ?? 0) > options.maxDirtyMinutes) {
      status = rankStatus(status, "block");
      asks.push(`Dirty tree is older than ${options.maxDirtyMinutes} minutes.`);
      repairs.push("Run checkpoint cycle and close current drift with a commit.");
    }

    if (state.ahead > options.maxUnpushedCommits) {
      status = rankStatus(status, "block");
      asks.push(
        `Branch is ${state.ahead} commits ahead (limit ${options.maxUnpushedCommits}).`,
      );
      repairs.push("Push branch to upstream or reduce ahead count before new work.");
    }

    presences.push(makePresence("fork-tax-canticle", status, unique(facts), unique(asks), unique(repairs), refs));
  }

  {
    let status = "pass";
    const facts = [
      `tracked_changes=${state.trackedChanges}`,
      `staged_changes=${state.stagedChanges}`,
      `untracked_changes=${state.untrackedChanges}`,
    ];
    const asks = [];
    const repairs = [];

    if (state.untrackedChanges > options.maxUntrackedItems) {
      status = rankStatus(status, "watch");
      asks.push(
        `Untracked set is ${state.untrackedChanges} items (threshold ${options.maxUntrackedItems}).`,
      );
      repairs.push("Triage untracked files: commit intentionally or ignore explicitly.");
    }

    if (state.trackedChanges > 0 && state.stagedChanges === 0) {
      status = rankStatus(status, "watch");
      asks.push("Tracked changes exist but no staged checkpoint.");
      repairs.push("Stage tracked changes (`git add -u`) before checkpoint commit.");
    }

    presences.push(makePresence("file-sentinel", status, unique(facts), unique(asks), unique(repairs), refs));
  }

  {
    let status = "pass";
    const facts = [
      `remote_origin=${state.remoteOrigin || "none"}`,
      `upstream=${state.upstream || "none"}`,
    ];
    const asks = [];
    const repairs = [];

    if (!state.remoteOrigin) {
      status = rankStatus(status, "block");
      asks.push("No remote origin configured.");
      repairs.push("Add remote origin so push events can pay fork tax publicly.");
    } else if (!state.upstream) {
      status = rankStatus(status, "watch");
      asks.push("Branch has no upstream tracking reference.");
      repairs.push("Set upstream with `git push -u origin <branch>`.");
    }

    if (state.behind > 0) {
      status = rankStatus(status, "watch");
      asks.push(`Branch is behind upstream by ${state.behind} commits.`);
      repairs.push("Reconcile remote changes before pushing new checkpoints.");
    }

    presences.push(makePresence("witness-thread", status, unique(facts), unique(asks), unique(repairs), refs));
  }

  {
    let status = "pass";
    const facts = [`receipts_log_present=${state.receiptsPresent}`];
    const asks = [];
    const repairs = [];
    if (!state.receiptsPresent) {
      status = "block";
      asks.push("receipts.log missing at repository root.");
      repairs.push("Create receipts.log before claiming fork-tax compliance.");
    }
    presences.push(makePresence("keeper-of-receipts", status, facts, asks, repairs, refs));
  }

  {
    const facts = [
      `owner=${options.owner}`,
      `dod=${options.dod}`,
    ];
    const asks = [];
    const repairs = [];
    let status = "pass";
    if (!safeField(options.owner)) {
      status = "watch";
      asks.push("Owner is empty for checkpoint protocol.");
      repairs.push("Provide --owner when running cycle.");
    }
    if (!safeField(options.dod)) {
      status = rankStatus(status, "watch");
      asks.push("DoD is empty for checkpoint protocol.");
      repairs.push("Provide --dod when running cycle.");
    }
    presences.push(makePresence("anchor-registry", status, facts, asks, repairs, refs));
  }

  return presences;
}

function buildSayIntent(presences) {
  const facts = [];
  const asks = [];
  const repairs = [];
  for (const presence of presences) {
    const intent = presence.say_intent;
    facts.push(...intent.facts.slice(0, 2));
    asks.push(...intent.asks.slice(0, 2));
    repairs.push(...intent.repairs.slice(0, 2));
  }
  return {
    facts: unique(facts).slice(0, 8),
    asks: unique(asks).slice(0, 8),
    repairs: unique(repairs).slice(0, 8),
    constraints: {
      no_new_facts: true,
      cite_refs: true,
      max_lines: 8,
    },
  };
}

function buildGate(presences, state) {
  const blocked = [];
  for (const presence of presences) {
    if (presence.status === "block") {
      blocked.push(presence.id);
    }
  }
  const commitReady = state.dirty;
  const pushReady = Boolean(state.remoteOrigin) && Boolean(state.upstream) && state.behind === 0;
  return {
    commit_ready: commitReady,
    push_ready: pushReady,
    blocked_by: blocked,
  };
}

function buildAudit(state, options) {
  const presences = evaluatePresences(state, options);
  return {
    record: `${RECORD_ID}.audit`,
    ts: new Date().toISOString(),
    thresholds: {
      max_dirty_minutes: options.maxDirtyMinutes,
      max_unpushed_commits: options.maxUnpushedCommits,
      max_untracked_items: options.maxUntrackedItems,
    },
    git: {
      repo_root: state.repoRoot,
      branch: state.branch,
      upstream: state.upstream,
      remote_origin: state.remoteOrigin,
      dirty: state.dirty,
      tracked_changes: state.trackedChanges,
      staged_changes: state.stagedChanges,
      unstaged_changes: state.unstagedChanges,
      untracked_changes: state.untrackedChanges,
      ahead: state.ahead,
      behind: state.behind,
      minutes_since_commit: state.minutesSinceCommit,
      samples: {
        staged: state.stagedPaths.slice(0, 8),
        unstaged: state.unstagedPaths.slice(0, 8),
        untracked: state.untrackedPaths.slice(0, 8),
      },
    },
    presences,
    say_intent: buildSayIntent(presences),
    gate: buildGate(presences, state),
  };
}

function defaultCheckpointMessage() {
  const stamp = new Date().toISOString().replace("T", " ").slice(0, 16);
  return `chore(fork-tax): checkpoint ${stamp}`;
}

function appendDecisionReceipt(repoRoot, options, note, refs) {
  const receiptsPath = path.join(repoRoot, "receipts.log");
  const line = [
    `ts=${new Date().toISOString()}`,
    "kind=:decision",
    "origin=working-tree",
    `owner=${safeField(options.owner)}`,
    `dod=${safeField(options.dod)}`,
    "pi=fork-tax-git-v1",
    `host=${safeField(options.remoteOrigin || "local:no-remote")}`,
    "manifest=manifest.lith",
    `refs=${safeField(refs.join(","))}`,
    `note=${safeField(note)}`,
  ].join(" | ");
  fs.appendFileSync(receiptsPath, `${line}\n`, "utf8");
  return {
    path: "receipts.log",
    line,
  };
}

function runCycle(repoRoot, options) {
  const actions = [];
  const before = collectGitState(repoRoot);
  const beforeAudit = buildAudit(before, options);

  if (!before.dirty) {
    return {
      record: `${RECORD_ID}.cycle`,
      ts: new Date().toISOString(),
      mode: "no-op",
      reason: "working tree clean",
      before: beforeAudit,
      actions,
    };
  }

  if (options.includeUntracked) {
    runGit(before.repoRoot, ["add", "-A"]);
    actions.push({ kind: "stage", command: "git add -A" });
  } else {
    runGit(before.repoRoot, ["add", "-u"]);
    actions.push({ kind: "stage", command: "git add -u" });
  }

  const stagedState = collectGitState(before.repoRoot);
  if (stagedState.stagedChanges === 0) {
    return {
      record: `${RECORD_ID}.cycle`,
      ts: new Date().toISOString(),
      mode: "no-op",
      reason: "no staged tracked changes",
      before: beforeAudit,
      after: buildAudit(stagedState, options),
      actions,
    };
  }

  const commitMessage = safeField(options.message) || defaultCheckpointMessage();
  const refs = [
    "contracts/contract_fork_tax_git_v1.mjs",
    ".opencode/command/promethean.fork-tax-git.v1.md",
    "specs/drafts/fork-tax-git-cadence-protocol.md",
  ];
  const receipt = appendDecisionReceipt(before.repoRoot, { ...options, remoteOrigin: before.remoteOrigin }, commitMessage, refs);
  actions.push({ kind: "receipt", path: receipt.path });

  runGit(before.repoRoot, ["add", "receipts.log"]);
  actions.push({ kind: "stage", command: "git add receipts.log" });

  runGit(before.repoRoot, ["commit", "-m", commitMessage]);
  const commit = runGit(before.repoRoot, ["rev-parse", "--short", "HEAD"]);
  actions.push({ kind: "commit", message: commitMessage, sha: commit });

  let pushResult = null;
  if (options.push) {
    const stateAfterCommit = collectGitState(before.repoRoot);
    if (!stateAfterCommit.remoteOrigin) {
      pushResult = {
        ok: false,
        reason: "missing-remote-origin",
      };
    } else if (!stateAfterCommit.upstream) {
      runGit(before.repoRoot, ["push", "-u", "origin", stateAfterCommit.branch]);
      pushResult = {
        ok: true,
        command: `git push -u origin ${stateAfterCommit.branch}`,
      };
    } else {
      runGit(before.repoRoot, ["push"]);
      pushResult = {
        ok: true,
        command: "git push",
      };
    }
    actions.push({ kind: "push", result: pushResult });
  }

  const after = collectGitState(before.repoRoot);
  return {
    record: `${RECORD_ID}.cycle`,
    ts: new Date().toISOString(),
    mode: "applied",
    before: beforeAudit,
    after: buildAudit(after, options),
    actions,
    push: pushResult,
  };
}

function runAudit(repoRoot, options) {
  const state = collectGitState(repoRoot);
  return buildAudit(state, options);
}

function runSelfTests() {
  {
    const parsed = parseStatus(
      "## main...origin/main [ahead 2, behind 1]\n M a.txt\nM  b.txt\n?? c.bin\n",
    );
    assert.equal(parsed.branch, "main");
    assert.equal(parsed.upstream, "origin/main");
    assert.equal(parsed.ahead, 2);
    assert.equal(parsed.behind, 1);
    assert.equal(parsed.staged.length, 1);
    assert.equal(parsed.unstaged.length, 1);
    assert.equal(parsed.untracked.length, 1);
  }

  {
    const line = safeField("a | b\n c\r\n");
    assert.equal(line, "a b c");
  }

  {
    const state = {
      repoRoot: "/tmp/repo",
      branch: "main",
      upstream: null,
      ahead: 7,
      behind: 0,
      dirty: true,
      trackedChanges: 3,
      stagedChanges: 0,
      unstagedChanges: 3,
      untrackedChanges: 91,
      stagedPaths: [],
      unstagedPaths: ["a", "b", "c"],
      untrackedPaths: ["x"],
      commitEpoch: Math.floor(Date.now() / 1000) - 60 * 180,
      minutesSinceCommit: 180,
      remoteOrigin: null,
      receiptsPresent: true,
    };
    const options = { ...DEFAULTS };
    const audit = buildAudit(state, options);
    assert.equal(audit.gate.commit_ready, true);
    assert.equal(audit.gate.push_ready, false);
    assert.ok(audit.gate.blocked_by.includes("fork-tax-canticle"));
    assert.ok(audit.gate.blocked_by.includes("witness-thread"));
  }

  process.stdout.write(
    JSON.stringify({ record: `${RECORD_ID}.test`, ok: true, ts: new Date().toISOString() }, null, 2) + "\n",
  );
}

function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.mode === "test") {
    runSelfTests();
    return;
  }
  if (options.mode === "audit") {
    const payload = runAudit(options.repoRoot, options);
    process.stdout.write(`${JSON.stringify(payload, null, 2)}\n`);
    return;
  }
  if (options.mode === "cycle") {
    const payload = runCycle(options.repoRoot, options);
    process.stdout.write(`${JSON.stringify(payload, null, 2)}\n`);
    return;
  }
  throw new Error(`Unsupported mode: ${options.mode}`);
}

main();
