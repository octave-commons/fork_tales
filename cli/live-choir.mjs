#!/usr/bin/env node
import readline from "node:readline";
import { createLiveEngine } from "../lib/live/engine.mjs";
import { loadLiveConfig } from "../lib/live/config.mjs";

function parseArgs(argv) {
  const out = { configPath: null, json: true };
  for (let i = 0; i < argv.length; i += 1) {
    const token = argv[i];
    if (token === "--config" && argv[i + 1]) {
      out.configPath = argv[i + 1];
      i += 1;
      continue;
    }
    if (token === "--plain") {
      out.json = false;
    }
  }
  return out;
}

function printEvents(events, jsonMode) {
  for (const event of events) {
    if (jsonMode) {
      process.stdout.write(JSON.stringify(event) + "\n");
      continue;
    }
    if (event.kind === "user") {
      process.stdout.write(`you> ${event.text}\n`);
      continue;
    }
    process.stdout.write(`${event.label}> ${event.said}\n`);
  }
}

const args = parseArgs(process.argv.slice(2));
const config = loadLiveConfig(args.configPath);
const engine = createLiveEngine(config);

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  terminal: process.stdin.isTTY
});

if (process.stdin.isTTY) {
  process.stdout.write("live-choir ready. type a line, or /quit\n");
}

rl.on("line", async (line) => {
  const text = line.trim();
  if (!text) return;
  if (text === "/quit" || text === "/exit") {
    rl.close();
    return;
  }
  try {
    const events = await engine.processLine(text, "live-chat");
    printEvents(events, args.json);
  } catch (error) {
    process.stderr.write(`[live-choir] ${error.message}\n`);
  }
});

rl.on("close", () => {
  process.exit(0);
});
