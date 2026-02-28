#!/usr/bin/env node
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const ralphLoopPath = join(__dirname, "ralph-loop.mjs");

const args = process.argv.slice(2);
args.push("--ulw");

const child = spawn(process.execPath, [ralphLoopPath, ...args], {
  stdio: "inherit"
});

child.on("close", (code) => {
  process.exit(code);
});
