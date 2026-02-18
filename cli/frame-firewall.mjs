#!/usr/bin/env node
import fs from "node:fs";
import { analyzeUtterance } from "../lib/analyze.mjs";

function readAllStdin() {
  return new Promise((resolve, reject) => {
    let data = "";
    process.stdin.setEncoding("utf8");
    process.stdin.on("data", (c) => {
      data += c;
    });
    process.stdin.on("end", () => resolve(data));
    process.stdin.on("error", reject);
  });
}

const argv = process.argv.slice(2);
const file = argv.find((a) => !a.startsWith("--"));
const wantRewrite = argv.includes("--rewrite");

const input = file ? fs.readFileSync(file, "utf8") : await readAllStdin();

const lines = input
  .split(/\r?\n/)
  .map((s) => s.trim())
  .filter(Boolean);

for (let i = 0; i < lines.length; i++) {
  const text = lines[i];
  const out = analyzeUtterance({
    id: `u${String(i + 1).padStart(4, "0")}`,
    text,
    source: file ? `file:${file}` : "stdin",
    ts: new Date().toISOString(),
    wantRewrite
  });
  process.stdout.write(JSON.stringify(out) + "\n");
}
