#!/usr/bin/env node
import { spawn } from "node:child_process";
import { parseArgs } from "node:util";

const options = {
  "max-iterations": { type: "string", default: "5" },
  "completion-promise": { type: "string", default: "DONE" },
  "min-tool-calls": { type: "string", default: "1" },
  "on-promise-no-work": { type: "string", default: "reject" },
  "ulw": { type: "boolean", default: false }
};

const { values, positionals } = parseArgs({ args: process.argv.slice(2), options, allowPositionals: true, strict: false });

const task = positionals.join(" ").trim();
if (!task) {
  console.error("Usage: ralph-loop \"<task>\" [--max-iterations <N>] [--completion-promise <PROMISE>]");
  process.exit(1);
}

const maxIterations = parseInt(values["max-iterations"], 10);
const promiseToken = values["completion-promise"];
const minToolCalls = parseInt(values["min-tool-calls"], 10);
const isUlw = values["ulw"];

const promiseMarker = `<promise>${promiseToken}</promise>`;

const startPrompt = `${task}

---
STOPPING RULE: Only when fully done, output the exact promise marker ${promiseMarker}.
Must match exactly, no extra whitespace inside the tag, no variants.
If not done, do not output the promise.`;

const baseContinuationPrompt = `You attempted to end the loop, but you did not output the completion promise ${promiseMarker}. Continue working on the original task below. Only output the promise when the task is fully complete. Original task: ${task}`;

const bypassContinuationPrompt = `Promise detected, but no tool calls/work were performed; you must actually complete the task before emitting the promise. Original task: ${task}`;

async function runOpencode(prompt, isContinue) {
  return new Promise((resolve, reject) => {
    const args = ["run"];
    if (isContinue) {
      args.push("--continue");
    }
    args.push(prompt);

    const child = spawn("opencode", args, { stdio: ["inherit", "pipe", "pipe"] });
    let output = "";
    let stderrOutput = "";

    child.stdout.on("data", (data) => {
      process.stdout.write(data);
      output += data.toString();
    });

    child.stderr.on("data", (data) => {
      process.stderr.write(data);
      stderrOutput += data.toString();
    });

    child.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`opencode exited with code ${code}`));
      } else {
        resolve({ output, stderrOutput });
      }
    });
  });
}

// Simple heuristic for tool calls (looking for typical OpenCode tool usage markers in output/stderr)
function countToolCalls(output, stderrOutput) {
  const matches = output.match(/call:.*\{/g);
  return matches ? matches.length : 0;
}

async function main() {
  let iteration = 0;
  let isContinue = false;
  let currentPrompt = startPrompt;
  
  while (iteration < maxIterations) {
    console.log(`\n--- Loop Iteration ${iteration} ---`);
    
    try {
      const { output, stderrOutput } = await runOpencode(currentPrompt, isContinue);
      
      const hasPromise = output.includes(promiseMarker);
      const toolCalls = countToolCalls(output, stderrOutput);
      
      if (hasPromise) {
        if (toolCalls >= minToolCalls || minToolCalls === 0) {
          console.log(`\n[ralph-loop] PROMISE_ACCEPTED (Iter ${iteration}, Tool calls: ${toolCalls})`);
          process.exit(0);
        } else {
          console.log(`\n[ralph-loop] PROMISE_REJECTED (Bypass prevention: 0 tool calls). Reinjecting...`);
          currentPrompt = isUlw ? `[ULTRAWORK MODE] ${bypassContinuationPrompt}` : bypassContinuationPrompt;
        }
      } else {
        console.log(`\n[ralph-loop] Promise missing. Blocking exit. Reinjecting prompt...`);
        currentPrompt = isUlw ? `[ULTRAWORK MODE] ${baseContinuationPrompt}` : baseContinuationPrompt;
      }
      
      isContinue = true;
      iteration++;
    } catch (err) {
      console.error(`\n[ralph-loop] Execution failed: ${err.message}`);
      process.exit(1);
    }
  }

  console.log("\n[ralph-loop] MAX_ITERATIONS_REACHED");
  process.exit(1);
}

main().catch(console.error);
