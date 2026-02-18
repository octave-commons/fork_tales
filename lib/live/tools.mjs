function lineify(text) {
  return text.replace(/\s+/g, " ").trim();
}

function singLine(input) {
  const style = input.style ?? "plain";
  const text = lineify(input.text ?? "");
  if (!text) return "♪ ... ♪";
  if (style === "staccato") return `♪ ${text.split(" ").join(" / ")} ♪`;
  if (style === "echo") return `♪ ${text} ... ${text} ♪`;
  return `♪ ${text} ♪`;
}

function frameEcho(input) {
  const frames = Array.isArray(input.frames) ? input.frames : [];
  if (!frames.length) return "No active frame pressure.";
  return `Frames: ${frames.join(", ")}`;
}

function tightenClaim(input) {
  const needs = Array.isArray(input.needs) ? input.needs : [];
  if (!needs.length) return "Claim already concrete.";
  return `Make it concrete with: ${needs.join(", ")}.`;
}

function offerOptions(input) {
  const base = lineify(input.text ?? "");
  return [
    `Option A: ${base} with owner + deadline`,
    `Option B: ${base} with evidence first`,
    `Option C: pause and define DoD`
  ].join(" | ");
}

const TOOL_IMPL = {
  sing_line: singLine,
  frame_echo: frameEcho,
  tighten_claim: tightenClaim,
  offer_options: offerOptions
};

export function runTool(toolName, input) {
  const fn = TOOL_IMPL[toolName];
  if (!fn) throw new Error(`Unknown tool: ${toolName}`);
  return fn(input ?? {});
}

export function listTools() {
  return Object.keys(TOOL_IMPL);
}
