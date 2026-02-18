import { analyzeUtterance } from "../analyze.mjs";
import { MockEntityLLM } from "./llm.mjs";
import { runTool } from "./tools.mjs";

function createEntityRuntime(entity, config) {
  const presenceProfile = config.presences[entity.presence] ?? { motiveByFrame: {} };
  if (entity.model.provider !== "mock") {
    throw new Error(`Unsupported provider '${entity.model.provider}' for entity '${entity.id}'.`);
  }
  return {
    entity,
    llm: new MockEntityLLM(entity, presenceProfile)
  };
}

export function createLiveEngine(config) {
  const runtimes = config.entities.map((entity) => createEntityRuntime(entity, config));
  const transcript = [];
  let turn = 0;

  async function processLine(userText, source = "stdin") {
    turn += 1;
    const id = `u${String(turn).padStart(4, "0")}`;
    const analysis = analyzeUtterance({
      id,
      text: userText,
      source,
      ts: new Date().toISOString(),
      wantRewrite: false
    });

    const events = [
      {
        kind: "user",
        id,
        text: userText,
        analysis
      }
    ];

    for (const runtime of runtimes) {
      const draft = await runtime.llm.respond({
        userText,
        analysis,
        transcript
      });

      const toolOutputs = [];
      for (const call of draft.toolCalls) {
        if (!runtime.entity.tools.includes(call.name)) continue;
        const output = runTool(call.name, call.input);
        toolOutputs.push({
          name: call.name,
          output
        });
      }

      const spoken = toolOutputs.find((t) => t.name === "sing_line")?.output ?? draft.text;
      const event = {
        kind: "entity",
        id: runtime.entity.id,
        label: runtime.entity.label,
        presence: runtime.entity.presence,
        said: spoken,
        tools: toolOutputs
      };

      transcript.push(event);
      events.push(event);
    }

    return events;
  }

  return {
    processLine,
    transcript
  };
}
