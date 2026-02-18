export class MockEntityLLM {
  constructor(entity, presenceProfile) {
    this.entity = entity;
    this.presenceProfile = presenceProfile ?? { motiveByFrame: {} };
  }

  async respond(context) {
    const primaryFrame = context.analysis.frames[0] ?? "none";
    const motive = this.presenceProfile.motiveByFrame[primaryFrame] ?? "keep signal clear";
    const needs = context.analysis.needs.length ? context.analysis.needs.join(", ") : "none";

    const toolCalls = [];
    if (this.entity.tools.includes("frame_echo") && context.analysis.frames.length) {
      toolCalls.push({
        name: "frame_echo",
        input: { frames: context.analysis.frames }
      });
    }
    if (this.entity.tools.includes("tighten_claim") && context.analysis.needs.length) {
      toolCalls.push({
        name: "tighten_claim",
        input: { needs: context.analysis.needs }
      });
    }
    if (this.entity.tools.includes("offer_options") && context.analysis.etaGap > 0) {
      toolCalls.push({
        name: "offer_options",
        input: { text: context.userText }
      });
    }

    let text = `${this.entity.label}: motive=${motive}; etaGap=${context.analysis.etaGap}; needs=${needs}.`;
    if (this.entity.tools.includes("sing_line")) {
      toolCalls.push({
        name: "sing_line",
        input: {
          text,
          style: context.analysis.etaGap > 2 ? "staccato" : "plain"
        }
      });
      text = "";
    }

    return { text, toolCalls };
  }
}
