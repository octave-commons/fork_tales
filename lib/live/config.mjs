import fs from "node:fs";

export function defaultLiveConfig() {
  return {
    entities: [
      {
        id: "river",
        label: "Receipt River",
        presence: "witness",
        model: { provider: "mock", name: "river-voice" },
        tools: ["sing_line", "frame_echo"]
      },
      {
        id: "forge",
        label: "Forge Sparrow",
        presence: "challenge",
        model: { provider: "mock", name: "forge-voice" },
        tools: ["sing_line", "tighten_claim"]
      },
      {
        id: "chorus",
        label: "Chorus-3",
        presence: "bridge",
        model: { provider: "mock", name: "chorus-voice" },
        tools: ["sing_line", "offer_options"]
      }
    ],
    presences: {
      witness: {
        motiveByFrame: {
          guilt: "restore agency",
          authority: "ask for evidence",
          urgency: "stabilize tempo",
          vagueness: "ask for specifics",
          agency_theft: "return choice"
        }
      },
      challenge: {
        motiveByFrame: {
          guilt: "reject coercion",
          authority: "demand proof",
          urgency: "demand deadline",
          vagueness: "pin down commitments",
          agency_theft: "refuse blind trust"
        }
      },
      bridge: {
        motiveByFrame: {
          guilt: "offer alternatives",
          authority: "cite evidence path",
          urgency: "scope to one next step",
          vagueness: "convert to owner+DoD",
          agency_theft: "preserve autonomy"
        }
      }
    }
  };
}

export function loadLiveConfig(configPath) {
  if (!configPath) return defaultLiveConfig();
  const raw = fs.readFileSync(configPath, "utf8");
  return JSON.parse(raw);
}
