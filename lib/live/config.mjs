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
