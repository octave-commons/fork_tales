export const PRESENCE_OPERATIONAL_ROLE_BY_ID: Record<string, string> = {
  witness_thread: "crawl-routing",
  keeper_of_receipts: "file-analysis",
  mage_of_receipts: "image-captioning",
  anchor_registry: "council-orchestration",
  gates_of_truth: "compliance-gating",
  view_lens_keeper: "camera-guidance",
  health_sentinel_gpu1: "compute-scheduler",
  health_sentinel_gpu0: "compute-scheduler",
  health_sentinel_npu0: "compute-scheduler",
  health_sentinel_cpu: "compute-scheduler",
};

export const PANEL_TOOL_HINTS: Record<string, string[]> = {
  "nexus.ui.command_center": ["call", "say", "webrtc"],
  "nexus.ui.chat.witness_thread": ["ledger", "lineage", "particles"],
  "nexus.ui.chat.chaos": ["chat", "nearby", "pin"],
  "nexus.ui.chat.stability": ["chat", "nearby", "pin"],
  "nexus.ui.chat.symmetry": ["chat", "nearby", "pin"],
  "nexus.ui.web_graph_weaver": ["crawl", "queue", "graph"],
  "nexus.ui.threat_radar": ["threat", "security", "review"],
  "nexus.ui.inspiration_atlas": ["search", "curate", "seed"],
  "nexus.ui.entity_vitals": ["vitals", "telemetry", "watch"],
  "nexus.ui.projection_ledger": ["projection", "trace", "audit"],
  "nexus.ui.autopilot_ledger": ["autopilot", "risk", "gates"],
  "nexus.ui.world_log": ["receipts", "events", "review"],
  "nexus.ui.stability_observatory": ["study", "drift", "council"],
  "nexus.ui.runtime_config": ["config", "constants", "tuning"],
  "nexus.ui.daimoi_presence": ["daimoi", "presence", "focus"],
  "nexus.ui.omni_archive": ["catalog", "memories", "artifacts"],
  "nexus.ui.myth_commons": ["interact", "pray", "speak"],
  "nexus.ui.dedicated_views": ["overlay", "focus", "monitor"],
  "nexus.ui.glass_viewport": ["glass", "camera", "pan"],
};

export const COUNCIL_BOOST_STORAGE_KEY = "eta_mu.council_boosts.v1";
export const TERTIARY_PIN_STORAGE_KEY = "eta_mu.tertiary_pin.v1";
export const MUSE_WORKSPACE_STORAGE_KEY = "eta_mu.muse_workspace.v1";
export const INTERFACE_OPACITY_STORAGE_KEY = "eta_mu.interface_opacity.v2";
export const GLASS_VIEWPORT_PANEL_ID = "nexus.ui.glass_viewport";
export const RUNTIME_CONFIG_PANEL_ID = "nexus.ui.runtime_config";
export const INTERFACE_OPACITY_MIN = 0.72;
export const INTERFACE_OPACITY_MAX = 1;
export const DEFAULT_INTERFACE_TRANSPARENCY_PERCENT = 0;
export const DEFAULT_INTERFACE_OPACITY = 1 - (DEFAULT_INTERFACE_TRANSPARENCY_PERCENT / 100);

export const FIXED_MUSE_PRESENCES = [
  {
    id: "nexus.ui.chat.witness_thread",
    presenceId: "witness_thread",
    label: "Witness Thread",
  },
  {
    id: "nexus.ui.chat.chaos",
    presenceId: "chaos",
    label: "Chaos",
  },
  {
    id: "nexus.ui.chat.stability",
    presenceId: "stability",
    label: "Stability",
  },
  {
    id: "nexus.ui.chat.symmetry",
    presenceId: "symmetry",
    label: "Symmetry",
  },
] as const;

export const APP_WORKSPACE_NORMALIZE_OPTIONS = {
  maxPinnedFileNodeIds: 48,
  maxSearchQueryLength: 180,
  maxPinnedNexusSummaries: 24,
} as const;

export const USER_PRESENCE_BATCH_IDLE_FLUSH_MS = 2400;
export const USER_PRESENCE_BATCH_MAX_WINDOW_MS = 60_000;
export const USER_PRESENCE_BATCH_MAX_EVENTS = 36;

export function isGlassPrimaryPanelId(panelId: string): boolean {
  return panelId === GLASS_VIEWPORT_PANEL_ID || panelId === "nexus.ui.dedicated_views";
}
