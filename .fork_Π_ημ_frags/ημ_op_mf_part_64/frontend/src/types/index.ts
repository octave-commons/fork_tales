export interface WorldPayload {
  part: number;
  seed_label: string;
  file_count: number;
  generated_at: string;
  roles: Record<string, number>;
  constraints: string[];
}

export interface CatalogItem {
  part: string;
  name: string;
  role: string;
  display_name: { en: string; ja: string };
  display_role: { en: string; ja: string };
  kind: string;
  bytes: number;
  mtime_utc: string;
  rel_path: string;
  url: string;
}

export interface EntityVitals {
  [key: string]: string | number;
}

export interface EntityState {
  id: string;
  bpm: number;
  stability: number;
  resonance: number;
  vitals?: EntityVitals;
}

export interface SimPoint {
  x: number;
  y: number;
  size: number;
  r: number;
  g: number;
  b: number;
}

export interface Echo {
  id: string;
  text: string;
  x: number;
  y: number;
  hue: number;
  life: number;
}

export interface MythSummary {
  generated_at: string;
  event_type: string;
  ledger_size: number;
  top_cover_claim: string;
  top_cover_weight: number;
  cover_attribution: Record<string, number>;
  media_attribution: Record<string, number>;
  remote: Record<string, unknown> | null;
}

export interface WorldPresence {
  id: string;
  name: { en: string; ja: string };
  type: string;
}

export interface WorldPerson {
  id: string;
  name: { en: string; ja: string };
  role: { en: string; ja: string };
  instrument: string;
  prays_to: string;
  devotion: number;
  prayer_intensity: number;
  mood: number;
  hymn_bpm: number;
}

export interface WorldSong {
  id: string;
  leader: { en: string; ja: string };
  title: { en: string; ja: string };
  bpm: number;
  energy: number;
}

export interface WorldBook {
  id: string;
  title: { en: string; ja: string };
  author: { en: string; ja: string };
  excerpt: { en: string; ja: string };
  written_at_tick: number;
}

export interface WorldSummary {
  generated_at: string;
  tick: number;
  presences: WorldPresence[];
  people: WorldPerson[];
  songs: WorldSong[];
  books: WorldBook[];
  prayer_intensity: number;
}

export interface WorldInteractionResponse {
  ok: boolean;
  error?: string;
  action?: string;
  tick?: number;
  speaker?: { en: string; ja: string };
  presence?: {
    id: string;
    name: { en: string; ja: string };
    type: string;
  };
  line_en: string;
  line_ja: string;
  voice_text_en?: string;
  voice_text_ja?: string;
  prayer_intensity?: number;
  devotion?: number;
}

export interface SimulationState {
  timestamp: string;
  total: number;
  audio: number;
  image: number;
  video: number;
  points: SimPoint[];
  entities?: EntityState[];
  echoes?: Echo[];
  presence_dynamics?: PresenceDynamics;
  myth?: MythSummary;
  world?: WorldSummary;
  projection?: UIProjectionBundle;
  perspective?: UIPerspective;
}

export type UIPerspective = "hybrid" | "causal-time" | "swimlanes";

export interface UIProjectionPerspectiveOption {
  id: UIPerspective;
  symbol: string;
  name: string;
  merge: string;
  description: string;
  default: boolean;
}

export interface UIProjectionFieldSchema {
  field: string;
  name: string;
  delta_keys: string[];
  interpretation: {
    en: string;
    ja: string;
  };
}

export interface UIProjectionFieldSnapshot {
  record: string;
  id: string;
  ts: number;
  vectors: Record<string, Record<string, string | number>>;
  applied_reiso: string[];
  merge_mode: string;
  ticks: string[];
}

export interface UIProjectionCoherence {
  record: string;
  id: string;
  ts: number;
  centroid: {
    x: number;
    y: number;
    z: number;
  };
  tension: number;
  drift: number;
  entropy: number;
  dominant_perspective: UIPerspective;
  perspective_score: number;
}

export interface UIProjectionElement {
  record: string;
  id: string;
  kind: string;
  title: string;
  binds_to: string[];
  field_bindings: Record<string, number>;
  presence?: string;
  tags?: string[];
  lane?: string;
  memory_scope?: string;
}

export interface UIProjectionElementState {
  record: string;
  element_id: string;
  ts: number;
  mass: number;
  priority: number;
  area: number;
  opacity: number;
  pulse: number;
  sources: string[];
  explain: {
    field_signal: number;
    presence_signal: number;
    queue_signal: number;
    causal_signal: number;
    dominant_field: string;
    dominant_level: number;
    field_bindings: Record<string, number>;
    reason_en: string;
    reason_ja: string;
    coherence_tension: number;
  };
}

export interface UIProjectionLayout {
  record: string;
  id: string;
  ts: number;
  perspective: UIPerspective;
  elements: string[];
  rects: Record<string, { x: number; y: number; w: number; h: number }>;
  states: UIProjectionElementState[];
  clamps: {
    record: string;
    min_area: number;
    max_area: number;
    max_pulse: number;
    decay_half_life: Record<string, number>;
  };
  notes?: string;
}

export interface UIProjectionChatSession {
  record: string;
  id: string;
  ts: number;
  presence: string;
  lens_element: string;
  field_bindings: Record<string, number>;
  memory_scope: string;
  tags: string[];
  status: string;
}

export interface UIProjectionBundle {
  record: string;
  contract: string;
  ts: number;
  perspective: UIPerspective;
  default_perspective: UIPerspective;
  perspectives: UIProjectionPerspectiveOption[];
  field_schemas: UIProjectionFieldSchema[];
  field_snapshot: UIProjectionFieldSnapshot;
  coherence: UIProjectionCoherence;
  elements: UIProjectionElement[];
  states: UIProjectionElementState[];
  layout: UIProjectionLayout;
  chat_sessions: UIProjectionChatSession[];
  vector_view: {
    record: string;
    id: string;
    ts: number;
    mode: string;
    axes: Record<string, string>;
    show_causality: boolean;
  };
  tick_view: {
    record: string;
    id: string;
    ts: number;
    sources: string[];
    window: Record<string, number>;
    show_causal: boolean;
    merge: string;
  };
  queue: {
    pending_count: number;
    event_count: number;
  };
}

export interface PresenceImpact {
  id: string;
  en: string;
  ja: string;
  affected_by: {
    files: number;
    clicks: number;
  };
  affects: {
    world: number;
    ledger: number;
  };
  notes_en: string;
  notes_ja: string;
}

export interface ForkTaxState {
  law_en: string;
  law_ja: string;
  debt: number;
  paid: number;
  balance: number;
  paid_ratio: number;
}

export interface GhostRoleState {
  id: string;
  en: string;
  ja: string;
  auto_commit_pulse: number;
  queue_pending: number;
  actions_60s?: number;
  status_en: string;
  status_ja: string;
}

export interface PresenceDynamics {
  generated_at: string;
  click_events: number;
  file_events: number;
  recent_click_targets: string[];
  recent_file_paths: string[];
  river_flow: {
    unit: string;
    rate: number;
    turbulence: number;
  };
  ghost: GhostRoleState;
  fork_tax: ForkTaxState;
  witness_thread?: WitnessThreadState;
  presence_impacts: PresenceImpact[];
}

export interface WitnessThreadLineageEntry {
  kind: string;
  ref: string;
  why_en: string;
  why_ja: string;
}

export interface WitnessThreadState {
  id: string;
  en: string;
  ja: string;
  continuity_index: number;
  click_pressure: number;
  file_pressure: number;
  linked_presences: string[];
  lineage: WitnessThreadLineageEntry[];
  notes_en: string;
  notes_ja: string;
}

export interface MixMeta {
  sources: number;
  sample_rate: number;
  duration_seconds: number;
  fingerprint?: string;
}

export interface Catalog {
  generated_at: string;
  part_roots: string[];
  counts: Record<string, number>;
  canonical_terms: Array<{ en: string; ja: string }>;
  ui_default_perspective?: UIPerspective;
  ui_perspectives?: UIProjectionPerspectiveOption[];
  ui_projection?: UIProjectionBundle;
  entity_manifest?: Array<any>;
  cover_fields: Array<{
    id: string;
    part: string;
    display_name: { en: string; ja: string };
    display_role: { en: string; ja: string };
    url: string;
    seed: string;
  }>;
  task_queue?: {
    queue_log: string;
    pending_count: number;
    dedupe_keys: number;
    event_count: number;
  };
  presence_runtime?: {
    generated_at: string;
    clicks_45s: number;
    file_changes_120s: number;
    recent_click_targets: string[];
    recent_file_paths: string[];
    fork_tax?: ForkTaxState;
    ghost?: GhostRoleState;
  };
  items: CatalogItem[];
}

export type VoiceDeliveryMode = "whispered" | "spoken" | "canticle";

export interface InstrumentState {
  masterLevel: number;
  pulseLevel: number;
  artifactLevel: number;
  transportRate: number;
  voiceRate: number;
  voicePitch: number;
  voiceGain: number;
  delivery: VoiceDeliveryMode;
}

export interface InstrumentPad {
  id: string;
  key: string;
  note: string;
  labelEn: string;
  labelJa: string;
  freqHz: number;
}

export interface VoiceLine {
  id: string;
  en: string;
  ja: string;
  line_en: string;
  line_ja: string;
}

export interface VoicePack {
  mode: string;
  model: string | null;
  lines: VoiceLine[];
  generated_at: string;
}

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  text: string;
}

export type PresenceRole =
  | "arranger"
  | "lyricist"
  | "sound-designer"
  | "critic"
  | "conductor"
  | "vocal";

export type DecisionType = "accept" | "reject" | "revise" | "merge";
export type ArtifactKind = "audio" | "midi" | "lyrics" | "notes" | "render" | "project" | "preset";
export type MusicTrackType = "drums" | "bass" | "pad" | "lead" | "vox" | "fx";
export type ClipContentType = "audio" | "midi" | "text";
export type UtteranceClass = "eta-claim" | "mu-proof" | "neutral";

export interface PresenceMusicPresence {
  id: string;
  name: string;
  role: PresenceRole;
  style: Record<string, string | number | boolean>;
  voice: Record<string, string | number | boolean>;
  capabilities: string[];
}

export interface PresenceMusicArtifact {
  id: string;
  kind: ArtifactKind;
  path: string;
  hash?: string;
  meta: Record<string, string | number | boolean>;
}

export interface PresenceMusicSession {
  id: string;
  title: string;
  tempoBpm: number;
  timeSignature: string;
  key: string;
  moodTags: string[];
  loopId: string;
  loopBars: number;
  rangeSeconds: number;
  policyRef?: PresenceMusicArtifact;
}

export interface PresenceMusicUtterance {
  id: string;
  ts: string;
  speaker: string;
  text: string;
  class: UtteranceClass;
  reasons: string[];
  needs: string[];
  artifactRefs: string[];
}

export interface PresenceMusicDecision {
  id: string;
  ts: string;
  owner: string;
  type: DecisionType;
  target: string;
  rationale: string;
  evidence: string[];
}

export interface PresenceMusicPlan {
  id: string;
  owner: string;
  deadline?: string;
  definitionOfDone: string[];
  steps: string[];
  options: string[];
  links: string[];
}

export interface PresenceMusicTrack {
  id: string;
  name: string;
  type: MusicTrackType;
  presence: string;
  constraints: string[];
  artifacts: string[];
}

export interface PresenceMusicClip {
  id: string;
  trackId: string;
  bars: number;
  start: number;
  len: number;
  content: ClipContentType;
  artifact: PresenceMusicArtifact;
  tags: string[];
}

export interface PresenceMusicLyric {
  id: string;
  lang: "en" | "ja" | "mix";
  text: string;
  meter: Record<string, string | number>;
  artifact?: PresenceMusicArtifact;
}

export interface PresenceMusicLoop {
  id: string;
  round: number;
  prompt: string;
  inputs: string[];
  outputs: string[];
  votes: string[];
  ledgerRows: string[];
}

export interface EtaMuLedgerRow {
  ts: string;
  idx: number;
  utterance: string;
  classification: UtteranceClass;
  eta_claim: boolean;
  mu_proof: boolean;
  artifact_refs: string[];
}

export interface EtaMuLedgerResponse {
  ok: boolean;
  rows: EtaMuLedgerRow[];
  jsonl: string;
}
