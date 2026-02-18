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
  z?: number;
  size: number;
  r: number;
  g: number;
  b: number;
}

export interface BackendFieldParticle {
  id: string;
  presence_id: string;
  presence_role?: string;
  particle_mode?: "neutral" | "role-bound";
  x: number;
  y: number;
  size: number;
  r: number;
  g: number;
  b: number;
}

export interface FileGraphEmbeddingParticle {
  id: string;
  x: number;
  y: number;
  hue: number;
  cohesion: number;
  text_density: number;
}

export interface FileGraphInboxState {
  record: string;
  path: string;
  pending_count: number;
  processed_count: number;
  skipped_count?: number;
  failed_count: number;
  rejected_count?: number;
  deferred_count?: number;
  is_empty: boolean;
  knowledge_entries: number;
  registry_entries?: number;
  last_ingested_at: string;
  spaces?: Record<string, unknown>;
  stats?: Record<string, number>;
  packets?: {
    count: number;
    last: string;
  };
  artifacts?: Record<string, string>;
  errors: Array<{ path: string; error: string }>;
}

export interface FileGraphEmbedLayerPoint {
  id: string;
  key: string;
  label: string;
  collection: string;
  space_id: string;
  space_signature: string;
  model_name: string;
  x: number;
  y: number;
  hue: number;
  active: boolean;
  source: string;
  reference_count: number;
  embed_ids: string[];
}

export interface FileGraphEmbedLayerSummary {
  id: string;
  key: string;
  label: string;
  collection: string;
  space_id: string;
  space_signature: string;
  model_name: string;
  file_count: number;
  reference_count: number;
  active: boolean;
}

export interface FileGraphConceptPresence {
  id: string;
  cluster_id: string;
  label: string;
  label_ja: string;
  terms: string[];
  cohesion: number;
  file_count: number;
  members: string[];
  x: number;
  y: number;
  hue: number;
  created_by: string;
}

export interface FileGraphOrganizerPresence {
  id: string;
  node_id: string;
  node_type: "presence";
  presence_kind: "organizer";
  field: string;
  label: string;
  label_ja: string;
  x: number;
  y: number;
  hue: number;
  created_count: number;
}

export interface FileGraphNode {
  id: string;
  node_id: string;
  node_type: "field" | "file" | "presence" | "tag";
  field?: string;
  tag?: string;
  label: string;
  label_ja?: string;
  presence_kind?: "organizer" | "concept";
  name?: string;
  kind?: string;
  x: number;
  y: number;
  hue: number;
  importance?: number;
  source_rel_path?: string;
  archived_rel_path?: string;
  url?: string;
  dominant_field?: string;
  dominant_presence?: string;
  field_scores?: Record<string, number>;
  text_excerpt?: string;
  summary?: string;
  tags?: string[];
  labels?: string[];
  member_count?: number;
  embed_layer_points?: FileGraphEmbedLayerPoint[];
  embed_layer_count?: number;
  vecstore_collection?: string;
  concept_presence_id?: string;
  concept_presence_label?: string;
  organized_by?: string;
}

export interface FileGraphEdge {
  id: string;
  source: string;
  target: string;
  field: string;
  weight: number;
  kind: string;
}

export interface FileGraph {
  record: string;
  generated_at: string;
  inbox: FileGraphInboxState;
  nodes: FileGraphNode[];
  field_nodes: FileGraphNode[];
  tag_nodes?: FileGraphNode[];
  file_nodes: FileGraphNode[];
  embed_layers?: FileGraphEmbedLayerSummary[];
  organizer_presence?: FileGraphOrganizerPresence;
  concept_presences?: FileGraphConceptPresence[];
  embedding_particles?: FileGraphEmbeddingParticle[];
  edges: FileGraphEdge[];
  stats: {
    field_count: number;
    file_count: number;
    edge_count: number;
    kind_counts: Record<string, number>;
    field_counts: Record<string, number>;
    embed_layer_count?: number;
    embed_layer_active_count?: number;
    organizer_presence_count?: number;
    concept_presence_count?: number;
    organized_file_count?: number;
    tag_count?: number;
    tag_edge_count?: number;
    tag_pair_edge_count?: number;
    docmeta_enriched_count?: number;
    knowledge_entries: number;
  };
}

export interface CrawlerGraphNode {
  id: string;
  node_id: string;
  node_type: "field" | "crawler";
  field?: string;
  label: string;
  label_ja?: string;
  crawler_kind?: string;
  x: number;
  y: number;
  hue: number;
  importance?: number;
  url?: string;
  domain?: string;
  title?: string;
  status?: string;
  content_type?: string;
  compliance?: string;
  dominant_field?: string;
  dominant_presence?: string;
  field_scores?: Record<string, number>;
}

export interface CrawlerGraphEdge {
  id: string;
  source: string;
  target: string;
  field: string;
  weight: number;
  kind: string;
}

export interface CrawlerGraph {
  record: string;
  generated_at: string;
  source: {
    endpoint: string;
    service: string;
  };
  status: Record<string, unknown>;
  nodes: CrawlerGraphNode[];
  field_nodes: CrawlerGraphNode[];
  crawler_nodes: CrawlerGraphNode[];
  edges: CrawlerGraphEdge[];
  stats: {
    field_count: number;
    crawler_count: number;
    edge_count: number;
    kind_counts: Record<string, number>;
    field_counts: Record<string, number>;
    nodes_total: number;
    edges_total: number;
    url_nodes_total: number;
  };
}

export interface TruthStateClaim {
  id: string;
  text: string;
  status: "proved" | "refuted" | "undecided";
  kappa: number;
  world: string;
  proof_refs: string[];
  theta: number;
}

export interface TruthStateProofEntry {
  kind: string;
  ref: string;
  present: boolean;
  detail?: string;
}

export interface TruthState {
  record: string;
  generated_at: string;
  name_binding: {
    id: string;
    symbol: string;
    glyph: string;
    ascii: string;
    law: string;
  };
  world: {
    id: string;
    "ctx/ω-world": string;
    ctx_omega_world: string;
  };
  claim: TruthStateClaim;
  claims: TruthStateClaim[];
  guard: {
    theta: number;
    passes: boolean;
  };
  gate: {
    target: string;
    blocked: boolean;
    reasons: string[];
  };
  invariants: Record<string, boolean>;
  proof: {
    required_kinds: string[];
    entries: TruthStateProofEntry[];
    counts: {
      total: number;
      present: number;
      by_kind: Record<string, number>;
    };
  };
  artifacts: {
    pi_zip_count: number;
    host_handle: string;
    host_has_github_gist: boolean;
    truth_receipt_count: number;
    decision_receipt_count?: number;
  };
  schema: {
    source: string;
    required_refs: string[];
    required_hashes: string[];
    host_handle: string;
    missing_refs: string[];
    missing_hashes: string[];
  };
  needs: string[];
}

export interface LogicalGraphNode {
  id: string;
  kind: "file" | "fact" | "rule" | "derivation" | "contradiction" | "gate" | string;
  label: string;
  status?: string;
  confidence?: number;
  x: number;
  y: number;
  file_id?: string;
  source_uri?: string;
  path?: string;
  provenance?: Record<string, unknown>;
}

export interface LogicalGraphEdge {
  id: string;
  source: string;
  target: string;
  kind: string;
  weight: number;
}

export interface LogicalGraph {
  record: string;
  generated_at: string;
  nodes: LogicalGraphNode[];
  edges: LogicalGraphEdge[];
  joins: {
    file_ids: string[];
    file_index: Record<string, string>;
    source_to_file: Record<string, string>;
  };
  stats: {
    file_nodes: number;
    tag_nodes?: number;
    fact_nodes: number;
    rule_nodes: number;
    derivation_nodes: number;
    contradiction_nodes: number;
    gate_nodes: number;
    tag_edges?: number;
    edge_count: number;
  };
}

export interface PainFieldTestFailure {
  id: string;
  name: string;
  status: string;
  message: string;
  severity: number;
  covered_files: string[];
  file_ids: string[];
}

export interface PainFieldNodeHeat {
  node_id: string;
  kind: string;
  heat: number;
  x: number;
  y: number;
  file_id?: string;
  label?: string;
}

export interface PainFieldDebugTarget {
  meaning: string;
  glyph: string;
  grounded: boolean;
  source: string;
  node_id: string;
  file_id: string;
  region_id: string;
  path: string;
  label: string;
  heat: number;
  x: number;
  y: number;
  reason: string;
}

export interface PainField {
  record: string;
  generated_at: string;
  active: boolean;
  decay: number;
  hops: number;
  failing_tests: PainFieldTestFailure[];
  node_heat: PainFieldNodeHeat[];
  debug?: PainFieldDebugTarget;
  grounded_meanings?: Record<string, PainFieldDebugTarget>;
  max_heat: number;
  join_key: string;
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
  field_particles?: BackendFieldParticle[];
  file_graph?: FileGraph;
  crawler_graph?: CrawlerGraph;
  truth_state?: TruthState;
  logical_graph?: LogicalGraph;
  pain_field?: PainField;
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
    resource?: number;
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

export interface ResourceDeviceSnapshot {
  utilization: number;
  status: string;
  memory?: number;
  temperature?: number;
  queue_depth?: number;
  device?: string;
  load_avg?: {
    m1: number;
    m5: number;
    m15: number;
  };
  memory_pressure?: number;
}

export interface ResourceHeartbeatSnapshot {
  record: string;
  generated_at: string;
  window_seconds?: number;
  devices: {
    cpu?: ResourceDeviceSnapshot;
    gpu1?: ResourceDeviceSnapshot;
    gpu2?: ResourceDeviceSnapshot;
    npu0?: ResourceDeviceSnapshot;
    [key: string]: ResourceDeviceSnapshot | undefined;
  };
  hot_devices?: string[];
  auto_backend?: {
    embeddings_order?: string[];
    text_order?: string[];
  };
  log_watch?: {
    path?: string;
    line_count?: number;
    error_count?: number;
    warn_count?: number;
    error_ratio?: number;
    warn_ratio?: number;
    latest?: string;
  };
}

export interface PresenceDynamics {
  generated_at: string;
  click_events: number;
  file_events: number;
  log_events_180s?: number;
  resource_events_180s?: number;
  compute_jobs_180s?: number;
  compute_summary?: {
    llm_jobs: number;
    embedding_jobs: number;
    ok_count: number;
    error_count: number;
    resource_counts: Record<string, number>;
  };
  compute_jobs?: Array<{
    id: string;
    at: string;
    ts: number;
    kind: string;
    op: string;
    backend: string;
    resource: string;
    emitter_presence_id: string;
    target_presence_id: string;
    model: string;
    status: string;
    latency_ms?: number | null;
    error?: string;
  }>;
  field_particles_record?: string;
  field_particles?: BackendFieldParticle[];
  simulation_budget?: {
    point_limit: number;
    point_limit_max: number;
    cpu_utilization: number;
  };
  recent_click_targets: string[];
  recent_file_paths: string[];
  recent_logs?: Array<{
    level: string;
    source: string;
    message: string;
  }>;
  last_log?: {
    level: string;
    source: string;
    message: string;
  };
  log_summary?: {
    event_count: number;
    error_count: number;
    warn_count: number;
  };
  resource_heartbeat?: ResourceHeartbeatSnapshot;
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

export interface TaskQueueTask {
  id: string;
  kind: string;
  dedupe_key?: string;
  payload?: Record<string, unknown>;
  created_at?: string;
  owner?: string;
}

export interface TaskQueueSnapshot {
  queue_log: string;
  pending_count: number;
  dedupe_keys: number;
  event_count: number;
  pending?: TaskQueueTask[];
}

export interface CouncilVote {
  member_id: string;
  vote: "yes" | "no" | "abstain" | string;
  reason: string;
  mode?: string;
  actor?: string;
  ts: string;
}

export interface CouncilTally {
  yes: number;
  no: number;
  abstain: number;
  required_yes: number;
  approved: boolean;
}

export interface CouncilDecision {
  id: string;
  kind: string;
  status: string;
  created_at: string;
  created_unix: number;
  source_event?: {
    type: string;
    data?: Record<string, unknown>;
  };
  resource?: {
    source_rel_path?: string;
    field?: string;
    node_id?: string;
  };
  council?: {
    id: string;
    members: string[];
    required_yes: number;
    votes: CouncilVote[];
    tally: CouncilTally;
  };
  gate?: {
    blocked: boolean;
    reasons: string[];
  };
  action?: {
    attempted: boolean;
    ok: boolean;
    result: string;
    reason?: string;
    unix_ts?: number;
    services?: string[];
  };
}

export interface CouncilSnapshot {
  decision_log: string;
  event_count: number;
  decision_count: number;
  pending_count: number;
  approved_count: number;
  auto_restart_enabled: boolean;
  require_council: boolean;
  cooldown_seconds: number;
  decisions?: CouncilDecision[];
}

export interface CouncilApiResponse {
  ok: boolean;
  council: CouncilSnapshot;
}

export interface DriftGateBlock {
  target: string;
  reason: string;
  question_ids?: string[];
}

export interface DriftScanPayload {
  ok: boolean;
  generated_at: string;
  active_drifts: Array<{
    id: string;
    severity: string;
    detail: string;
    question_ids?: string[];
  }>;
  blocked_gates: DriftGateBlock[];
  receipts?: {
    path: string;
    parse_ok: boolean;
    rows: number;
    has_intent_ref: boolean;
  };
  open_questions?: {
    total: number;
    resolved_count: number;
    unresolved_count: number;
  };
  receipts_parse?: {
    path: string;
    ok: boolean;
    rows: number;
    has_intent_ref: boolean;
  };
}

export interface StudyWarning {
  code: string;
  severity: string;
  message: string;
}

export interface StudyStability {
  score: number;
  label: string;
  components: {
    blocked_gate_penalty: number;
    drift_penalty: number;
    queue_penalty: number;
    council_penalty: number;
    truth_penalty: number;
    resource_penalty?: number;
    resource_log_penalty?: number;
  };
}

export interface StudySignals {
  blocked_gate_count: number;
  active_drift_count: number;
  queue_pending_count: number;
  queue_event_count: number;
  council_pending_count: number;
  council_approved_count: number;
  council_decision_count: number;
  decision_status_counts: Record<string, number>;
  truth_gate_blocked: boolean;
  open_questions_unresolved: number;
  resource_hot_count?: number;
  resource_log_error_ratio?: number;
}

export interface StudyRuntime {
  part_root: string;
  vault_root: string;
  receipts_path: string;
  receipts_parse_ok: boolean;
  receipts_rows: number;
  receipts_has_intent_ref: boolean;
  receipts_path_within_vault: boolean;
  resource?: ResourceHeartbeatSnapshot;
}

export interface StudySnapshotPayload {
  ok: boolean;
  record: string;
  generated_at: string;
  stability: StudyStability;
  signals: StudySignals;
  runtime: StudyRuntime;
  warnings: StudyWarning[];
  drift: DriftScanPayload;
  queue: TaskQueueSnapshot;
  council: CouncilSnapshot;
}

export interface EntityManifestItem {
  id: string;
  en: string;
  ja: string;
  hue: number;
  type?: string;
  x?: number;
  y?: number;
  freq?: number;
  flavor_vitals?: Record<string, unknown>;
}

export interface NamedFieldItem {
  id: string;
  en: string;
  ja: string;
  type?: string;
  x?: number;
  y?: number;
  freq?: number;
  hue?: number;
}

export interface WorldLogEventRelation {
  event_id: string;
  node_id?: string;
  score: number;
  kind: string;
}

export interface WorldLogEvent {
  id: string;
  ts: string;
  source: string;
  kind: string;
  status: string;
  title: string;
  detail: string;
  refs: string[];
  tags: string[];
  path?: string;
  embedding_id?: string;
  node_id?: string;
  x?: number;
  y?: number;
  dominant_field?: string;
  dominant_presence?: string;
  dominant_weight?: number;
  relations?: WorldLogEventRelation[];
}

export interface WorldLogPayload {
  ok: boolean;
  record: string;
  generated_at: string;
  count: number;
  limit: number;
  pending_inbox: number;
  sources: Record<string, number>;
  kinds: Record<string, number>;
  relation_count: number;
  events: WorldLogEvent[];
}

export interface Catalog {
  generated_at: string;
  part_roots: string[];
  counts: Record<string, number>;
  canonical_terms: Array<{ en: string; ja: string }>;
  eta_mu_inbox?: FileGraphInboxState;
  file_graph?: FileGraph;
  crawler_graph?: CrawlerGraph;
  truth_state?: TruthState;
  logical_graph?: LogicalGraph;
  pain_field?: PainField;
  test_failures?: Array<Record<string, unknown>>;
  test_coverage?: Record<string, unknown>;
  ui_default_perspective?: UIPerspective;
  ui_perspectives?: UIProjectionPerspectiveOption[];
  ui_projection?: UIProjectionBundle;
  entity_manifest?: EntityManifestItem[];
  named_fields?: NamedFieldItem[];
  world_log?: WorldLogPayload;
  cover_fields: Array<{
    id: string;
    part: string;
    display_name: { en: string; ja: string };
    display_role: { en: string; ja: string };
    url: string;
    seed: string;
  }>;
  task_queue?: TaskQueueSnapshot;
  council?: CouncilSnapshot;
  presence_runtime?: {
    generated_at: string;
    clicks_45s: number;
    file_changes_120s: number;
    log_events_180s?: number;
    resource_events_180s?: number;
    compute_jobs_180s?: number;
    compute_summary?: {
      llm_jobs: number;
      embedding_jobs: number;
      ok_count: number;
      error_count: number;
      resource_counts: Record<string, number>;
    };
    compute_jobs?: Array<{
      id: string;
      at: string;
      ts: number;
      kind: string;
      op: string;
      backend: string;
      resource: string;
      emitter_presence_id: string;
      target_presence_id: string;
      model: string;
      status: string;
      latency_ms?: number | null;
      error?: string;
    }>;
    recent_click_targets: string[];
    recent_file_paths: string[];
    recent_logs?: Array<{
      level: string;
      source: string;
      message: string;
    }>;
    last_log?: {
      level: string;
      source: string;
      message: string;
    };
    log_summary?: {
      event_count: number;
      error_count: number;
      warn_count: number;
    };
    resource_heartbeat?: ResourceHeartbeatSnapshot;
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
