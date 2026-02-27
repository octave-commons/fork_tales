export type AutopilotGate = "confidence" | "risk" | "permission" | "health" | "unknown";
export type AskUrgency = "low" | "med" | "high";

export type AutopilotActionResult = {
  ok: boolean;
  summary: string;
  meta?: Record<string, unknown>;
};

export type IntentHypothesis = {
  goal: string;
  confidence: number;
  alternatives?: Array<{ goal: string; confidence: number }>;
  rationale?: string;
};

export type PlannedAction<Context> = {
  id: string;
  label: string;
  goal: string;
  risk: number;
  cost: number;
  requiredPerms: string[];
  run: () => Promise<AutopilotActionResult>;
  verify?: (
    before: Context,
    after: Context,
    result: AutopilotActionResult,
  ) => boolean | Promise<boolean>;
};

export type GateVerdict<Context> =
  | { ok: true; action: PlannedAction<Context> }
  | { ok: false; ask: AskPayload };

export interface AutopilotActionEvent {
  ts: string;
  actionId: string;
  intent: string;
  confidence: number;
  risk: number;
  perms: string[];
  result: "ok" | "failed" | "skipped";
  summary: string;
  gate?: AutopilotGate;
}

export interface AutopilotHooks<Context> {
  sense: () => Promise<Context>;
  hypothesize: (ctx: Context) => Promise<IntentHypothesis>;
  plan: (ctx: Context, goal: string) => Promise<PlannedAction<Context>>;
  gate: (ctx: Context, hyp: IntentHypothesis, action: PlannedAction<Context>) => GateVerdict<Context>;
  onActionEvent?: (event: AutopilotActionEvent) => void;
  onAsk?: (ask: AskPayload) => void;
  onTickError?: (error: unknown) => void;
  tickDelayMs?: number;
}

export interface AskPayload {
  reason: string;
  need: string;
  options?: string[];
  context?: Record<string, unknown>;
  urgency?: AskUrgency;
  gate?: AutopilotGate;
}

const CHAT_PANEL_SELECTOR = "#chat-panel";
const CHAT_INPUT_SELECTOR = "#chat-input";

export function requestUserInput(payload: AskPayload): void {
  const chatEl = document.querySelector<HTMLElement>(CHAT_PANEL_SELECTOR);
  chatEl?.scrollIntoView({ behavior: "smooth", block: "end" });

  const input = document.querySelector<HTMLTextAreaElement>(CHAT_INPUT_SELECTOR);
  input?.focus();

  window.dispatchEvent(new CustomEvent<AskPayload>("autopilot:ask", { detail: payload }));

  window.dispatchEvent(
    new CustomEvent("ui:toast", {
      detail: { title: "Need your input", body: payload.need },
    }),
  );
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

export class Autopilot<Context> {
  private readonly hooks: AutopilotHooks<Context>;
  private running = false;
  private waitingForInput = false;
  private generation = 0;

  constructor(hooks: AutopilotHooks<Context>) {
    this.hooks = hooks;
  }

  start(): void {
    if (this.running) {
      return;
    }
    this.running = true;
    this.generation += 1;
    const currentGeneration = this.generation;
    void this.loop(currentGeneration);
  }

  stop(): void {
    this.running = false;
    this.waitingForInput = false;
  }

  resume(): void {
    this.waitingForInput = false;
    this.start();
  }

  isRunning(): boolean {
    return this.running;
  }

  isWaitingForInput(): boolean {
    return this.waitingForInput;
  }

  private async loop(generation: number): Promise<void> {
    const tickDelayMs = Math.max(250, this.hooks.tickDelayMs ?? 1200);

    while (this.running && generation === this.generation) {
      try {
        const before = await this.hooks.sense();
        const hypothesis = await this.hooks.hypothesize(before);
        const action = await this.hooks.plan(before, hypothesis.goal);
        const verdict = this.hooks.gate(before, hypothesis, action);

        if (!verdict.ok) {
          this.running = false;
          this.waitingForInput = true;
          this.hooks.onAsk?.(verdict.ask);
          this.hooks.onActionEvent?.({
            ts: new Date().toISOString(),
            actionId: action.id,
            intent: hypothesis.goal,
            confidence: hypothesis.confidence,
            risk: action.risk,
            perms: action.requiredPerms,
            result: "skipped",
            summary: verdict.ask.need,
            ...(verdict.ask.gate ? { gate: verdict.ask.gate } : {}),
          });
          requestUserInput(verdict.ask);
          return;
        }

        const result = await verdict.action.run();
        const after = await this.hooks.sense();
        let verified = result.ok;
        if (verdict.action.verify) {
          verified = await verdict.action.verify(before, after, result);
        }

        this.hooks.onActionEvent?.({
          ts: new Date().toISOString(),
          actionId: verdict.action.id,
          intent: hypothesis.goal,
          confidence: hypothesis.confidence,
          risk: verdict.action.risk,
          perms: verdict.action.requiredPerms,
          result: verified ? "ok" : "failed",
          summary: result.summary,
        });
      } catch (error) {
        this.hooks.onTickError?.(error);
      }

      await sleep(tickDelayMs);
    }
  }
}
