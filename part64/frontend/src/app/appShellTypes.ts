import type { PanelConfig } from "./worldPanelLayout";

export interface OverlayApi {
  pulseAt?: (x: number, y: number, power: number, target?: string) => void;
  singAll?: () => void;
  getAnchorRatio?: (kind: string, targetId: string) => { x: number; y: number; kind: string; label?: string } | null;
  projectRatioToClient?: (xRatio: number, yRatio: number) => { x: number; y: number; w: number; h: number };
  interactAt?: (
    xRatio: number,
    yRatio: number,
    options?: { openWorldscreen?: boolean },
  ) => { hitNode: boolean; openedWorldscreen: boolean; target: string; xRatio: number; yRatio: number };
  interactClientAt?: (
    clientX: number,
    clientY: number,
    options?: { openWorldscreen?: boolean },
  ) => { hitNode: boolean; openedWorldscreen: boolean; target: string; xRatio: number; yRatio: number };
}

export interface UiToast {
  id: number;
  title: string;
  body: string;
}

export interface UserPresenceInputPayload {
  kind: string;
  target: string;
  message?: string;
  xRatio?: number;
  yRatio?: number;
  embedDaimoi?: boolean;
  meta?: Record<string, unknown>;
}

export type ParticleDisposition = "neutral" | "role-bound";

export interface RankedPanel extends PanelConfig {
  priority: number;
  depth: number;
  councilScore: number;
  councilBoost: number;
  councilReason: string;
  presenceId: string;
  presenceLabel: string;
  presenceLabelJa: string;
  presenceRole: string;
  particleDisposition: ParticleDisposition;
  particleCount: number;
  toolHints: string[];
}
