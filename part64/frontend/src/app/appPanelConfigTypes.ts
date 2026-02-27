import { type AutopilotActionEvent } from "../autopilot";
import type {
  Catalog,
  MuseWorkspaceContext,
  SimulationState,
  UIProjectionBundle,
  UIProjectionElementState,
  WorldInteractionResponse,
} from "../types";
import { type UserPresenceInputPayload } from "./appShellTypes";
import { type CoreSimulationTuning } from "./coreSimulationConfig";
import { type WorldAnchorTarget } from "./worldPanelLayout";

export interface UseAppPanelConfigsArgs {
  activeMusePresenceId: string;
  activeProjection: UIProjectionBundle | null;
  autopilotEvents: AutopilotActionEvent[];
  catalog: Catalog | null;
  deferredCoreSimulationTuning: CoreSimulationTuning;
  deferredPanelsReady: boolean;
  flyCameraToAnchor: (anchor: WorldAnchorTarget) => void;
  handleMuseWorkspaceBindingsChange: (presenceId: string, fileNodeIds: string[]) => void;
  handleMuseWorkspaceContextChange: (presenceId: string, workspace: MuseWorkspaceContext) => void;
  handleMuseWorkspaceSend: (text: string, musePresenceId: string, workspace: MuseWorkspaceContext) => void;
  handleRecord: () => Promise<void>;
  handleSendVoice: (musePresenceId: string, workspace: MuseWorkspaceContext) => Promise<void>;
  handleTranscribe: () => Promise<string | undefined>;
  handleUserPresenceInput: (payload: UserPresenceInputPayload) => void;
  handleWorldInteract: (personId: string, action: "speak" | "pray" | "sing") => Promise<void>;
  interactingPersonId: string | null;
  isRecording: boolean;
  isThinking: boolean;
  museWorkspaceBindings: Record<string, string[]>;
  museWorkspaceContexts: Record<string, MuseWorkspaceContext>;
  projectionStateByElement: Map<string, UIProjectionElementState>;
  setActiveMusePresenceId: (presenceId: string) => void;
  simulation: SimulationState | null;
  voiceInputMeta: string;
  worldInteraction: WorldInteractionResponse | null;
}
