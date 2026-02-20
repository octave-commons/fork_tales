import { ChatPanel } from "./Chat";
import type {
  Catalog,
  MuseWorkspaceContext,
  SimulationState,
  UIProjectionChatSession,
  UIProjectionElementState,
} from "../../types";

interface Props {
  museId: string;
  onSend: (text: string, musePresenceId: string, workspace: MuseWorkspaceContext) => void;
  onRecord: () => void;
  onTranscribe: () => void;
  onSendVoice: (musePresenceId: string, workspace: MuseWorkspaceContext) => void;
  isRecording: boolean;
  isThinking: boolean;
  voiceInputMeta: string;
  catalog: Catalog | null;
  simulation: SimulationState | null;
  workspaceContext?: MuseWorkspaceContext | null;
  onWorkspaceContextChange?: (musePresenceId: string, workspace: MuseWorkspaceContext) => void;
  onWorkspaceBindingsChange?: (musePresenceId: string, pinnedFileNodeIds: string[]) => void;
  chatLensState?: UIProjectionElementState | null;
  activeChatSession?: UIProjectionChatSession | null;
}

export function MusePresencePanel({
  museId,
  onSend,
  onRecord,
  onTranscribe,
  onSendVoice,
  isRecording,
  isThinking,
  voiceInputMeta,
  catalog,
  simulation,
  workspaceContext,
  onWorkspaceContextChange,
  onWorkspaceBindingsChange,
  chatLensState,
  activeChatSession,
}: Props) {
  return (
    <ChatPanel
      onSend={onSend}
      onRecord={onRecord}
      onTranscribe={onTranscribe}
      onSendVoice={onSendVoice}
      isRecording={isRecording}
      isThinking={isThinking}
      voiceInputMeta={voiceInputMeta}
      catalog={catalog}
      simulation={simulation}
      fixedMusePresenceId={museId}
      workspaceContext={workspaceContext}
      onWorkspaceContextChange={onWorkspaceContextChange}
      onWorkspaceBindingsChange={onWorkspaceBindingsChange}
      chatLensState={chatLensState}
      activeChatSession={activeChatSession}
      minimalMuseView
    />
  );
}
