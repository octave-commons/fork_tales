import { useState, useRef, useEffect, useCallback } from "react";
import { Send, Mic, FileAudio, MessageSquare } from "lucide-react";
import type { ChatMessage } from "../../types";
import type { AskPayload } from "../../autopilot";

interface Props {
  onSend: (text: string) => void;
  onRecord: () => void;
  onTranscribe: () => void;
  onSendVoice: () => void;
  isRecording: boolean;
  isThinking: boolean;
  voiceInputMeta: string;
}

const AUTOPILOT_OPTION_LIMIT = 5;

function formatContextValue(value: unknown): string {
  if (value === null || value === undefined) {
    return String(value);
  }
  if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  try {
    return JSON.stringify(value);
  } catch {
    return "[unserializable]";
  }
}

function normalizeAskPayload(payload: AskPayload): AskPayload {
  const reason = String(payload.reason || "autopilot is blocked").trim() || "autopilot is blocked";
  const need = String(payload.need || "reply with a decision").trim();
  const options = (payload.options || [])
    .map((option) => option.trim())
    .filter((option, index, all) => option.length > 0 && all.indexOf(option) === index)
    .slice(0, AUTOPILOT_OPTION_LIMIT);

  return {
    reason,
    need,
    options,
    context: payload.context,
    urgency: payload.urgency,
    gate: payload.gate || "unknown",
  };
}

function askSystemMessage(payload: AskPayload): string {
  const gate = payload.gate || "unknown";
  const urgency = payload.urgency ? ` (${payload.urgency})` : "";
  return `autopilot blocked [${gate}]${urgency}\n${payload.reason}\nneed: ${payload.need}`;
}

export function ChatPanel({
  onSend,
  onRecord,
  onTranscribe,
  onSendVoice,
  isRecording,
  isThinking,
  voiceInputMeta
}: Props) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [pendingAsk, setPendingAsk] = useState<AskPayload | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const sendUserMessage = useCallback(
    (text: string): boolean => {
      const trimmed = text.trim();
      if (!trimmed || isThinking) {
        return false;
      }
      onSend(trimmed);
      setMessages((prev) => [...prev, { role: "user", text: trimmed }]);
      setInput("");
      setPendingAsk(null);
      return true;
    },
    [isThinking, onSend],
  );

  const handleSend = () => {
    sendUserMessage(input);
  };

  const handleAskOption = (option: string) => {
    if (sendUserMessage(option)) {
      return;
    }

    setInput(option);
    window.requestAnimationFrame(() => {
      if (!inputRef.current) {
        return;
      }
      inputRef.current.focus();
      const cursor = option.length;
      inputRef.current.setSelectionRange(cursor, cursor);
    });
  };

  useEffect(() => {
    if(messages.length >= 0 && scrollRef.current) {
        scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  // Hook for external messages
  useEffect(() => {
    const handler: EventListener = (event) => {
      const customEvent = event as CustomEvent<ChatMessage>;
      if (!customEvent.detail) {
        return;
      }
      setMessages((prev) => [...prev, customEvent.detail]);
    };
    window.addEventListener("chat-message", handler);
    return () => window.removeEventListener("chat-message", handler);
  }, []);

  useEffect(() => {
    const handler: EventListener = (event) => {
      const customEvent = event as CustomEvent<AskPayload>;
      if (!customEvent.detail) {
        return;
      }
      const normalized = normalizeAskPayload(customEvent.detail);
      if (!normalized.need) {
        return;
      }

      setPendingAsk(normalized);
      setMessages((prev) => [...prev, { role: "system", text: askSystemMessage(normalized) }]);

      window.requestAnimationFrame(() => {
        inputRef.current?.focus();
      });
    };

    window.addEventListener("autopilot:ask", handler);
    return () => window.removeEventListener("autopilot:ask", handler);
  }, []);

  return (
    <div
      id="chat-panel"
      className="mt-3 border border-[var(--line)] rounded-xl bg-gradient-to-b from-[rgba(45,46,39,0.92)] to-[rgba(31,32,29,0.94)] p-3"
    >
      <p className="font-semibold mb-2 text-sm">Live Chat / 対話チャット</p>
      
      <div 
        ref={scrollRef}
        className="border border-[var(--line)] rounded-lg bg-[rgba(31,32,29,0.86)] p-2 min-h-[130px] max-h-[240px] overflow-auto grid gap-2"
      >
        {messages.map((msg, i) => (
          <div 
            key={`${i}-${msg.role}`} 
            className={`
              border border-[var(--line)] rounded-lg p-2 text-sm whitespace-pre-wrap leading-relaxed
              ${msg.role === 'user' ? 'bg-[rgba(102,217,239,0.16)]' : 'bg-[rgba(45,46,39,0.9)]'}
            `}
          >
            <span className="font-bold text-xs opacity-70 block mb-1">
              {msg.role === 'user' ? 'you / あなた' : msg.role === 'assistant' ? 'world / 世界' : 'system'}
            </span>
            {msg.text}
          </div>
        ))}
        {isThinking && (
            <div className="border border-dashed border-[rgba(102,217,239,0.44)] rounded-lg p-2 text-sm bg-[rgba(102,217,239,0.12)] animate-pulse text-[#66d9ef]">
                world is thinking... / 世界が思考中...
            </div>
        )}
      </div>

      {pendingAsk ? (
        <div className="mt-2 border border-[rgba(249,38,114,0.5)] rounded-lg bg-[rgba(249,38,114,0.1)] p-2">
          <p className="text-xs font-semibold text-[#f92672]">Autopilot Request / 自動操縦の質問</p>
          <p className="text-[11px] text-muted mt-1">
            gate: <code>{pendingAsk.gate || "unknown"}</code>
            {pendingAsk.urgency ? (
              <>
                {" "}| urgency: <code>{pendingAsk.urgency}</code>
              </>
            ) : null}
          </p>
          <p className="text-xs text-muted mt-1">{pendingAsk.reason}</p>
          <p className="text-sm text-ink mt-1">{pendingAsk.need}</p>
          {pendingAsk.context ? (
            <p className="text-[11px] text-muted mt-1 font-mono">
              {Object.entries(pendingAsk.context)
                .slice(0, 3)
                .map(([key, value]) => `${key}=${formatContextValue(value)}`)
                .join(" | ")}
            </p>
          ) : null}
          {pendingAsk.options && pendingAsk.options.length > 0 ? (
            <div className="mt-2 flex flex-wrap gap-2">
              {pendingAsk.options.map((option) => (
                <button
                  key={option}
                  type="button"
                  onClick={() => handleAskOption(option)}
                  className="text-xs border border-[rgba(249,38,114,0.55)] rounded-md px-2 py-1 bg-[rgba(249,38,114,0.16)] hover:bg-[rgba(249,38,114,0.24)] transition-colors"
                >
                  {option}
                </button>
              ))}
            </div>
          ) : null}
        </div>
      ) : null}

      <div className="grid grid-cols-[1fr_auto] gap-2 mt-2">
        <textarea
          id="chat-input"
          ref={inputRef}
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder={isThinking ? "Thinking... / 思考中..." : "speak or type... / 話すか入力してください"}
          disabled={isThinking}
          className="w-full min-h-[74px] max-h-[180px] resize-y border border-[var(--line)] rounded-lg p-2 font-inherit bg-[rgba(39,40,34,0.9)] text-ink disabled:opacity-50"
          onKeyDown={e => {
            if(e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSend();
            }
          }}
        />
        <button 
          type="button"
          onClick={handleSend}
          disabled={isThinking}
          className="self-end border border-[var(--line)] rounded-lg px-3 py-2 bg-[rgba(102,217,239,0.2)] hover:bg-[rgba(102,217,239,0.28)] transition-colors disabled:opacity-50"
        >
          <Send size={18} />
        </button>
      </div>

      <div className="flex gap-2 mt-2 flex-wrap">
        <button 
          type="button"
          onClick={onRecord}
          className={`flex items-center gap-2 btn-voice px-3 py-2 rounded-lg border border-[var(--line)] ${isRecording ? 'animate-pulse text-[#f92672]' : ''}`}
        >
          <Mic size={16} />
          <span className="text-xs">Record</span>
        </button>
        <button 
          type="button"
          onClick={onTranscribe}
          className="flex items-center gap-2 btn-voice px-3 py-2 rounded-lg border border-[var(--line)]"
        >
          <FileAudio size={16} />
          <span className="text-xs">Transcribe</span>
        </button>
        <button 
          type="button"
          onClick={onSendVoice}
          className="flex items-center gap-2 btn-voice px-3 py-2 rounded-lg border border-[var(--line)]"
        >
          <MessageSquare size={16} />
          <span className="text-xs">Send Voice</span>
        </button>
      </div>
      
      <p className="text-xs text-muted mt-1">{voiceInputMeta}</p>
      <p className="text-[10px] text-muted/80 mt-1">commands: <code>/ledger ...</code> <code>/say witness_thread hello</code> <code>/drift</code> <code>/push-truth --dry-run</code></p>
    </div>
  );
}
