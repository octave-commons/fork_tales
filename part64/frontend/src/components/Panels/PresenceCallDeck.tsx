import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Phone, PhoneOff, Radio, SendHorizontal } from "lucide-react";
import { runtimeBaseUrl } from "../../runtime/endpoints";
import type { Catalog, SimulationState, WorldPresence } from "../../types";

type CallStatus = "idle" | "connecting" | "connected" | "error";
type TranscriptRole = "user" | "presence" | "system";

interface TranscriptEntry {
  id: string;
  role: TranscriptRole;
  text: string;
  presenceId?: string;
  ts: string;
}

interface AudioGraph {
  context: AudioContext;
  destination: MediaStreamAudioDestinationNode;
  mixSource: MediaElementAudioSourceNode;
  speechSource: MediaElementAudioSourceNode;
  mixGain: GainNode;
  speechGain: GainNode;
}

interface CallSession {
  outbound: RTCPeerConnection;
  inbound: RTCPeerConnection;
  remoteStream: MediaStream;
  presenceId: string;
}

interface Props {
  catalog: Catalog | null;
  simulation: SimulationState | null;
}

const FALLBACK_PRESENCES: WorldPresence[] = [
  {
    id: "witness_thread",
    name: { en: "Witness Thread", ja: "証人の糸" },
    type: "presence",
  },
  {
    id: "receipt_river",
    name: { en: "Receipt River", ja: "領収書の川" },
    type: "presence",
  },
  {
    id: "gates_of_truth",
    name: { en: "Gates of Truth", ja: "真理の門" },
    type: "presence",
  },
];

function formatCallStatus(status: CallStatus): string {
  if (status === "connected") {
    return "Connected";
  }
  if (status === "connecting") {
    return "Connecting";
  }
  if (status === "error") {
    return "Error";
  }
  return "Idle";
}

function statusToneClass(status: CallStatus): string {
  if (status === "connected") {
    return "text-[#a6e22e]";
  }
  if (status === "connecting") {
    return "text-[#66d9ef]";
  }
  if (status === "error") {
    return "text-[#f92672]";
  }
  return "text-muted";
}

function buildPresenceList(catalog: Catalog | null, simulation: SimulationState | null): WorldPresence[] {
  const worldPresences = simulation?.world?.presences ?? [];
  if (worldPresences.length > 0) {
    return worldPresences;
  }

  const entityManifest = Array.isArray(catalog?.entity_manifest) ? catalog.entity_manifest : [];
  if (entityManifest.length > 0) {
    return entityManifest
      .map((row, index) => {
        const id = String(row?.id ?? "").trim();
        if (!id) {
          return null;
        }
        const en = String(row?.en ?? `Presence ${index + 1}`);
        const ja = String(row?.ja ?? en);
        return {
          id,
          name: { en, ja },
          type: String(row?.type ?? "presence"),
        } satisfies WorldPresence;
      })
      .filter((row): row is WorldPresence => row !== null);
  }

  return FALLBACK_PRESENCES;
}

export function PresenceCallDeck({ catalog, simulation }: Props) {
  const [selectedPresenceId, setSelectedPresenceId] = useState<string>("");
  const [activePresenceId, setActivePresenceId] = useState<string | null>(null);
  const [callStatus, setCallStatus] = useState<CallStatus>("idle");
  const [callError, setCallError] = useState<string | null>(null);
  const [question, setQuestion] = useState("");
  const [isAsking, setIsAsking] = useState(false);
  const [transcript, setTranscript] = useState<TranscriptEntry[]>([]);

  const mixAudioRef = useRef<HTMLAudioElement | null>(null);
  const speechAudioRef = useRef<HTMLAudioElement | null>(null);
  const remoteAudioRef = useRef<HTMLAudioElement | null>(null);
  const audioGraphRef = useRef<AudioGraph | null>(null);
  const callSessionRef = useRef<CallSession | null>(null);
  const speechObjectUrlRef = useRef<string | null>(null);

  const presences = useMemo(() => buildPresenceList(catalog, simulation), [catalog, simulation]);

  const selectedPresence = useMemo(
    () => presences.find((presence) => presence.id === selectedPresenceId) ?? null,
    [presences, selectedPresenceId],
  );

  const appendTranscript = useCallback((role: TranscriptRole, text: string, presenceId?: string) => {
    setTranscript((prev) => {
      const entry: TranscriptEntry = {
        id: `${Date.now()}-${Math.random().toString(16).slice(2, 8)}`,
        role,
        text,
        presenceId,
        ts: new Date().toISOString(),
      };
      return [...prev.slice(-47), entry];
    });
  }, []);

  const ensureAudioGraph = useCallback(async (): Promise<AudioGraph> => {
    const cached = audioGraphRef.current;
    if (cached) {
      if (cached.context.state === "suspended") {
        await cached.context.resume();
      }
      return cached;
    }

    const mixElement = mixAudioRef.current;
    const speechElement = speechAudioRef.current;
    if (!mixElement || !speechElement) {
      throw new Error("Audio nodes are not ready");
    }

    mixElement.muted = true;
    speechElement.muted = true;

    const context = new AudioContext();
    const destination = context.createMediaStreamDestination();
    const mixSource = context.createMediaElementSource(mixElement);
    const speechSource = context.createMediaElementSource(speechElement);
    const mixGain = context.createGain();
    const speechGain = context.createGain();

    mixGain.gain.value = 0.82;
    speechGain.gain.value = 1;

    mixSource.connect(mixGain);
    speechSource.connect(speechGain);
    mixGain.connect(destination);
    speechGain.connect(destination);

    const graph: AudioGraph = {
      context,
      destination,
      mixSource,
      speechSource,
      mixGain,
      speechGain,
    };
    audioGraphRef.current = graph;
    return graph;
  }, []);

  const closeCallSession = useCallback(() => {
    const current = callSessionRef.current;
    if (!current) {
      return;
    }

    current.outbound.onicecandidate = null;
    current.inbound.onicecandidate = null;
    current.inbound.ontrack = null;

    current.outbound.getSenders().forEach((sender) => {
      try {
        current.outbound.removeTrack(sender);
      } catch {
        // ignore if sender is already detached
      }
    });
    current.remoteStream.getTracks().forEach((track) => {
      track.stop();
    });
    current.outbound.close();
    current.inbound.close();
    callSessionRef.current = null;

    const remoteAudio = remoteAudioRef.current;
    if (remoteAudio) {
      remoteAudio.pause();
      remoteAudio.srcObject = null;
    }
  }, []);

  const stopCall = useCallback(() => {
    closeCallSession();
    setCallStatus("idle");
    setActivePresenceId(null);
    setCallError(null);

    const mixElement = mixAudioRef.current;
    if (mixElement) {
      mixElement.pause();
    }

    appendTranscript("system", "Call ended.");
  }, [appendTranscript, closeCallSession]);

  const startCall = useCallback(async () => {
    if (!selectedPresence) {
      return;
    }

    setCallStatus("connecting");
    setCallError(null);

    closeCallSession();

    try {
      const graph = await ensureAudioGraph();
      const mixElement = mixAudioRef.current;
      if (!mixElement) {
        throw new Error("Missing mix source element");
      }

      mixElement.loop = true;
      if (!mixElement.src) {
        const base = runtimeBaseUrl();
        mixElement.src = base ? `${base}/stream/mix.wav` : "/stream/mix.wav";
      }
      try {
        await mixElement.play();
      } catch {
        appendTranscript(
          "system",
          "Mix stream did not start. Call stays up with spoken Presence replies only.",
          selectedPresence.id,
        );
      }

      const outbound = new RTCPeerConnection();
      const inbound = new RTCPeerConnection();
      const remoteStream = new MediaStream();

      outbound.onicecandidate = (event) => {
        if (!event.candidate) {
          return;
        }
        void inbound.addIceCandidate(event.candidate).catch(() => {
          return;
        });
      };

      inbound.onicecandidate = (event) => {
        if (!event.candidate) {
          return;
        }
        void outbound.addIceCandidate(event.candidate).catch(() => {
          return;
        });
      };

      inbound.ontrack = (event) => {
        if (event.streams.length > 0) {
          event.streams[0].getTracks().forEach((track) => {
            if (!remoteStream.getTrackById(track.id)) {
              remoteStream.addTrack(track);
            }
          });
        } else if (!remoteStream.getTrackById(event.track.id)) {
          remoteStream.addTrack(event.track);
        }

        const remoteAudio = remoteAudioRef.current;
        if (remoteAudio) {
          remoteAudio.srcObject = remoteStream;
          void remoteAudio.play().catch(() => {
            return;
          });
        }
      };

      graph.destination.stream.getAudioTracks().forEach((track) => {
        outbound.addTrack(track, graph.destination.stream);
      });

      const offer = await outbound.createOffer({
        offerToReceiveAudio: true,
        offerToReceiveVideo: false,
      });
      await outbound.setLocalDescription(offer);
      await inbound.setRemoteDescription(offer);

      const answer = await inbound.createAnswer();
      await inbound.setLocalDescription(answer);
      await outbound.setRemoteDescription(answer);

      callSessionRef.current = {
        outbound,
        inbound,
        remoteStream,
        presenceId: selectedPresence.id,
      };

      setCallStatus("connected");
      setActivePresenceId(selectedPresence.id);
      appendTranscript(
        "system",
        `Connected to ${selectedPresence.name.en}. Audio-first call is live (music + spoken words).`,
        selectedPresence.id,
      );
    } catch (error) {
      closeCallSession();
      setCallStatus("error");
      setActivePresenceId(null);
      const message = error instanceof Error ? error.message : "Failed to start call";
      setCallError(message);
      appendTranscript("system", `Call failed: ${message}`);
    }
  }, [appendTranscript, closeCallSession, ensureAudioGraph, selectedPresence]);

  const playPresenceSpeech = useCallback(async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed) {
      return;
    }

    const speechElement = speechAudioRef.current;
    if (!speechElement) {
      return;
    }

    if (speechObjectUrlRef.current) {
      URL.revokeObjectURL(speechObjectUrlRef.current);
      speechObjectUrlRef.current = null;
    }

    const base = runtimeBaseUrl();
    const response = await fetch(
      `${base}/api/tts?text=${encodeURIComponent(trimmed)}&speed=1.0`,
    );
    if (!response.ok) {
      throw new Error(`TTS failed (${response.status})`);
    }

    const blob = await response.blob();
    const objectUrl = URL.createObjectURL(blob);
    speechObjectUrlRef.current = objectUrl;

    speechElement.src = objectUrl;
    speechElement.currentTime = 0;
    speechElement.onended = () => {
      if (speechObjectUrlRef.current) {
        URL.revokeObjectURL(speechObjectUrlRef.current);
        speechObjectUrlRef.current = null;
      }
      speechElement.removeAttribute("src");
    };
    speechElement.onerror = () => {
      if (speechObjectUrlRef.current) {
        URL.revokeObjectURL(speechObjectUrlRef.current);
        speechObjectUrlRef.current = null;
      }
      speechElement.removeAttribute("src");
    };

    await speechElement.play();
  }, []);

  const askPresence = useCallback(async () => {
    const trimmed = question.trim();
    if (!selectedPresence || !trimmed || isAsking) {
      return;
    }

    setIsAsking(true);
    setCallError(null);
    appendTranscript("user", trimmed, selectedPresence.id);

    try {
      const base = runtimeBaseUrl();
      const response = await fetch(`${base}/api/presence/say`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          presence_id: selectedPresence.id,
          text: trimmed,
        }),
      });

      if (!response.ok) {
        throw new Error(`presence/say failed (${response.status})`);
      }

      const payload = (await response.json()) as {
        rendered_text?: string;
        presence_name?: { en?: string; ja?: string };
      };

      const renderedText = String(payload.rendered_text ?? "").trim() || "(no rendered text)";
      appendTranscript("presence", renderedText, selectedPresence.id);
      setQuestion("");

      if (callStatus === "connected" && callSessionRef.current?.presenceId === selectedPresence.id) {
        await playPresenceSpeech(renderedText);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : "Presence call failed";
      setCallError(message);
      appendTranscript("system", `Presence request failed: ${message}`, selectedPresence.id);
    } finally {
      setIsAsking(false);
    }
  }, [appendTranscript, callStatus, isAsking, playPresenceSpeech, question, selectedPresence]);

  useEffect(() => {
    if (presences.length === 0) {
      return;
    }
    if (!selectedPresenceId || !presences.some((presence) => presence.id === selectedPresenceId)) {
      setSelectedPresenceId(presences[0].id);
    }
  }, [presences, selectedPresenceId]);

  useEffect(() => {
    return () => {
      closeCallSession();
      if (speechObjectUrlRef.current) {
        URL.revokeObjectURL(speechObjectUrlRef.current);
        speechObjectUrlRef.current = null;
      }
      const graph = audioGraphRef.current;
      if (graph && graph.context.state !== "closed") {
        void graph.context.close();
      }
      audioGraphRef.current = null;
    };
  }, [closeCallSession]);

  return (
    <div className="card relative overflow-hidden !mt-0 h-full flex flex-col gap-4">
      <div className="absolute top-0 left-0 w-1 h-full bg-gradient-to-b from-[#66d9ef] to-transparent opacity-70" />
      
      <header className="border-b border-white/5 pb-4">
        <div className="flex justify-between items-start">
          <h2 className="text-2xl font-bold mb-1 text-ink tracking-tight">Presence Call Deck</h2>
          <div className="px-2 py-0.5 rounded bg-white/5 text-[10px] font-mono text-muted border border-white/5">
            WEBRTC LANE
          </div>
        </div>
        <p className="text-muted text-xs opacity-80 max-w-[90%]">
          Direct audio channel to simulated presences. Field music + spoken replies.
        </p>
      </header>

      <div className="space-y-4 flex-1">
        <div className="grid gap-4 md:grid-cols-[1fr_auto] md:items-end bg-[rgba(0,0,0,0.2)] p-3 rounded-xl border border-white/5">
          <label className="grid gap-1.5 text-xs font-semibold uppercase tracking-wider text-muted">
            Select Presence
            <select
              value={selectedPresenceId}
              onChange={(event) => setSelectedPresenceId(event.target.value)}
              className="rounded-lg border border-[var(--line)] bg-[rgba(20,20,18,0.95)] px-3 py-2.5 text-sm text-ink focus:ring-1 focus:ring-[#66d9ef] focus:border-[#66d9ef] outline-none transition-all"
            >
              {presences.map((presence) => (
                <option key={presence.id} value={presence.id}>
                  {presence.name.en}
                </option>
              ))}
            </select>
          </label>

          {callStatus === "connected" ? (
            <button
              type="button"
              onClick={stopCall}
              className="btn-base flex items-center justify-center gap-2 px-5 py-2.5 border-[rgba(249,38,114,0.45)] bg-[rgba(249,38,114,0.1)] text-[#f92672] hover:bg-[rgba(249,38,114,0.2)] hover:border-[rgba(249,38,114,0.8)] transition-all rounded-lg font-medium"
            >
              <PhoneOff size={16} />
              End Call
            </button>
          ) : (
            <button
              type="button"
              onClick={() => void startCall()}
              className="btn-base flex items-center justify-center gap-2 px-5 py-2.5 border-[rgba(102,217,239,0.62)] bg-[rgba(102,217,239,0.1)] text-[#66d9ef] hover:bg-[rgba(102,217,239,0.2)] hover:border-[#66d9ef] transition-all rounded-lg font-medium shadow-[0_0_15px_-5px_rgba(102,217,239,0.3)]"
            >
              <Phone size={16} />
              Start Call
            </button>
          )}
        </div>

        <section className={`rounded-xl border transition-all duration-300 ${
          callStatus === 'connected' 
            ? 'border-[#a6e22e]/30 bg-[rgba(166,226,46,0.03)]' 
            : 'border-[var(--line)] bg-[rgba(31,32,29,0.88)]'
        } p-4 space-y-3`}>
          <div className="flex items-center justify-between">
            <p className="text-xs font-semibold uppercase tracking-wider text-muted flex items-center gap-2">
              <Radio size={14} className={`${statusToneClass(callStatus)} ${callStatus === 'connected' ? 'animate-pulse' : ''}`} />
              Signal Status
            </p>
            <span className={`text-xs font-mono px-2 py-0.5 rounded-full border ${
              callStatus === 'connected' 
                ? 'bg-[#a6e22e]/10 border-[#a6e22e]/30 text-[#a6e22e]' 
                : 'bg-white/5 border-white/10 text-muted'
            }`}>
              {formatCallStatus(callStatus)}
              {activePresenceId ? ` : ${activePresenceId}` : ''}
            </span>
          </div>
          
          <div className="relative rounded-lg overflow-hidden bg-black/40 border border-white/5 p-1">
             <audio ref={remoteAudioRef} controls autoPlay className="w-full h-8 opacity-80 hover:opacity-100 transition-opacity">
               <track kind="captions" />
             </audio>
          </div>
          
          <div className="flex items-center gap-2 text-[10px] text-muted opacity-70">
            <div className="w-1.5 h-1.5 rounded-full bg-current" />
            <p>Stream: <code>mix.wav</code> + Presence TTS</p>
          </div>
          
          {callError ? (
            <div className="text-xs text-[#f92672] bg-[#f92672]/10 p-2 rounded border border-[#f92672]/20 flex items-center gap-2">
               <span>⚠️</span> {callError}
            </div>
          ) : null}
        </section>

        <section className="rounded-xl border border-[var(--line)] bg-[rgba(39,40,34,0.84)] p-1">
          <div className="relative">
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder={`Message ${selectedPresence?.name.en ?? "Presence"}...`}
              className="w-full min-h-[80px] rounded-lg bg-transparent px-4 py-3 text-sm text-ink outline-none resize-none placeholder:text-muted/50"
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void askPresence();
                }
              }}
            />
            <div className="absolute bottom-2 right-2">
              <button
                type="button"
                onClick={() => void askPresence()}
                disabled={isAsking || !question.trim()}
                className="btn-base flex items-center gap-2 px-3 py-1.5 rounded-md bg-white/5 hover:bg-white/10 text-xs font-medium border border-white/10 disabled:opacity-30 disabled:cursor-not-allowed transition-all"
              >
                <SendHorizontal size={12} />
                {isAsking ? "Sending..." : "Send"}
              </button>
            </div>
          </div>
        </section>

        <section className="rounded-xl border border-[var(--line)] bg-[rgba(31,32,29,0.8)] p-4 flex flex-col h-[320px]">
          <div className="flex items-center justify-between mb-3">
            <p className="text-xs font-semibold uppercase tracking-wider text-muted">Call Transcript</p>
            <span className="text-[10px] text-muted opacity-60">Live feed</span>
          </div>
          
          <div className="flex-1 overflow-y-auto space-y-3 pr-2 scrollbar-thin">
            {transcript.length === 0 ? (
              <div className="h-full flex flex-col items-center justify-center text-muted opacity-50 space-y-2">
                 <p className="text-xs italic">No call events yet.</p>
              </div>
            ) : (
              transcript.map((entry) => (
                <div
                  key={entry.id}
                  className={`flex flex-col max-w-[90%] ${
                    entry.role === "user" ? "ml-auto items-end" : "mr-auto items-start"
                  }`}
                >
                  <div
                    className={`rounded-2xl px-3 py-2 text-xs whitespace-pre-wrap shadow-sm ${
                      entry.role === "user"
                        ? "bg-[#66d9ef] text-[#272822] rounded-tr-none"
                        : entry.role === "presence"
                          ? "bg-[#272822] border border-[#a6e22e] text-[#a6e22e] rounded-tl-none"
                          : "bg-[rgba(255,255,255,0.05)] text-muted italic border border-dashed border-white/10 w-full text-center"
                    }`}
                  >
                    {entry.role !== "system" && (
                      <p className="text-[9px] font-bold opacity-70 mb-0.5 uppercase tracking-wider">
                         {entry.role === "user" ? "You" : entry.presenceId}
                      </p>
                    )}
                    <p className="leading-relaxed">{entry.text}</p>
                  </div>
                  <span className="text-[9px] text-muted opacity-40 mt-1 px-1">
                    {entry.ts.split("T")[1].split(".")[0]}
                  </span>
                </div>
              ))
            )}
          </div>
        </section>
      </div>

      <audio
        ref={mixAudioRef}
        preload="none"
        loop
        muted
        src={runtimeBaseUrl() ? `${runtimeBaseUrl()}/stream/mix.wav` : "/stream/mix.wav"}
        className="hidden"
      >
        <track kind="captions" />
      </audio>
      <audio ref={speechAudioRef} preload="auto" muted className="hidden">
        <track kind="captions" />
      </audio>
    </div>
  );
}
