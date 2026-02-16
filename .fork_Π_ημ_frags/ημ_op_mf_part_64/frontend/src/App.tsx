import { useState, useCallback, useEffect, useMemo, useRef } from "react";
import { useWorldState } from "./hooks/useWorldState";
import { useVoice } from "./hooks/useVoice";
import { SimulationCanvas } from "./components/Simulation/Canvas";
import { VitalsPanel } from "./components/Panels/Vitals";
import { ChatPanel } from "./components/Panels/Chat";
import { SoundConsole } from "./components/Panels/Controls";
import { CatalogPanel } from "./components/Panels/Catalog";
import { OmniPanel } from "./components/Panels/Omni";
import { MythWorldPanel } from "./components/Panels/MythWorld";
import { PresenceMusicCommandCenter } from "./components/Panels/PresenceMusicCommandCenter";
import { WebGraphWeaverPanel } from "./components/Panels/WebGraphWeaverPanel";
import { InspirationAtlasPanel } from "./components/Panels/InspirationAtlasPanel";
import type {
  InstrumentPad,
  InstrumentState,
  UIPerspective,
  UIProjectionBundle,
  UIProjectionElementState,
  WorldInteractionResponse,
} from "./types";

const MIX_POSITION_STORAGE_KEY = "eta_mu_mix_position_seconds";

const DEFAULT_INSTRUMENT: InstrumentState = {
  masterLevel: 0.82,
  pulseLevel: 0.56,
  artifactLevel: 0.74,
  transportRate: 1,
  voiceRate: 0.96,
  voicePitch: 1,
  voiceGain: 0.86,
  delivery: "spoken",
};

const INSTRUMENT_PADS: InstrumentPad[] = [
  { id: "pad-c4", key: "4", note: "C4", labelEn: "Receipt Pulse", labelJa: "領収脈", freqHz: 261.63 },
  { id: "pad-d4", key: "5", note: "D4", labelEn: "Witness Thread", labelJa: "証糸", freqHz: 293.66 },
  { id: "pad-e4", key: "6", note: "E4", labelEn: "Fork Canticle", labelJa: "フォーク聖歌", freqHz: 329.63 },
  { id: "pad-g4", key: "7", note: "G4", labelEn: "Mage Choir", labelJa: "魔導合唱", freqHz: 392.0 },
  { id: "pad-a4", key: "8", note: "A4", labelEn: "Keeper Drone", labelJa: "番人持続", freqHz: 440.0 },
  { id: "pad-c5", key: "9", note: "C5", labelEn: "Anchor Lift", labelJa: "錨上昇", freqHz: 523.25 },
  { id: "pad-d5", key: "0", note: "D5", labelEn: "Truth Gate", labelJa: "真理門", freqHz: 587.33 },
];

const PAD_BY_KEY = new Map(INSTRUMENT_PADS.map((pad) => [pad.key, pad.id]));
const PAD_BY_ID = new Map(INSTRUMENT_PADS.map((pad) => [pad.id, pad]));

const DELIVERY_PROFILE = {
  whispered: { attackSec: 0.05, releaseSec: 0.24, vibratoHz: 5.2, vibratoDepth: 3.8 },
  spoken: { attackSec: 0.03, releaseSec: 0.2, vibratoHz: 4.4, vibratoDepth: 2.6 },
  canticle: { attackSec: 0.08, releaseSec: 0.34, vibratoHz: 5.8, vibratoDepth: 5.2 },
} as const;

interface ActivePadSynth {
  gain: GainNode;
  filter: BiquadFilterNode;
  oscillators: OscillatorNode[];
  lfoOsc: OscillatorNode;
  lfoDepth: GainNode;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function isEditableElement(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) {
    return false;
  }
  const tag = target.tagName;
  return target.isContentEditable || tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
}

export default function App() {
  const [uiPerspective, setUiPerspective] = useState<UIPerspective>("hybrid");
  const { catalog, simulation, projection, isConnected } = useWorldState(uiPerspective);
  const [preferJa, setPreferJa] = useState(true);
  const [instrument, setInstrument] = useState<InstrumentState>(DEFAULT_INSTRUMENT);
  const { stop, sing, active, queue, pack, speakText } = useVoice({ preferJa, instrument });
  
  const [overlayApi, setOverlayApi] = useState<any>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [voiceInputMeta, setVoiceInputMeta] = useState("voice input idle / 音声入力待機");
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [worldInteraction, setWorldInteraction] = useState<WorldInteractionResponse | null>(null);
  const [interactingPersonId, setInteractingPersonId] = useState<string | null>(null);
  const [performanceArmed, setPerformanceArmed] = useState(true);
  const [activePadIds, setActivePadIds] = useState<string[]>([]);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const activePadSynthsRef = useRef<Map<string, ActivePadSynth>>(new Map());
  const activePadKeysRef = useRef<Set<string>>(new Set());

  const syncActivePads = useCallback(() => {
    const next = INSTRUMENT_PADS
      .filter((pad) => activePadSynthsRef.current.has(pad.id))
      .map((pad) => pad.id);
    setActivePadIds(next);
  }, []);

  const ensureAudioContext = useCallback((): AudioContext => {
    const ExistingCtx = window.AudioContext || (window as Window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!audioCtxRef.current) {
      audioCtxRef.current = new ExistingCtx();
    }
    if (audioCtxRef.current.state === "suspended") {
      void audioCtxRef.current.resume();
    }
    return audioCtxRef.current;
  }, []);

  const stopPad = useCallback((padId: string) => {
    const synth = activePadSynthsRef.current.get(padId);
    if (!synth || !audioCtxRef.current) {
      return;
    }
    const profile = DELIVERY_PROFILE[instrument.delivery];
    const ctx = audioCtxRef.current;
    const now = ctx.currentTime;
    synth.gain.gain.cancelScheduledValues(now);
    synth.gain.gain.setValueAtTime(Math.max(0.0001, synth.gain.gain.value), now);
    synth.gain.gain.exponentialRampToValueAtTime(0.0001, now + profile.releaseSec);
    const stopAt = now + profile.releaseSec + 0.04;
    synth.oscillators.forEach((osc) => {
      osc.stop(stopAt);
    });
    synth.lfoOsc.stop(stopAt);
    activePadSynthsRef.current.delete(padId);
    activePadKeysRef.current.delete(padId);
    syncActivePads();
  }, [instrument.delivery, syncActivePads]);

  const startPad = useCallback((padId: string) => {
    if (activePadSynthsRef.current.has(padId)) {
      return;
    }
    const pad = PAD_BY_ID.get(padId);
    if (!pad) {
      return;
    }

    const ctx = ensureAudioContext();
    const profile = DELIVERY_PROFILE[instrument.delivery];
    const now = ctx.currentTime;

    const destinationGain = ctx.createGain();
    const filter = ctx.createBiquadFilter();
    filter.type = "lowpass";
    filter.frequency.value = 1800 + instrument.pulseLevel * 1600;
    filter.Q.value = 0.6 + instrument.artifactLevel * 2.0;

    const oscA = ctx.createOscillator();
    const oscB = ctx.createOscillator();
    oscA.type = "triangle";
    oscB.type = "sine";
    const baseFreq = pad.freqHz * instrument.voicePitch;
    oscA.frequency.setValueAtTime(baseFreq, now);
    oscB.frequency.setValueAtTime(baseFreq * 2.01, now);

    const lfoOsc = ctx.createOscillator();
    const lfoDepth = ctx.createGain();
    lfoOsc.frequency.value = profile.vibratoHz;
    lfoDepth.gain.value = profile.vibratoDepth;
    lfoOsc.connect(lfoDepth);
    lfoDepth.connect(oscA.detune);
    lfoDepth.connect(oscB.detune);

    const padGainTarget = clamp(
      instrument.masterLevel * instrument.artifactLevel * instrument.voiceGain * 0.32,
      0.03,
      0.45,
    );

    destinationGain.gain.setValueAtTime(0.0001, now);
    destinationGain.gain.exponentialRampToValueAtTime(padGainTarget, now + profile.attackSec);

    oscA.connect(filter);
    oscB.connect(filter);
    filter.connect(destinationGain);
    destinationGain.connect(ctx.destination);

    oscA.start(now);
    oscB.start(now);
    lfoOsc.start(now);

    activePadSynthsRef.current.set(padId, {
      gain: destinationGain,
      filter,
      oscillators: [oscA, oscB],
      lfoOsc,
      lfoDepth,
    });
    activePadKeysRef.current.add(padId);
    syncActivePads();
    setVoiceInputMeta(`instrument pad: ${pad.labelEn} / ${pad.labelJa}`);
  }, [ensureAudioContext, instrument.artifactLevel, instrument.delivery, instrument.masterLevel, instrument.pulseLevel, instrument.voiceGain, instrument.voicePitch, syncActivePads]);

  const handleFileUpload = useCallback(async (file: File) => {
    const reader = new FileReader();
    reader.onload = async () => {
      const b64 = (reader.result as string).split(',')[1];
      setVoiceInputMeta(`learning frequency: ${file.name}...`);
      try {
        const baseUrl = window.location.port === '5173' ? 'http://127.0.0.1:8787' : '';
        const res = await fetch(`${baseUrl}/api/upload`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            name: file.name,
            base64: b64,
            mime: file.type
          })
        });
        const data = await res.json();
        if (data.ok) {
          setVoiceInputMeta(`learned: ${data.text || file.name}`);
          // Add system message to chat
          window.dispatchEvent(new CustomEvent("chat-message", {
            detail: { role: "system", text: `The Weaver has learned a new frequency from ${file.name}: "${data.text}"` }
          }));
        }
      } catch (e) {
        setVoiceInputMeta("learning failed");
      }
    };
    reader.readAsDataURL(file);
  }, []);

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const files = Array.from(e.dataTransfer.files);
    files.forEach(file => {
      if (file.type.startsWith('audio/')) {
        handleFileUpload(file);
      }
    });
  }, [handleFileUpload]);

  const onDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const onDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  // Voice Recording Logic
  const handleRecord = useCallback(async () => {
    if(isRecording) return;
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const mediaRecorder = new MediaRecorder(stream);
        const chunks: BlobPart[] = [];
        
        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) chunks.push(e.data);
        };
        
        mediaRecorder.onstop = () => {
            const blob = new Blob(chunks, { type: "audio/webm" });
            setRecordedBlob(blob);
            setVoiceInputMeta(`voice captured / 音声取得: ${Math.round(blob.size/1024)}KB`);
            stream.getTracks().forEach((t) => {
              t.stop();
            });
            setIsRecording(false);
        };
        
        mediaRecorder.start();
        setIsRecording(true);
        setVoiceInputMeta("recording voice / 録音中");
        
        setTimeout(() => {
            if(mediaRecorder.state === "recording") mediaRecorder.stop();
        }, 8000);
        
    } catch(e) {
        setVoiceInputMeta("mic permission denied / マイク許可なし");
    }
  }, [isRecording]);

  const handleTranscribe = useCallback(async () => {
    if(!recordedBlob) return;
    const buf = await recordedBlob.arrayBuffer();
    // Convert to base64
    let binary = '';
    const bytes = new Uint8Array(buf);
    for (let i = 0; i < bytes.byteLength; i++) {
        binary += String.fromCharCode(bytes[i]);
    }
    const b64 = btoa(binary);

    try {
        const baseUrl = window.location.port === '5173' ? 'http://127.0.0.1:8787' : '';
        const res = await fetch(`${baseUrl}/api/transcribe`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ audio_base64: b64, mime: recordedBlob.type })
        });
        const data = await res.json();
        if(data.ok) {
            setVoiceInputMeta(`transcribed: ${data.text}`);
            return data.text;
        } else {
            setVoiceInputMeta(`error: ${data.error}`);
        }
    } catch(e) {
        setVoiceInputMeta("transcribe failed");
    }
  }, [recordedBlob]);

  const handleSendVoice = useCallback(async () => {
    const text = await handleTranscribe();
    if(text) {
        window.dispatchEvent(new CustomEvent("chat-message", { 
            detail: { role: "user", text } 
        }));
        
        const baseUrl = window.location.port === '5173' ? 'http://127.0.0.1:8787' : '';
        fetch(`${baseUrl}/api/chat`, {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({ messages: [{role: "user", text}] })
        }).then(r => r.json()).then(data => {
             window.dispatchEvent(new CustomEvent("chat-message", { 
                detail: { role: "assistant", text: data.reply } 
            }));
            if(data.reply.includes("[[PULSE]]") && overlayApi) overlayApi.pulseAt(0.5, 0.5, 1.0);
            if(data.reply.includes("[[SING]]") && overlayApi) overlayApi.singAll();
        });
    }
  }, [handleTranscribe, overlayApi]);

  const handleHandoff = useCallback(async () => {
    try {
        const baseUrl = window.location.port === '5173' ? 'http://127.0.0.1:8787' : '';
        const res = await fetch(`${baseUrl}/api/handoff`);
        if(res.ok) {
            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            const a = document.createElement("a");
            a.href = url;
            a.download = `HANDOFF_Part_70_${new Date().toISOString().split('T')[0]}.md`;
            a.click();
            URL.revokeObjectURL(url);
            setVoiceInputMeta("handoff report generated / 引き継ぎレポート作成完了");
        }
    } catch(e) {
        setVoiceInputMeta("handoff failed");
    }
  }, []);

  const applyInstrumentToAudio = useCallback((smoothPulseRamp = false) => {
    const audios = Array.from(document.querySelectorAll("audio"));
    audios.forEach((audio) => {
      const isPulseStream = audio.id === "mix-stream";
      const targetVolume = clamp(
        instrument.masterLevel * (isPulseStream ? instrument.pulseLevel : instrument.artifactLevel),
        0,
        1,
      );

      audio.playbackRate = instrument.transportRate;

      if (audio.muted) {
        return;
      }

      if (smoothPulseRamp && isPulseStream) {
        const startVolume = Math.min(audio.volume, targetVolume * 0.25);
        const started = performance.now();
        audio.volume = startVolume;
        const durationMs = 1100;

        const ramp = (now: number) => {
          const progress = Math.min((now - started) / durationMs, 1);
          const eased = progress * progress * (3 - 2 * progress);
          audio.volume = startVolume + (targetVolume - startVolume) * eased;
          if (progress < 1) {
            requestAnimationFrame(ramp);
          }
        };
        requestAnimationFrame(ramp);
        return;
      }

      audio.volume = targetVolume;
    });
  }, [instrument.artifactLevel, instrument.masterLevel, instrument.pulseLevel, instrument.transportRate]);

  useEffect(() => {
    applyInstrumentToAudio(false);
  }, [applyInstrumentToAudio]);

  useEffect(() => {
    const ctx = audioCtxRef.current;
    if (!ctx) {
      return;
    }

    const now = ctx.currentTime;
    const profile = DELIVERY_PROFILE[instrument.delivery];
    activePadSynthsRef.current.forEach((synth, padId) => {
      const pad = PAD_BY_ID.get(padId);
      if (!pad) {
        return;
      }

      const baseFreq = pad.freqHz * instrument.voicePitch;
      synth.oscillators[0].frequency.setTargetAtTime(baseFreq, now, 0.02);
      synth.oscillators[1].frequency.setTargetAtTime(baseFreq * 2.01, now, 0.02);
      synth.filter.frequency.setTargetAtTime(1800 + instrument.pulseLevel * 1600, now, 0.03);
      synth.filter.Q.setTargetAtTime(0.6 + instrument.artifactLevel * 2.0, now, 0.03);
      synth.lfoOsc.frequency.setTargetAtTime(profile.vibratoHz, now, 0.04);
      synth.lfoDepth.gain.setTargetAtTime(profile.vibratoDepth, now, 0.04);

      const targetGain = clamp(
        instrument.masterLevel * instrument.artifactLevel * instrument.voiceGain * 0.32,
        0.03,
        0.45,
      );
      synth.gain.gain.setTargetAtTime(targetGain, now, 0.03);
    });
  }, [
    instrument.artifactLevel,
    instrument.delivery,
    instrument.masterLevel,
    instrument.pulseLevel,
    instrument.voiceGain,
    instrument.voicePitch,
  ]);

  useEffect(() => {
    const mixStream = document.getElementById("mix-stream") as HTMLAudioElement | null;
    if (!mixStream) {
      return;
    }

    const restorePosition = () => {
      try {
        const raw = window.sessionStorage.getItem(MIX_POSITION_STORAGE_KEY);
        const saved = raw === null ? Number.NaN : Number(raw);
        if (!Number.isFinite(saved) || saved <= 0) {
          return;
        }
        if (Number.isFinite(mixStream.duration) && saved >= mixStream.duration) {
          return;
        }
        mixStream.currentTime = saved;
      } catch {
        return;
      }
    };

    const persistPosition = () => {
      try {
        window.sessionStorage.setItem(MIX_POSITION_STORAGE_KEY, String(mixStream.currentTime));
      } catch {
        return;
      }
    };

    mixStream.addEventListener("loadedmetadata", restorePosition);
    mixStream.addEventListener("timeupdate", persistPosition);
    mixStream.addEventListener("pause", persistPosition);

    if (mixStream.readyState > 0) {
      restorePosition();
    }

    return () => {
      mixStream.removeEventListener("loadedmetadata", restorePosition);
      mixStream.removeEventListener("timeupdate", persistPosition);
      mixStream.removeEventListener("pause", persistPosition);
    };
  }, []);

  const handlePlayAll = useCallback(() => {
    const audios = document.querySelectorAll("audio");
    audios.forEach((a) => {
      void a.play().catch(() => {});
    });
    applyInstrumentToAudio(true);
  }, [applyInstrumentToAudio]);

  const handlePauseAll = useCallback(() => {
    const audios = document.querySelectorAll("audio");
    audios.forEach((a) => {
      a.pause();
    });
  }, []);

  const handleMuteAll = useCallback(() => {
    const audios = document.querySelectorAll("audio");
    audios.forEach(a => { a.muted = true; });
  }, []);

  const handleUnmuteAll = useCallback(() => {
    const audios = document.querySelectorAll("audio");
    audios.forEach(a => { a.muted = false; });
    applyInstrumentToAudio(false);
  }, [applyInstrumentToAudio]);

  const handleReloadMix = () => {
    const mix = document.getElementById("mix-stream") as HTMLAudioElement;
    if(mix) {
        mix.src = `/stream/mix.wav?t=${Date.now()}`;
        mix.load();
        mix.addEventListener("loadedmetadata", () => {
          applyInstrumentToAudio(false);
        }, { once: true });
    }
  };

  useEffect(() => {
    if (!performanceArmed) {
      activePadSynthsRef.current.forEach((_synth, padId) => {
        stopPad(padId);
      });
      return;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.repeat || event.metaKey || event.ctrlKey || event.altKey) {
        return;
      }
      if (isEditableElement(event.target)) {
        return;
      }

      const lower = event.key.toLowerCase();
      const padId = PAD_BY_KEY.get(lower);
      if (padId) {
        if (!activePadKeysRef.current.has(padId)) {
          startPad(padId);
        }
        event.preventDefault();
        return;
      }

      let handled = true;

      if (event.key === " ") {
        const audioEls = Array.from(document.querySelectorAll("audio"));
        const anyPlaying = audioEls.some((audio) => !audio.paused);
        if (anyPlaying) {
          handlePauseAll();
          setVoiceInputMeta("instrument: transport paused / 輸送停止");
        } else {
          handlePlayAll();
          setVoiceInputMeta("instrument: transport running / 輸送再開");
        }
      } else if (lower === "m") {
        const audioEls = Array.from(document.querySelectorAll("audio"));
        const shouldMute = audioEls.some((audio) => !audio.muted);
        if (shouldMute) {
          handleMuteAll();
          setVoiceInputMeta("instrument: mute on / ミュートON");
        } else {
          handleUnmuteAll();
          setVoiceInputMeta("instrument: mute off / ミュートOFF");
        }
      } else if (lower === "q") {
        void sing("canonical");
        setVoiceInputMeta("instrument: canonical phrase / 正準フレーズ");
      } else if (lower === "w") {
        void sing("ollama");
        setVoiceInputMeta("instrument: ollama phrase / Ollamaフレーズ");
      } else if (lower === "x") {
        stop();
        setVoiceInputMeta("instrument: voices cut / 声停止");
      } else if (lower === "f") {
        overlayApi?.singAll();
        setVoiceInputMeta("instrument: field choir / 場の合唱");
      } else if (lower === "j") {
        setPreferJa((prev) => !prev);
      } else if (lower === "[") {
        setInstrument((prev) => ({ ...prev, masterLevel: clamp(prev.masterLevel - 0.04, 0.1, 1) }));
      } else if (lower === "]") {
        setInstrument((prev) => ({ ...prev, masterLevel: clamp(prev.masterLevel + 0.04, 0.1, 1) }));
      } else if (lower === "-") {
        setInstrument((prev) => ({ ...prev, pulseLevel: clamp(prev.pulseLevel - 0.04, 0.05, 1) }));
      } else if (lower === "=") {
        setInstrument((prev) => ({ ...prev, pulseLevel: clamp(prev.pulseLevel + 0.04, 0.05, 1) }));
      } else if (lower === ",") {
        setInstrument((prev) => ({ ...prev, transportRate: clamp(prev.transportRate - 0.03, 0.75, 1.25) }));
      } else if (lower === ".") {
        setInstrument((prev) => ({ ...prev, transportRate: clamp(prev.transportRate + 0.03, 0.75, 1.25) }));
      } else if (lower === "o") {
        setInstrument((prev) => ({ ...prev, voiceRate: clamp(prev.voiceRate - 0.03, 0.7, 1.35) }));
      } else if (lower === "p") {
        setInstrument((prev) => ({ ...prev, voiceRate: clamp(prev.voiceRate + 0.03, 0.7, 1.35) }));
      } else if (lower === "k") {
        setInstrument((prev) => ({ ...prev, voicePitch: clamp(prev.voicePitch - 0.03, 0.75, 1.45) }));
      } else if (lower === "l") {
        setInstrument((prev) => ({ ...prev, voicePitch: clamp(prev.voicePitch + 0.03, 0.75, 1.45) }));
      } else if (lower === "n") {
        setInstrument((prev) => ({ ...prev, artifactLevel: clamp(prev.artifactLevel - 0.04, 0.05, 1) }));
      } else if (lower === "b") {
        setInstrument((prev) => ({ ...prev, artifactLevel: clamp(prev.artifactLevel + 0.04, 0.05, 1) }));
      } else if (lower === ";") {
        setInstrument((prev) => ({ ...prev, voiceGain: clamp(prev.voiceGain - 0.03, 0.2, 1) }));
      } else if (lower === "'") {
        setInstrument((prev) => ({ ...prev, voiceGain: clamp(prev.voiceGain + 0.03, 0.2, 1) }));
      } else if (lower === "1") {
        setInstrument((prev) => ({ ...prev, delivery: "whispered" }));
      } else if (lower === "2") {
        setInstrument((prev) => ({ ...prev, delivery: "spoken" }));
      } else if (lower === "3") {
        setInstrument((prev) => ({ ...prev, delivery: "canticle" }));
      } else {
        handled = false;
      }

      if (handled) {
        event.preventDefault();
      }
    };

    const handleKeyUp = (event: KeyboardEvent) => {
      const lower = event.key.toLowerCase();
      const padId = PAD_BY_KEY.get(lower);
      if (!padId) {
        return;
      }
      stopPad(padId);
      event.preventDefault();
    };

    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
    };
  }, [
    handleMuteAll,
    handlePauseAll,
    handlePlayAll,
    handleUnmuteAll,
    overlayApi,
    performanceArmed,
    sing,
    startPad,
    stopPad,
    stop,
  ]);

  useEffect(() => {
    const synths = activePadSynthsRef.current;
    return () => {
      synths.forEach((_synth, padId) => {
        stopPad(padId);
      });
      if (audioCtxRef.current && audioCtxRef.current.state !== "closed") {
        void audioCtxRef.current.close();
      }
    };
  }, [stopPad]);

  useEffect(() => {
    const clearPads = () => {
      activePadSynthsRef.current.forEach((_synth, padId) => {
        stopPad(padId);
      });
    };

    window.addEventListener("blur", clearPads);
    document.addEventListener("visibilitychange", clearPads);
    return () => {
      window.removeEventListener("blur", clearPads);
      document.removeEventListener("visibilitychange", clearPads);
    };
  }, [stopPad]);

  const voiceMeta = `voice mode: ${pack?.mode || 'canonical'} | delivery: ${instrument.delivery} | pads: ${activePadIds.length} | active: ${active} | queue: ${queue.length}`;

  const emitSystemMessage = useCallback((text: string) => {
    window.dispatchEvent(new CustomEvent('chat-message', {
      detail: {
        role: 'system',
        text,
      },
    }));
  }, []);

  const handleLedgerCommand = useCallback(async (text: string): Promise<boolean> => {
    const trimmed = text.trim();
    if (!trimmed.toLowerCase().startsWith('/ledger')) {
      return false;
    }

    const payloadText = trimmed.replace(/^\/ledger\s*/i, '');
    const baseUrl = window.location.port === '5173' ? 'http://127.0.0.1:8787' : '';
    const utterances = payloadText
      ? payloadText.split('|').map((row) => row.trim()).filter((row) => row.length > 0)
      : [];

    try {
      const res = await fetch(`${baseUrl}/api/eta-mu-ledger`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ utterances }),
      });
      const data = await res.json();
      const body = data?.jsonl ? data.jsonl.trim() : '(no utterances)';
      emitSystemMessage(`eta/mu ledger\n${body}`);
    } catch (_e) {
      emitSystemMessage('eta/mu ledger failed');
    }
    return true;
  }, [emitSystemMessage]);

  const handlePresenceSayCommand = useCallback(async (text: string): Promise<boolean> => {
    const trimmed = text.trim();
    if (!trimmed.toLowerCase().startsWith('/say')) {
      return false;
    }

    const args = trimmed.replace(/^\/say\s*/i, '');
    const [presenceIdRaw, ...rest] = args.split(/\s+/).filter((token) => token.length > 0);
    const presence_id = presenceIdRaw || 'witness_thread';
    const messageText = rest.join(' ');
    const baseUrl = window.location.port === '5173' ? 'http://127.0.0.1:8787' : '';

    try {
      const res = await fetch(`${baseUrl}/api/presence/say`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          presence_id,
          text: messageText,
        }),
      });
      const data = await res.json();
      emitSystemMessage(
        `${data?.presence_name?.en || presence_id} / say\n${data?.rendered_text || '(no render)'}\n` +
        `facts=${data?.say_intent?.facts?.length || 0} asks=${data?.say_intent?.asks?.length || 0} repairs=${data?.say_intent?.repairs?.length || 0}`,
      );
    } catch (_e) {
      emitSystemMessage('presence say failed');
    }
    return true;
  }, [emitSystemMessage]);

  const handleDriftCommand = useCallback(async (text: string): Promise<boolean> => {
    const trimmed = text.trim();
    if (trimmed.toLowerCase() !== '/drift') {
      return false;
    }

    const baseUrl = window.location.port === '5173' ? 'http://127.0.0.1:8787' : '';
    try {
      const res = await fetch(`${baseUrl}/api/drift/scan`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      const data = await res.json();
      const drifts = Array.isArray(data?.active_drifts) ? data.active_drifts.length : 0;
      const blocked = Array.isArray(data?.blocked_gates) ? data.blocked_gates.length : 0;
      emitSystemMessage(`drift scan\nactive_drifts=${drifts} blocked_gates=${blocked}`);
    } catch (_e) {
      emitSystemMessage('drift scan failed');
    }
    return true;
  }, [emitSystemMessage]);

  const handlePushTruthDryRunCommand = useCallback(async (text: string): Promise<boolean> => {
    const trimmed = text.trim().toLowerCase();
    if (trimmed !== '/push-truth --dry-run') {
      return false;
    }

    const baseUrl = window.location.port === '5173' ? 'http://127.0.0.1:8787' : '';
    try {
      const res = await fetch(`${baseUrl}/api/push-truth/dry-run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      const data = await res.json();
      const blocked = data?.gate?.blocked ? 'blocked' : 'pass';
      const needs = Array.isArray(data?.needs) ? data.needs.join(', ') : '';
      emitSystemMessage(`push-truth dry-run\ngate=${blocked}\nneeds=${needs || '(none)'}`);
    } catch (_e) {
      emitSystemMessage('push-truth dry-run failed');
    }
    return true;
  }, [emitSystemMessage]);

  const handleChatCommand = useCallback(async (text: string): Promise<boolean> => {
    if (await handleLedgerCommand(text)) {
      return true;
    }
    if (await handlePresenceSayCommand(text)) {
      return true;
    }
    if (await handleDriftCommand(text)) {
      return true;
    }
    if (await handlePushTruthDryRunCommand(text)) {
      return true;
    }
    return false;
  }, [handleDriftCommand, handleLedgerCommand, handlePresenceSayCommand, handlePushTruthDryRunCommand]);

  const handleWorldInteract = useCallback(async (personId: string, action: 'speak' | 'pray' | 'sing') => {
    setInteractingPersonId(personId);
    try {
      const baseUrl = window.location.port === '5173' ? 'http://127.0.0.1:8787' : '';
      const res = await fetch(`${baseUrl}/api/world/interact`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ person_id: personId, action }),
      });
      const data = await res.json();
      setWorldInteraction(data);

      if (data?.ok) {
        window.dispatchEvent(new CustomEvent('chat-message', {
          detail: {
            role: 'assistant',
            text: `${data.line_en}\n${data.line_ja}`,
          },
        }));
        speakText(data.voice_text_en || data.line_en || '', data.voice_text_ja || data.line_ja || '');
      }
    } catch (_e) {
      setWorldInteraction({
        ok: false,
        line_en: 'Interaction failed. The field is unstable.',
        line_ja: '対話に失敗。場が不安定です。',
      });
    } finally {
      setInteractingPersonId(null);
    }
  }, [speakText]);

  const activeProjection: UIProjectionBundle | null =
    projection ?? simulation?.projection ?? catalog?.ui_projection ?? null;

  const projectionStateByElement = useMemo(() => {
    const map = new Map<string, UIProjectionElementState>();
    if (!activeProjection) {
      return map;
    }
    activeProjection.states.forEach((state) => {
      map.set(state.element_id, state);
    });
    return map;
  }, [activeProjection]);

  const projectionRectByElement = useMemo(() => {
    const map = new Map<string, { x: number; y: number; w: number; h: number }>();
    const rects = activeProjection?.layout?.rects ?? {};
    Object.entries(rects).forEach(([elementId, rect]) => {
      map.set(elementId, rect);
    });
    return map;
  }, [activeProjection]);

  const projectionStyleFor = useCallback((elementId: string, fallbackSpan = 12) => {
    const rect = projectionRectByElement.get(elementId);
    const state = projectionStateByElement.get(elementId);
    const colSpan = rect ? clamp(Math.round(rect.w * 12), 3, 12) : fallbackSpan;
    const rowSpan = rect ? clamp(Math.round(rect.h * 10), 2, 6) : 3;
    const pulseScale = state ? 1 + (clamp(state.pulse, 0, 1) * 0.014) : 1;
    return {
      gridColumn: `span ${colSpan} / span ${colSpan}`,
      gridRow: `span ${rowSpan} / span ${rowSpan}`,
      opacity: state ? clamp(state.opacity, 0.5, 1) : 1,
      transform: state ? `scale(${pulseScale.toFixed(3)})` : undefined,
      transformOrigin: "center top",
      transition:
        "grid-column 220ms ease, grid-row 220ms ease, transform 260ms ease, opacity 220ms ease",
    } as const;
  }, [projectionRectByElement, projectionStateByElement]);

  const projectionPerspective = activeProjection?.perspective ?? uiPerspective;
  const projectionOptions =
    activeProjection?.perspectives ??
    catalog?.ui_perspectives ??
    [
      {
        id: "hybrid",
        symbol: "perspective.hybrid",
        name: "Hybrid",
        merge: "hybrid",
        description: "Wallclock ordering with causal overlays.",
        default: true,
      },
      {
        id: "causal-time",
        symbol: "perspective.causal-time",
        name: "Causal Time",
        merge: "causal-time",
        description: "Prioritize causal links over wallclock sequence.",
        default: false,
      },
      {
        id: "swimlanes",
        symbol: "perspective.swimlanes",
        name: "Swimlanes",
        merge: "swimlanes",
        description: "Parallel lanes with threaded causality.",
        default: false,
      },
    ];

  const projectionHighlights = useMemo(() => {
    const rows = [...projectionStateByElement.values()];
    rows.sort((a, b) => b.priority - a.priority);
    return rows.slice(0, 4);
  }, [projectionStateByElement]);

  const activeChatLens = activeProjection?.chat_sessions?.[0] ?? null;
  const chatLensState = projectionStateByElement.get("nexus.ui.chat.witness_thread") ?? null;

  return (
    <main 
      className={`max-w-[1100px] mx-auto p-6 md:p-12 pb-24 transition-colors ${isDragging ? 'bg-blue-50/50' : ''}`}
      onDrop={onDrop}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
    >
      <header className="mb-12 border-b-2 border-line pb-6">
        <h1 className="text-5xl font-bold tracking-tight mb-2 text-ink">eta-mu world daemon / ημ世界デーモン</h1>
        <div className="flex justify-between items-center">
          <p className="text-muted text-sm font-mono">
            Part <code>{catalog?.part_roots?.[0]?.split('/').pop() || "?"}</code> | Seed <code>{catalog?.generated_at?.split('T')[0]}</code>
          </p>
          {!isConnected && <span className="text-red-600 font-bold animate-pulse">● Disconnected / 切断</span>}
          {isConnected && <span className="text-green-600 font-bold flex items-center gap-2">● Connected / 接続中</span>}
        </div>
        <div className="mt-4 grid gap-3 lg:grid-cols-[1fr_auto] lg:items-center">
          <div className="text-xs text-muted space-y-1">
            <p>
              projection perspective: <code>{projectionPerspective}</code>
              {activeProjection?.layout?.clamps ? (
                <>
                  {" "}| clamps area <code>{activeProjection.layout.clamps.min_area.toFixed(2)}</code>-<code>{activeProjection.layout.clamps.max_area.toFixed(2)}</code>
                </>
              ) : null}
            </p>
            {activeChatLens ? (
              <p>
                chat lens: <code>{activeChatLens.presence}</code> | memory scope: <code>{activeChatLens.memory_scope}</code> | status: <code>{activeChatLens.status}</code>
              </p>
            ) : null}
          </div>
          <div className="flex flex-wrap gap-2">
            {projectionOptions.map((option) => (
              <button
                key={option.id}
                type="button"
                onClick={() => setUiPerspective(option.id as UIPerspective)}
                className={`border rounded-md px-3 py-1 text-xs font-semibold transition-colors ${projectionPerspective === option.id ? 'bg-[#1f3946] text-[#f4f9ff] border-[#1f3946]' : 'bg-[#f8f4ee] text-[#2f2a24] border-[var(--line)] hover:bg-white'}`}
                title={option.description}
              >
                {option.name}
              </button>
            ))}
          </div>
        </div>
      </header>

      <div className="grid grid-cols-1 xl:grid-cols-12 gap-6 items-start">
        <section className="xl:col-span-12" style={projectionStyleFor("nexus.ui.command_center", 12)}>
          <PresenceMusicCommandCenter
            catalog={catalog}
            simulation={simulation}
            instrument={instrument}
            activePadIds={activePadIds}
            performanceArmed={performanceArmed}
          />
        </section>

        <section className="xl:col-span-6" style={projectionStyleFor("nexus.ui.web_graph_weaver", 6)}>
          <WebGraphWeaverPanel />
        </section>

        <section className="xl:col-span-6" style={projectionStyleFor("nexus.ui.inspiration_atlas", 6)}>
          <InspirationAtlasPanel simulation={simulation} />
        </section>

        <section className="card !mt-0 relative overflow-hidden xl:col-span-12" style={projectionStyleFor("nexus.ui.simulation_map", 12)}>
          <div className="absolute top-0 left-0 w-1 h-full bg-blue-400 opacity-50" />
          <h2 className="text-3xl font-bold mb-6">Everything Dashboard (real-time)</h2>

          <SimulationCanvas
              simulation={simulation}
              catalog={catalog}
              onOverlayInit={setOverlayApi}
              height={400}
          />

          <div className="mt-8 space-y-6">
              <div>
                  <p className="font-bold text-lg mb-2 flex items-center gap-2">
                    Combined stream / 合成ストリーム
                  </p>
                   <audio
                       id="mix-stream"
                       controls
                       preload="none"
                       src={window.location.port === '5173' ? 'http://127.0.0.1:8787/stream/mix.wav' : '/stream/mix.wav'}
                       className="w-full bg-bg-0 rounded-lg p-1"
                  >
                    <track kind="captions" />
                  </audio>
               </div>

               <SoundConsole
                   onPlayAll={handlePlayAll}
                   onPauseAll={handlePauseAll}
                   onMuteAll={handleMuteAll}
                   onUnmuteAll={handleUnmuteAll}
                   onReloadMix={handleReloadMix}
                   onOverlayToggle={() => {}}
                   onSingFields={() => overlayApi?.singAll()}
                   onPrimeVoice={() => sing("canonical")}
                    onSingWords={() => sing("canonical")}
                    onSingOllama={() => sing("ollama")}
                    onStopVoices={stop}
                    onHandoff={handleHandoff}
                    performanceArmed={performanceArmed}
                    onTogglePerformanceArm={() => setPerformanceArmed((prev) => !prev)}
                    pads={INSTRUMENT_PADS}
                    activePadIds={activePadIds}
                    onPadStart={startPad}
                    onPadStop={stopPad}
                    preferJa={preferJa}
                    setPreferJa={setPreferJa}
                    voiceMeta={voiceMeta}
                    instrument={instrument}
                   onInstrumentChange={setInstrument}
               />

              {chatLensState ? (
                <p className="text-xs text-muted font-mono">
                  chat-lens mass <code>{chatLensState.mass.toFixed(2)}</code> | priority <code>{chatLensState.priority.toFixed(2)}</code> | reason <code>{chatLensState.explain.dominant_field}</code>
                </p>
              ) : null}
              <div
                style={{
                  opacity: chatLensState ? clamp(chatLensState.opacity, 0.5, 1) : 1,
                  transform: chatLensState ? `scale(${(1 + chatLensState.pulse * 0.012).toFixed(3)})` : undefined,
                  transformOrigin: "center top",
                  transition: "transform 200ms ease, opacity 200ms ease",
                }}
              >
                <ChatPanel
                  onSend={(text) => {
                      setIsThinking(true);
                      (async () => {
                        const consumed = await handleChatCommand(text);
                        if (consumed) {
                          return;
                        }

                        const baseUrl = window.location.port === '5173' ? 'http://127.0.0.1:8787' : '';
                        const response = await fetch(`${baseUrl}/api/chat`, {
                          method: 'POST',
                          headers: {'Content-Type': 'application/json'},
                          body: JSON.stringify({ messages: [{role: 'user', text}] })
                        });
                        const data = await response.json();
                        window.dispatchEvent(new CustomEvent('chat-message', {
                          detail: { role: 'assistant', text: data.reply }
                        }));
                        if (data.reply.includes('[[PULSE]]') && overlayApi) overlayApi.pulseAt(0.5, 0.5, 1.0);
                        if (data.reply.includes('[[SING]]') && overlayApi) overlayApi.singAll();
                      })().catch(() => {
                        window.dispatchEvent(new CustomEvent('chat-message', {
                          detail: { role: 'system', text: 'chat request failed' }
                        }));
                      }).finally(() => {
                        setIsThinking(false);
                      });
                  }}
                  onRecord={handleRecord}
                  onTranscribe={handleTranscribe}
                  onSendVoice={handleSendVoice}
                  isRecording={isRecording}
                  isThinking={isThinking}
                  voiceInputMeta={voiceInputMeta}
                />
              </div>
          </div>
        </section>

        <section className="card relative overflow-hidden xl:col-span-6" style={projectionStyleFor("nexus.ui.entity_vitals", 6)}>
          <div className="absolute top-0 left-0 w-1 h-full bg-green-400 opacity-50" />
          <h2 className="text-3xl font-bold mb-2">Entity Vitals / 実体バイタル</h2>
          <p className="text-muted mb-6">Live telemetry from the canonical named forms.</p>
          <VitalsPanel
            entities={simulation?.entities}
            catalog={catalog}
            presenceDynamics={simulation?.presence_dynamics}
          />
        </section>

        <section className="card relative overflow-hidden xl:col-span-6" style={projectionStyleFor("nexus.ui.omni_archive", 6)}>
          <div className="absolute top-0 left-0 w-1 h-full bg-cyan-400 opacity-55" />
          <h2 className="text-2xl font-bold mb-2">Projection Ledger / 映台帳</h2>
          <p className="text-muted mb-4">Explainable panel weights derived from field + presence forces.</p>
          <div className="space-y-2">
            {projectionHighlights.map((state) => (
              <div key={state.element_id} className="border border-[rgba(36,31,26,0.14)] rounded-lg bg-[rgba(255,255,255,0.82)] p-2">
                <p className="text-xs font-semibold text-ink">
                  <code>{state.element_id}</code>
                </p>
                <p className="text-[11px] text-muted font-mono">
                  mass {state.mass.toFixed(2)} | priority {state.priority.toFixed(2)} | area {state.area.toFixed(2)}
                </p>
                <p className="text-[11px] text-muted">{state.explain.reason_en}</p>
              </div>
            ))}
          </div>
        </section>

        <section className="card relative overflow-hidden xl:col-span-8" style={projectionStyleFor("nexus.ui.omni_archive", 8)}>
          <div className="absolute top-0 left-0 w-1 h-full bg-purple-400 opacity-50" />
          <h2 className="text-3xl font-bold mb-2">Omni Panel / 全感覚パネル</h2>
          <p className="text-muted mb-6">Receipt River, Mage of Receipts, and other cover entities.</p>
          <OmniPanel catalog={catalog} />
          <div className="mt-8">
            <h3 className="text-2xl font-bold mb-4">Vault Artifacts / 遺物録</h3>
            <CatalogPanel catalog={catalog} />
          </div>
        </section>

        <section className="card relative overflow-hidden xl:col-span-4" style={projectionStyleFor("nexus.ui.myth_commons", 4)}>
          <div className="absolute top-0 left-0 w-1 h-full bg-amber-400 opacity-60" />
          <h2 className="text-3xl font-bold mb-2">Myth Commons / 神話共同体</h2>
          <p className="text-muted mb-6">People sing, pray to the Presences, and keep writing the myth.</p>
          <MythWorldPanel
            simulation={simulation}
            interaction={worldInteraction}
            interactingPersonId={interactingPersonId}
            onInteract={handleWorldInteract}
          />
        </section>
      </div>
    </main>
  );
}
