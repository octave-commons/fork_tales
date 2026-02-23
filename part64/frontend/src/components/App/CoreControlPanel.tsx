import { useState } from "react";
import { OVERLAY_VIEW_OPTIONS, type OverlayViewId } from "../Simulation/Canvas";
import type { UIPerspective } from "../../types";
import {
  CORE_ORBIT_SPEED_MAX,
  CORE_ORBIT_SPEED_MIN,
  CORE_SIM_LAYER_DEPTH_MAX,
  CORE_SIM_LAYER_DEPTH_MIN,
  CORE_SIM_MOTION_SPEED_MAX,
  CORE_SIM_MOTION_SPEED_MIN,
  CORE_SIM_MOUSE_INFLUENCE_MAX,
  CORE_SIM_MOUSE_INFLUENCE_MIN,
  CORE_SIM_PARTICLE_DENSITY_MAX,
  CORE_SIM_PARTICLE_DENSITY_MIN,
  CORE_SIM_PARTICLE_SCALE_MAX,
  CORE_SIM_PARTICLE_SCALE_MIN,
  CORE_VISUAL_BRIGHTNESS_MAX,
  CORE_VISUAL_BRIGHTNESS_MIN,
  CORE_VISUAL_CONTRAST_MAX,
  CORE_VISUAL_CONTRAST_MIN,
  CORE_VISUAL_HUE_MAX,
  CORE_VISUAL_HUE_MIN,
  CORE_VISUAL_SATURATION_MAX,
  CORE_VISUAL_SATURATION_MIN,
  CORE_VISUAL_VIGNETTE_MAX,
  CORE_VISUAL_VIGNETTE_MIN,
  CORE_VISUAL_WASH_MAX,
  CORE_VISUAL_WASH_MIN,
  type CoreSimulationTuning,
  type CoreVisualTuning,
} from "../../app/coreSimulationConfig";

export type MouseDaimonMode = "push" | "pull" | "orbit" | "calm";

export interface MouseDaimonTuning {
  enabled: boolean;
  message: string;
  mode: MouseDaimonMode;
  radius: number;
  strength: number;
}

interface ProjectionOption {
  id: string;
  name: string;
  description: string;
}

interface Props {
  projectionPerspective: string;
  autopilotEnabled: boolean;
  autopilotStatus: "running" | "waiting" | "stopped";
  autopilotSummary: string;
  interfaceOpacity: number;
  coreCameraZoom: number;
  coreCameraPitch: number;
  coreCameraYaw: number;
  coreRenderedCameraPosition: { x: number; y: number; z: number };
  coreFlightEnabled: boolean;
  coreFlightSpeed: number;
  coreOrbitEnabled: boolean;
  coreOrbitSpeed: number;
  coreSimulationTuning: CoreSimulationTuning;
  coreVisualTuning: CoreVisualTuning;
  coreOverlayView: OverlayViewId;
  activeChatLens: { presence: string; status: string } | null;
  latestAutopilotEvent: { actionId: string; result: string } | null;
  projectionOptions: ProjectionOption[];
  mouseDaimonTuning: MouseDaimonTuning;
  onToggleAutopilot: () => void;
  onToggleCoreFlight: () => void;
  onToggleCoreOrbit: () => void;
  onNudgeCoreFlightSpeed: (delta: number) => void;
  onNudgeCoreOrbitSpeed: (delta: number) => void;
  onApplyCoreLayerPreset: (view: OverlayViewId) => void;
  onNudgeCoreZoom: (delta: number) => void;
  onResetCoreCamera: () => void;
  onSelectPerspective: (perspective: UIPerspective) => void;
  onSetInterfaceOpacity: (value: number) => void;
  onResetInterfaceOpacity: () => void;
  onBoostCoreVisibility: () => void;
  onResetCoreVisualTuning: () => void;
  onSetCoreVisualDial: (dial: keyof CoreVisualTuning, value: number) => void;
  onResetCoreSimulationTuning: () => void;
  onSetCoreSimulationDial: (dial: keyof CoreSimulationTuning, value: number) => void;
  onSetCoreOrbitSpeed: (value: number) => void;
  onSetMouseDaimonTuning: (tuning: Partial<MouseDaimonTuning>) => void;
  onOpenRuntimeConfig: () => void;
}

const INTERFACE_OPACITY_MIN = 0.38;
const INTERFACE_OPACITY_MAX = 1;

export function CoreControlPanel({
  projectionPerspective,
  autopilotEnabled,
  autopilotStatus,
  autopilotSummary,
  interfaceOpacity,
  coreCameraZoom,
  coreCameraPitch,
  coreCameraYaw,
  coreRenderedCameraPosition,
  coreFlightEnabled,
  coreFlightSpeed,
  coreOrbitEnabled,
  coreOrbitSpeed,
  coreSimulationTuning,
  coreVisualTuning,
  coreOverlayView,
  activeChatLens,
  latestAutopilotEvent,
  projectionOptions,
  mouseDaimonTuning,
  onToggleAutopilot,
  onToggleCoreFlight,
  onToggleCoreOrbit,
  onNudgeCoreFlightSpeed,
  onNudgeCoreOrbitSpeed,
  onApplyCoreLayerPreset,
  onNudgeCoreZoom,
  onResetCoreCamera,
  onSelectPerspective,
  onSetInterfaceOpacity,
  onResetInterfaceOpacity,
  onBoostCoreVisibility,
  onResetCoreVisualTuning,
  onSetCoreVisualDial,
  onResetCoreSimulationTuning,
  onSetCoreSimulationDial,
  onSetCoreOrbitSpeed,
  onSetMouseDaimonTuning,
  onOpenRuntimeConfig,
}: Props) {
  const [showVisualDials, setShowVisualDials] = useState(false);
  const [showSimulationControls, setShowSimulationControls] = useState(false);
  const interfaceTransparencyPercent = Math.round((1 - interfaceOpacity) * 100);

  return (
    <div className="grid gap-2">
      <div className="text-[10px] text-muted space-y-0.5 font-mono opacity-70">
        <div className="flex flex-wrap gap-x-3 gap-y-1">
          <span>perspective: <code>{projectionPerspective}</code></span>
          <span>autopilot: <code>{autopilotEnabled ? autopilotStatus : "stopped"}</code></span>
          <span className="opacity-80">note: <code>{autopilotSummary}</code></span>
          <span>interface: <code>{Math.round(interfaceOpacity * 100)}%</code> opacity</span>
        </div>
        <div className="flex flex-wrap gap-x-3 gap-y-1">
          <span>
            core-camera: <code>{coreCameraZoom.toFixed(2)}x</code> / pitch
            <code>{coreCameraPitch.toFixed(0)}deg</code> / yaw
            <code>{coreCameraYaw.toFixed(0)}deg</code> / xyz
            <code>{coreRenderedCameraPosition.x.toFixed(0)}</code>,
            <code>{coreRenderedCameraPosition.y.toFixed(0)}</code>,
            <code>{coreRenderedCameraPosition.z.toFixed(0)}</code>
          </span>
          <span>
            flight: <code>{coreFlightEnabled ? "armed" : "paused"}</code> speed
            <code>{coreFlightSpeed.toFixed(2)}x</code>
          </span>
          <span>
            orbit: <code>{coreOrbitEnabled ? "active" : "off"}</code> speed
            <code>{coreOrbitSpeed.toFixed(2)}x</code>
          </span>
          <span>
            particles: <code>{Math.round(coreSimulationTuning.particleDensity * 100)}%</code> scale
            <code>{coreSimulationTuning.particleScale.toFixed(2)}x</code> motion
            <code>{coreSimulationTuning.motionSpeed.toFixed(2)}x</code> mouse
            <code>{coreSimulationTuning.mouseInfluence.toFixed(2)}x</code> depth
            <code>{coreSimulationTuning.layerDepth.toFixed(2)}x</code>
          </span>
          {activeChatLens ? (
            <span>chat-lens: <code>{activeChatLens.presence}</code> ({activeChatLens.status})</span>
          ) : null}
          {latestAutopilotEvent ? (
            <span>last: <code>{latestAutopilotEvent.actionId}</code> ({latestAutopilotEvent.result})</span>
          ) : null}
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <button
          type="button"
          onClick={onToggleAutopilot}
          className={`border rounded px-2 py-0.5 text-[10px] font-semibold transition-colors ${
            autopilotEnabled
              ? "bg-[rgba(166,226,46,0.16)] border-[rgba(166,226,46,0.48)] text-[#a6e22e]"
              : "bg-[rgba(249,38,114,0.16)] border-[rgba(249,38,114,0.48)] text-[#f92672]"
          }`}
        >
          {autopilotEnabled ? "Autopilot On" : "Autopilot Off"}
        </button>

        <button
          type="button"
          onClick={onToggleCoreFlight}
          className={`border rounded px-2 py-0.5 text-[10px] font-semibold transition-colors ${
            coreFlightEnabled
              ? "bg-[rgba(122,214,255,0.18)] border-[rgba(122,214,255,0.52)] text-[#9de3ff]"
              : "bg-[rgba(180,180,180,0.14)] border-[rgba(182,182,182,0.34)] text-[#d2d7de]"
          }`}
        >
          {coreFlightEnabled ? "Flight Armed" : "Flight Paused"}
        </button>

        <button
          type="button"
          onClick={onToggleCoreOrbit}
          className={`border rounded px-2 py-0.5 text-[10px] font-semibold transition-colors ${
            coreOrbitEnabled
              ? "bg-[rgba(118,227,255,0.22)] border-[rgba(118,227,255,0.52)] text-[#9feaff]"
              : "bg-[rgba(180,180,180,0.14)] border-[rgba(182,182,182,0.34)] text-[#d2d7de]"
          }`}
        >
          {coreOrbitEnabled ? "Orbit On" : "Orbit Off"}
        </button>

        <div className="flex items-center gap-1 border rounded px-1 py-0.5 text-[10px] bg-[rgba(10,22,34,0.22)] border-[rgba(120,178,221,0.28)]">
          <button type="button" onClick={() => onNudgeCoreFlightSpeed(-0.12)} className="px-1 text-[#bdd9f2]">thrust-</button>
          <button type="button" onClick={() => onNudgeCoreFlightSpeed(0.12)} className="px-1 text-[#9ed6f8]">thrust+</button>
        </div>

        <div className="flex items-center gap-1 border rounded px-1 py-0.5 text-[10px] bg-[rgba(10,22,34,0.22)] border-[rgba(120,178,221,0.28)]">
          <button type="button" onClick={() => onNudgeCoreOrbitSpeed(-0.08)} className="px-1 text-[#bdd9f2]">orbit-</button>
          <button type="button" onClick={() => onNudgeCoreOrbitSpeed(0.08)} className="px-1 text-[#9ed6f8]">orbit+</button>
        </div>

        <select
          value={coreOverlayView}
          onChange={(event) => onApplyCoreLayerPreset(event.target.value as OverlayViewId)}
          className="border rounded px-2 py-0.5 text-[10px] font-semibold bg-[rgba(10,22,34,0.24)] text-[#9dd5f8] border-[rgba(120,178,221,0.32)]"
          title="simulation-core layer preset"
        >
          {OVERLAY_VIEW_OPTIONS.map((option) => (
            <option key={option.id} value={option.id}>
              core:{option.label}
            </option>
          ))}
        </select>

        <div className="flex items-center gap-1 border rounded px-1 py-0.5 text-[10px] bg-[rgba(10,22,34,0.22)] border-[rgba(120,178,221,0.28)]">
          <button type="button" onClick={() => onNudgeCoreZoom(-0.08)} className="px-1 text-[#9ed6f8]">-</button>
          <button type="button" onClick={() => onNudgeCoreZoom(0.08)} className="px-1 text-[#9ed6f8]">+</button>
          <button type="button" onClick={onResetCoreCamera} className="px-1 text-[#f3d9b8]">reset</button>
        </div>

        <button
          type="button"
          onClick={() => setShowVisualDials((prev) => !prev)}
          className="border rounded px-2 py-0.5 text-[10px] font-semibold transition-colors bg-[rgba(11,24,36,0.24)] text-[#cbe9ff] border-[rgba(120,178,221,0.32)]"
        >
          {showVisualDials ? "hide dials" : "show dials"}
        </button>

        <button
          type="button"
          onClick={() => setShowSimulationControls((prev) => !prev)}
          className="border rounded px-2 py-0.5 text-[10px] font-semibold transition-colors bg-[rgba(11,24,36,0.24)] text-[#cbe9ff] border-[rgba(120,178,221,0.32)]"
        >
          {showSimulationControls ? "hide sim" : "show sim"}
        </button>

        <button
          type="button"
          onClick={onOpenRuntimeConfig}
          className="border rounded px-2 py-0.5 text-[10px] font-semibold transition-colors bg-[rgba(174,129,255,0.2)] text-[#efe2ff] border-[rgba(174,129,255,0.48)]"
          title="open runtime config interface"
        >
          runtime config
        </button>

        {projectionOptions.map((option) => (
          <button
            key={option.id}
            type="button"
            onClick={() => onSelectPerspective(option.id as UIPerspective)}
            className={`border rounded px-2 py-0.5 text-[10px] font-semibold transition-colors ${
              projectionPerspective === option.id
                ? "bg-[rgba(102,217,239,0.2)] text-[#66d9ef] border-[rgba(102,217,239,0.7)]"
                : "bg-[rgba(39,40,34,0.78)] text-[var(--ink)] border-[var(--line)] hover:bg-[rgba(55,56,48,0.92)]"
            }`}
            title={option.description}
          >
            {option.name}
          </button>
        ))}
      </div>

      {showVisualDials || showSimulationControls ? (
        <div className="rounded-lg border border-[rgba(122,184,226,0.28)] bg-[rgba(9,18,28,0.18)] px-2 py-2">
          <div className="rounded-md border border-[rgba(122,184,226,0.24)] bg-[rgba(8,16,24,0.14)] px-2 py-2">
            <div className="flex items-center justify-between gap-2">
              <p className="text-[10px] uppercase tracking-[0.12em] text-[#9dd5f8]">interface transparency</p>
              <button
                type="button"
                onClick={onResetInterfaceOpacity}
                className="rounded border border-[rgba(178,205,228,0.36)] px-2 py-0.5 text-[10px] font-semibold text-[#d5e9fb] hover:bg-[rgba(102,154,196,0.2)]"
              >
                reset ui
              </button>
            </div>
            <label className="mt-2 grid gap-1">
              <span className="text-[10px] text-[#cde4f8]">
                transparency <code>{interfaceTransparencyPercent}%</code>
              </span>
              <input
                type="range"
                min={INTERFACE_OPACITY_MIN}
                max={INTERFACE_OPACITY_MAX}
                step={0.01}
                value={interfaceOpacity}
                onChange={(event) => onSetInterfaceOpacity(Number(event.target.value))}
                className="h-1 w-full accent-[#8ed8ff]"
              />
            </label>
          </div>

          {showVisualDials ? (
            <>
              <div className="mt-3 flex items-center justify-between gap-2">
                <p className="text-[10px] uppercase tracking-[0.12em] text-[#9dd5f8]">simulation dials</p>
                <div className="flex items-center gap-1">
                  <button
                    type="button"
                    onClick={onBoostCoreVisibility}
                    className="rounded border border-[rgba(180,235,197,0.44)] px-2 py-0.5 text-[10px] font-semibold text-[#d4f5dc] hover:bg-[rgba(88,170,118,0.24)]"
                  >
                    boost visibility
                  </button>
                  <button
                    type="button"
                    onClick={onResetCoreVisualTuning}
                    className="rounded border border-[rgba(178,205,228,0.36)] px-2 py-0.5 text-[10px] font-semibold text-[#d5e9fb] hover:bg-[rgba(102,154,196,0.2)]"
                  >
                    reset look
                  </button>
                </div>
              </div>

              <div className="mt-2 grid gap-2 sm:grid-cols-2">
                <label className="grid gap-1">
                  <span className="text-[10px] text-[#cde4f8]">brightness <code>{coreVisualTuning.brightness.toFixed(2)}</code></span>
                  <input
                    type="range"
                    min={CORE_VISUAL_BRIGHTNESS_MIN}
                    max={CORE_VISUAL_BRIGHTNESS_MAX}
                    step={0.01}
                    value={coreVisualTuning.brightness}
                    onChange={(event) => onSetCoreVisualDial("brightness", Number(event.target.value))}
                    className="h-1 w-full accent-[#8ed8ff]"
                  />
                </label>

                <label className="grid gap-1">
                  <span className="text-[10px] text-[#cde4f8]">contrast <code>{coreVisualTuning.contrast.toFixed(2)}</code></span>
                  <input
                    type="range"
                    min={CORE_VISUAL_CONTRAST_MIN}
                    max={CORE_VISUAL_CONTRAST_MAX}
                    step={0.01}
                    value={coreVisualTuning.contrast}
                    onChange={(event) => onSetCoreVisualDial("contrast", Number(event.target.value))}
                    className="h-1 w-full accent-[#8ed8ff]"
                  />
                </label>

                <label className="grid gap-1">
                  <span className="text-[10px] text-[#cde4f8]">saturation <code>{coreVisualTuning.saturation.toFixed(2)}</code></span>
                  <input
                    type="range"
                    min={CORE_VISUAL_SATURATION_MIN}
                    max={CORE_VISUAL_SATURATION_MAX}
                    step={0.01}
                    value={coreVisualTuning.saturation}
                    onChange={(event) => onSetCoreVisualDial("saturation", Number(event.target.value))}
                    className="h-1 w-full accent-[#8ed8ff]"
                  />
                </label>

                <label className="grid gap-1">
                  <span className="text-[10px] text-[#cde4f8]">hue shift <code>{coreVisualTuning.hueRotate.toFixed(0)}deg</code></span>
                  <input
                    type="range"
                    min={CORE_VISUAL_HUE_MIN}
                    max={CORE_VISUAL_HUE_MAX}
                    step={1}
                    value={coreVisualTuning.hueRotate}
                    onChange={(event) => onSetCoreVisualDial("hueRotate", Number(event.target.value))}
                    className="h-1 w-full accent-[#8ed8ff]"
                  />
                </label>

                <label className="grid gap-1">
                  <span className="text-[10px] text-[#cde4f8]">field dim <code>{coreVisualTuning.backgroundWash.toFixed(2)}</code></span>
                  <input
                    type="range"
                    min={CORE_VISUAL_WASH_MIN}
                    max={CORE_VISUAL_WASH_MAX}
                    step={0.01}
                    value={coreVisualTuning.backgroundWash}
                    onChange={(event) => onSetCoreVisualDial("backgroundWash", Number(event.target.value))}
                    className="h-1 w-full accent-[#8ed8ff]"
                  />
                </label>

                <label className="grid gap-1">
                  <span className="text-[10px] text-[#cde4f8]">edge vignette <code>{coreVisualTuning.vignette.toFixed(2)}x</code></span>
                  <input
                    type="range"
                    min={CORE_VISUAL_VIGNETTE_MIN}
                    max={CORE_VISUAL_VIGNETTE_MAX}
                    step={0.01}
                    value={coreVisualTuning.vignette}
                    onChange={(event) => onSetCoreVisualDial("vignette", Number(event.target.value))}
                    className="h-1 w-full accent-[#8ed8ff]"
                  />
                </label>
              </div>
            </>
          ) : null}

          {showSimulationControls ? (
            <div className={`${showVisualDials ? "mt-3" : ""} rounded-md border border-[rgba(116,176,216,0.24)] bg-[rgba(8,16,24,0.14)] px-2 py-2`}>
              <div className="flex items-center justify-between gap-2">
                <p className="text-[10px] uppercase tracking-[0.12em] text-[#9dd5f8]">simulation controls</p>
                <button
                  type="button"
                  onClick={onResetCoreSimulationTuning}
                  className="rounded border border-[rgba(178,205,228,0.36)] px-2 py-0.5 text-[10px] font-semibold text-[#d5e9fb] hover:bg-[rgba(102,154,196,0.2)]"
                >
                  reset sim
                </button>
              </div>

              <div className="mt-2 grid gap-2 sm:grid-cols-2">
                <label className="grid gap-1">
                  <span className="text-[10px] text-[#cde4f8]">particles <code>{Math.round(coreSimulationTuning.particleDensity * 100)}%</code></span>
                  <input
                    type="range"
                    min={CORE_SIM_PARTICLE_DENSITY_MIN}
                    max={CORE_SIM_PARTICLE_DENSITY_MAX}
                    step={0.01}
                    value={coreSimulationTuning.particleDensity}
                    onChange={(event) => onSetCoreSimulationDial("particleDensity", Number(event.target.value))}
                    className="h-1 w-full accent-[#8ed8ff]"
                  />
                </label>

                <label className="grid gap-1">
                  <span className="text-[10px] text-[#cde4f8]">particle size <code>{coreSimulationTuning.particleScale.toFixed(2)}x</code></span>
                  <input
                    type="range"
                    min={CORE_SIM_PARTICLE_SCALE_MIN}
                    max={CORE_SIM_PARTICLE_SCALE_MAX}
                    step={0.01}
                    value={coreSimulationTuning.particleScale}
                    onChange={(event) => onSetCoreSimulationDial("particleScale", Number(event.target.value))}
                    className="h-1 w-full accent-[#8ed8ff]"
                  />
                </label>

                <label className="grid gap-1">
                  <span className="text-[10px] text-[#cde4f8]">field motion <code>{coreSimulationTuning.motionSpeed.toFixed(2)}x</code></span>
                  <input
                    type="range"
                    min={CORE_SIM_MOTION_SPEED_MIN}
                    max={CORE_SIM_MOTION_SPEED_MAX}
                    step={0.01}
                    value={coreSimulationTuning.motionSpeed}
                    onChange={(event) => onSetCoreSimulationDial("motionSpeed", Number(event.target.value))}
                    className="h-1 w-full accent-[#8ed8ff]"
                  />
                </label>

                <label className="grid gap-1">
                  <span className="text-[10px] text-[#cde4f8]">mouse influence <code>{coreSimulationTuning.mouseInfluence.toFixed(2)}x</code></span>
                  <input
                    type="range"
                    min={CORE_SIM_MOUSE_INFLUENCE_MIN}
                    max={CORE_SIM_MOUSE_INFLUENCE_MAX}
                    step={0.01}
                    value={coreSimulationTuning.mouseInfluence}
                    onChange={(event) => onSetCoreSimulationDial("mouseInfluence", Number(event.target.value))}
                    className="h-1 w-full accent-[#8ed8ff]"
                  />
                </label>

                <label className="grid gap-1">
                  <span className="text-[10px] text-[#cde4f8]">layer depth <code>{coreSimulationTuning.layerDepth.toFixed(2)}x</code></span>
                  <input
                    type="range"
                    min={CORE_SIM_LAYER_DEPTH_MIN}
                    max={CORE_SIM_LAYER_DEPTH_MAX}
                    step={0.01}
                    value={coreSimulationTuning.layerDepth}
                    onChange={(event) => onSetCoreSimulationDial("layerDepth", Number(event.target.value))}
                    className="h-1 w-full accent-[#8ed8ff]"
                  />
                </label>

                <label className="grid gap-1">
                  <span className="text-[10px] text-[#cde4f8]">orbit speed <code>{coreOrbitSpeed.toFixed(2)}x</code></span>
                  <input
                    type="range"
                    min={CORE_ORBIT_SPEED_MIN}
                    max={CORE_ORBIT_SPEED_MAX}
                    step={0.01}
                    value={coreOrbitSpeed}
                    onChange={(event) => onSetCoreOrbitSpeed(Number(event.target.value))}
                    className="h-1 w-full accent-[#8ed8ff]"
                  />
                </label>
              </div>
            </div>
          ) : null}

          <div className={`${showSimulationControls ? "mt-3" : ""} rounded-md border border-[rgba(255,160,80,0.28)] bg-[rgba(40,28,16,0.12)] px-2 py-2`}>
            <div className="flex items-center justify-between gap-2">
              <p className="text-[10px] uppercase tracking-[0.12em] text-[#ffc878]">mouse daimon</p>
              <button
                type="button"
                onClick={() => onSetMouseDaimonTuning({ enabled: !mouseDaimonTuning.enabled })}
                className={`rounded border px-2 py-0.5 text-[10px] font-semibold transition-colors ${
                  mouseDaimonTuning.enabled
                    ? "border-[rgba(255,180,100,0.5)] bg-[rgba(255,160,80,0.24)] text-[#ffd4a0]"
                    : "border-[rgba(140,120,100,0.36)] bg-[rgba(60,50,40,0.24)] text-[#c8b090]"
                }`}
                >
                  {mouseDaimonTuning.enabled ? "enabled" : "disabled"}
                </button>
              </div>

              <div className="mt-2 grid gap-2 sm:grid-cols-2">
                <label className="grid gap-1">
                  <span className="text-[10px] text-[#e8c8a0]">message</span>
                  <input
                    type="text"
                    value={mouseDaimonTuning.message}
                    onChange={(e) => onSetMouseDaimonTuning({ message: e.target.value })}
                    maxLength={24}
                    className="rounded border border-[rgba(255,180,100,0.3)] bg-[rgba(40,30,20,0.5)] px-2 py-1 text-[11px] text-[#ffe4c4] placeholder-[rgba(200,160,120,0.5)] focus:border-[rgba(255,200,120,0.5)] focus:outline-none"
                    placeholder="witness"
                  />
                </label>

                <div className="grid gap-1">
                  <span className="text-[10px] text-[#e8c8a0]">mode</span>
                  <div className="flex flex-wrap gap-1">
                    {(["push", "pull", "orbit", "calm"] as const).map((mode) => (
                      <button
                        key={mode}
                        type="button"
                        onClick={() => onSetMouseDaimonTuning({ mode })}
                        className={`rounded px-2 py-0.5 text-[9px] font-semibold transition-colors ${
                          mouseDaimonTuning.mode === mode
                            ? "bg-[rgba(255,160,80,0.4)] text-[#fff4e0]"
                            : "text-[#d4b890] hover:bg-[rgba(255,140,60,0.2)]"
                        }`}
                      >
                        {mode}
                      </button>
                    ))}
                  </div>
                </div>

                <label className="grid gap-1">
                  <span className="text-[10px] text-[#e8c8a0]">radius <code>{Math.round(mouseDaimonTuning.radius * 100)}%</code></span>
                  <input
                    type="range"
                    min={0.06}
                    max={0.42}
                    step={0.02}
                    value={mouseDaimonTuning.radius}
                    onChange={(e) => onSetMouseDaimonTuning({ radius: Number(e.target.value) })}
                    className="h-1 w-full accent-[#ffaa55]"
                  />
                </label>

                <label className="grid gap-1">
                  <span className="text-[10px] text-[#e8c8a0]">strength <code>{Math.round(mouseDaimonTuning.strength * 100)}%</code></span>
                  <input
                    type="range"
                    min={0.1}
                    max={0.9}
                    step={0.05}
                    value={mouseDaimonTuning.strength}
                    onChange={(e) => onSetMouseDaimonTuning({ strength: Number(e.target.value) })}
                    className="h-1 w-full accent-[#ffaa55]"
                  />
                </label>
              </div>
            </div>
        </div>
      ) : null}
    </div>
  );
}
