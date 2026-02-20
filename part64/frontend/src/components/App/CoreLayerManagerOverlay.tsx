import {
  CORE_LAYER_OPTIONS,
  type CoreLayerId,
} from "../../app/coreSimulationConfig";

interface Props {
  activeLayerCount: number;
  inline?: boolean;
  isOpen: boolean;
  layerVisibility: Record<CoreLayerId, boolean>;
  onToggleOpen: () => void;
  onSetAllLayers: (enabled: boolean) => void;
  onSetLayerEnabled: (layerId: CoreLayerId, enabled: boolean) => void;
}

export function CoreLayerManagerOverlay({
  activeLayerCount,
  inline = false,
  isOpen,
  layerVisibility,
  onToggleOpen,
  onSetAllLayers,
  onSetLayerEnabled,
}: Props) {
  const shellClassName = inline
    ? "w-full"
    : "pointer-events-none fixed top-24 right-2 z-[70] w-[min(92vw,19rem)]";
  const cardClassName = inline
    ? "pointer-events-auto rounded-xl border border-[rgba(130,190,232,0.32)] bg-[linear-gradient(170deg,rgba(7,18,29,0.18),rgba(10,24,38,0.14))] p-2 shadow-[0_6px_12px_rgba(0,7,14,0.16)]"
    : "pointer-events-auto rounded-xl border border-[rgba(130,190,232,0.36)] bg-[linear-gradient(170deg,rgba(7,18,29,0.22),rgba(10,24,38,0.16))] p-2 shadow-[0_8px_18px_rgba(0,7,14,0.22)]";

  return (
    <div className={shellClassName}>
      <section className={cardClassName}>
        <header className="flex items-center justify-between gap-2">
          <div>
            <p className="text-[10px] uppercase tracking-[0.12em] text-[#a4dcff]">layers manager</p>
            <p className="text-[10px] text-[#c7e8ff]">active <code>{activeLayerCount}</code>/<code>{CORE_LAYER_OPTIONS.length}</code></p>
          </div>
          <button
            type="button"
            onClick={onToggleOpen}
            className="rounded border border-[rgba(157,204,236,0.36)] px-2 py-0.5 text-[10px] font-semibold text-[#cde7fa] hover:bg-[rgba(97,151,191,0.24)]"
          >
            {isOpen ? "hide" : "show"}
          </button>
        </header>

        {isOpen ? (
          <div className="mt-2 space-y-1.5">
            <div className="flex items-center gap-1">
              <button
                type="button"
                onClick={() => onSetAllLayers(true)}
                className="rounded border border-[rgba(146,224,184,0.42)] px-2 py-0.5 text-[10px] font-semibold text-[#bcf5d7] hover:bg-[rgba(84,156,116,0.24)]"
              >
                all on
              </button>
              <button
                type="button"
                onClick={() => onSetAllLayers(false)}
                className="rounded border border-[rgba(236,170,160,0.42)] px-2 py-0.5 text-[10px] font-semibold text-[#ffd5ca] hover:bg-[rgba(184,108,89,0.24)]"
              >
                all off
              </button>
            </div>

            <div className="grid gap-1">
              {CORE_LAYER_OPTIONS.map((layer) => (
                <label
                  key={layer.id}
                  className="flex items-center justify-between gap-2 rounded-md border border-[rgba(112,172,213,0.28)] bg-[rgba(9,20,32,0.6)] px-2 py-1"
                >
                  <span className="text-[10px] text-[#d6ecff]">{layer.label}</span>
                  <input
                    type="checkbox"
                    checked={layerVisibility[layer.id]}
                    onChange={(event) => onSetLayerEnabled(layer.id, event.target.checked)}
                    className="h-3.5 w-3.5 accent-[#8fd8ff]"
                  />
                </label>
              ))}
            </div>
          </div>
        ) : null}
      </section>
    </div>
  );
}
