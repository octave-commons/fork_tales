import { useState, useEffect } from "react";
import { runtimeApiUrl } from "../../runtime/endpoints";
import type { Catalog } from "../../types";

interface Props {
  catalog: Catalog | null;
}

interface Memory {
  id?: string;
  text: string;
  metadata?: {
    timestamp?: string;
  };
}

export function OmniPanel({ catalog }: Props) {
  const [memories, setMemories] = useState<Memory[]>([]);

  useEffect(() => {
    const fetchMemories = async () => {
      try {
        const res = await fetch(runtimeApiUrl("/api/memories"));
        if (res.ok) {
          const data = (await res.json()) as { memories?: Memory[] };
          setMemories(data.memories || []);
        }
      } catch {
        // ignore
      }
    };
    fetchMemories();
    const interval = setInterval(fetchMemories, 10000);
    return () => clearInterval(interval);
  }, []);

  if (!catalog || !catalog.cover_fields) return null;

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-[repeat(auto-fill,minmax(220px,1fr))] gap-3 mt-3">
        {catalog.cover_fields.map((cover) => (
          <article 
            key={cover.id}
            className="border border-[var(--line)] rounded-xl p-3 bg-gradient-to-b from-[rgba(45,46,39,0.9)] to-[rgba(31,32,29,0.94)]"
          >
            <div className="flex justify-between items-baseline mb-2">
              <strong className="text-sm font-semibold">{cover.display_name.en}</strong>
              <span className="text-xs text-muted">{cover.display_name.ja}</span>
            </div>
            
            <img 
              src={cover.url} 
              alt={cover.display_name.en}
              className="w-full rounded-lg block" 
              loading="lazy"
            />
            
            <p className="text-xs text-muted mt-2">Part {cover.part}</p>
          </article>
        ))}
      </div>

      {memories.length > 0 && (
        <div className="card !mt-6 bg-[rgba(166,226,46,0.12)] border-[rgba(166,226,46,0.36)]">
            <h4 className="text-sm font-bold mb-2 text-[#a6e22e]">Memory Fragments / 記憶の断片 (Echoes)</h4>
            <div className="space-y-2">
                {memories.map((m, i) => (
                    <div key={m.id || i} className="text-xs border-l-2 border-[rgba(166,226,46,0.5)] pl-2 py-1">
                        <span className="text-muted block font-mono">[{m.metadata?.timestamp || "???"}]</span>
                        {m.text}
                    </div>
                ))}
            </div>
        </div>
      )}
    </div>
  );
}
