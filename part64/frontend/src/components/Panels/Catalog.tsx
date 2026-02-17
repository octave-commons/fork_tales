import { useEffect, useMemo, useState } from "react";
import type { Catalog } from "../../types";
import {
  Archive,
  Eye,
  FileText,
  FolderTree,
  Image as ImageIcon,
  Play,
  Video,
} from "lucide-react";

interface Props {
  catalog: Catalog | null;
}

const MAX_VISIBLE_ITEMS = 24;

interface ZipMemberSummary {
  path: string;
  kind: string;
  ext: string;
  depth: number;
  is_dir: boolean;
  bytes: number;
  compressed_bytes: number;
  url: string;
}

interface ZipExtSummary {
  ext: string;
  count: number;
}

interface ZipTopLevelSummary {
  name: string;
  count: number;
}

interface ZipSummary {
  id: string;
  name: string;
  rel_path: string;
  url: string;
  bytes: number;
  mtime_utc: string;
  members_total: number;
  files_total: number;
  dirs_total: number;
  uncompressed_bytes_total: number;
  compressed_bytes_total: number;
  compression_ratio: number;
  members_truncated: boolean;
  error?: string;
  type_counts: Record<string, number>;
  extension_counts: ZipExtSummary[];
  top_level_entries: ZipTopLevelSummary[];
  members: ZipMemberSummary[];
}

interface ZipCatalogResponse {
  ok: boolean;
  generated_at: string;
  member_limit: number;
  zip_count: number;
  zips: ZipSummary[];
}

const MEMBER_KIND_COLORS: Record<string, string> = {
  audio: "#a6e22e",
  image: "#66d9ef",
  video: "#fd971f",
  text: "#e6db74",
  file: "#ae81ff",
  dir: "#75715e",
};

function formatBytes(bytes: number): string {
  const value = Number.isFinite(bytes) ? Math.max(0, bytes) : 0;
  if (value < 1024) {
    return `${value} B`;
  }
  const units = ["KB", "MB", "GB", "TB"];
  let amount = value / 1024;
  let unit = units[0];
  for (let index = 1; index < units.length && amount >= 1024; index += 1) {
    amount /= 1024;
    unit = units[index];
  }
  return `${amount.toFixed(amount >= 10 ? 1 : 2)} ${unit}`;
}

export function CatalogPanel({ catalog }: Props) {
  const [expanded, setExpanded] = useState(false);
  const [zipCatalog, setZipCatalog] = useState<ZipCatalogResponse | null>(null);
  const [zipLoading, setZipLoading] = useState(false);
  const [zipError, setZipError] = useState("");
  const [zipFilter, setZipFilter] = useState("");

  const fetchZipCatalog = useMemo(
    () => async () => {
      setZipLoading(true);
      setZipError("");
      try {
        const baseUrl = window.location.port === "5173" ? "http://127.0.0.1:8787" : "";
        const response = await fetch(`${baseUrl}/api/zips?member_limit=240`);
        if (!response.ok) {
          throw new Error(`zip catalog failed: ${response.status}`);
        }
        const payload = (await response.json()) as ZipCatalogResponse;
        if (!payload || !Array.isArray(payload.zips)) {
          throw new Error("invalid zip catalog payload");
        }
        setZipCatalog(payload);
      } catch (error) {
        setZipError(error instanceof Error ? error.message : "zip catalog unavailable");
      } finally {
        setZipLoading(false);
      }
    },
    [],
  );

  useEffect(() => {
    void fetchZipCatalog();
  }, [fetchZipCatalog]);

  if (!catalog || !catalog.items) return null;

  const items = catalog.items.filter((item) => item.role !== "cover_art");
  const visibleItems = expanded ? items : items.slice(0, MAX_VISIBLE_ITEMS);
  const hiddenCount = Math.max(0, items.length - visibleItems.length);
  const normalizedFilter = zipFilter.trim().toLowerCase();
  const zipRows = (zipCatalog?.zips ?? []).filter((zip) => {
    if (!normalizedFilter) {
      return true;
    }
    const zipMatch =
      zip.name.toLowerCase().includes(normalizedFilter) ||
      zip.rel_path.toLowerCase().includes(normalizedFilter);
    if (zipMatch) {
      return true;
    }
    return zip.members.some((member) => member.path.toLowerCase().includes(normalizedFilter));
  });
  const totalZipFiles = zipRows.reduce((sum, zip) => sum + zip.files_total, 0);
  const totalZipEntries = zipRows.reduce((sum, zip) => sum + zip.members_total, 0);

  const getIcon = (kind: string) => {
    switch (kind) {
      case "audio":
        return <Play size={16} />;
      case "image":
        return <ImageIcon size={16} />;
      case "video":
        return <Video size={16} />;
      default:
        return <FileText size={16} />;
    }
  };

  return (
    <div className="mt-3 space-y-4">
      <section className="rounded-xl border border-[var(--line)] bg-[rgba(39,40,34,0.84)] p-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="text-sm font-semibold flex items-center gap-2">
              <Archive size={16} />
              Zip Atlas / 圧縮アーカイブ
            </p>
            <p className="text-xs text-muted mt-1">
              Complete zip inventory with member map and type distribution.
            </p>
          </div>
          <p className="text-xs text-muted font-mono">
            zips <code>{zipRows.length}</code> | files <code>{totalZipFiles}</code> | members <code>{totalZipEntries}</code>
          </p>
        </div>

        <div className="mt-3 grid gap-2 md:grid-cols-[1fr_auto] md:items-center">
          <input
            value={zipFilter}
            onChange={(event) => setZipFilter(event.target.value)}
            placeholder="filter by zip name, path, or member"
            className="rounded-md border border-[var(--line)] bg-[rgba(31,32,29,0.88)] px-3 py-2 text-xs text-ink"
          />
          <button
            type="button"
            className="border border-[var(--line)] rounded-md bg-[rgba(39,40,34,0.9)] px-3 py-2 text-xs font-semibold text-ink hover:bg-[rgba(55,56,48,0.94)]"
            onClick={() => {
              void fetchZipCatalog();
            }}
          >
            Refresh Zips
          </button>
        </div>

        {zipLoading ? <p className="text-xs text-muted mt-2">loading zip inventory...</p> : null}
        {zipError ? <p className="text-xs text-[#f92672] mt-2">{zipError}</p> : null}

        {!zipLoading && !zipError && zipRows.length === 0 ? (
          <p className="text-xs text-muted mt-2">No zip archives matched this filter.</p>
        ) : null}

        <div className="mt-3 space-y-3 max-h-[42rem] overflow-auto pr-1">
          {zipRows.map((zip, index) => {
            const typeRows = Object.entries(zip.type_counts).sort((a, b) => b[1] - a[1]);
            const topExtensions = zip.extension_counts.slice(0, 7);
            const topFolders = zip.top_level_entries.slice(0, 6);
            const compressionPct = Math.round(zip.compression_ratio * 100);
            return (
              <details
                key={zip.id}
                className="rounded-lg border border-[var(--line)] bg-[rgba(31,32,29,0.84)] p-3"
                open={index === 0}
              >
                <summary className="list-none cursor-pointer">
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div className="min-w-0">
                      <p className="text-sm font-semibold truncate">{zip.name}</p>
                      <p className="text-[11px] text-muted font-mono truncate">{zip.rel_path}</p>
                    </div>
                    <div className="text-right text-[11px] text-muted font-mono">
                      <p>{formatBytes(zip.bytes)} zip</p>
                      <p>
                        {zip.files_total} files / {zip.dirs_total} dirs
                      </p>
                      <p>compression {compressionPct}%</p>
                    </div>
                  </div>
                  <div className="mt-2 h-2 rounded-full overflow-hidden bg-bg-1 flex">
                    {typeRows.map(([kind, count]) => {
                      const widthPct = (count / Math.max(1, zip.members_total)) * 100;
                      const color = MEMBER_KIND_COLORS[kind] ?? "#f8f8f2";
                      return (
                        <span
                          key={`${zip.id}-${kind}`}
                          style={{ width: `${widthPct}%`, backgroundColor: color }}
                          title={`${kind}: ${count}`}
                        />
                      );
                    })}
                  </div>
                </summary>

                {zip.error ? (
                  <p className="text-xs text-[#f92672] mt-2">{zip.error}</p>
                ) : (
                  <div className="mt-3 grid gap-3 lg:grid-cols-[1.5fr_1fr]">
                    <div className="rounded-md border border-[var(--line)] bg-[rgba(39,40,34,0.78)] p-2">
                      <p className="text-[11px] uppercase tracking-wide text-muted mb-2">
                        Member Map / 内容マップ
                      </p>
                      <div className="max-h-[17rem] overflow-auto pr-1 space-y-1">
                        {zip.members.map((member) => (
                          <div
                            key={`${zip.id}-${member.path}`}
                            className="grid grid-cols-[auto_1fr_auto] gap-2 text-[11px] border border-[var(--line)] rounded px-2 py-1 bg-[rgba(31,32,29,0.82)]"
                          >
                            <span
                              className="font-mono uppercase"
                              style={{ color: MEMBER_KIND_COLORS[member.kind] ?? "#f8f8f2" }}
                              title={member.kind}
                            >
                              {member.kind}
                            </span>
                            {member.url ? (
                              <a
                                href={member.url}
                                target="_blank"
                                rel="noreferrer"
                                className="truncate text-[#66d9ef] hover:underline"
                                title={member.path}
                              >
                                {member.path}
                              </a>
                            ) : (
                              <span className="truncate text-muted" title={member.path}>
                                {member.path}
                              </span>
                            )}
                            <span className="font-mono text-muted text-right">
                              {member.is_dir ? "--" : formatBytes(member.bytes)}
                            </span>
                          </div>
                        ))}
                        {zip.members_truncated ? (
                          <p className="text-[10px] text-muted font-mono">
                            showing first {zip.members.length} of {zip.members_total} entries
                          </p>
                        ) : null}
                      </div>
                    </div>

                    <div className="space-y-2">
                      <div className="rounded-md border border-[var(--line)] bg-[rgba(39,40,34,0.78)] p-2">
                        <p className="text-[11px] uppercase tracking-wide text-muted mb-2">
                          Type Mix / 種別分布
                        </p>
                        <div className="space-y-1.5 text-[11px]">
                          {typeRows.map(([kind, count]) => {
                            const pct = Math.round((count / Math.max(1, zip.members_total)) * 100);
                            return (
                              <div key={`${zip.id}-kind-${kind}`} className="grid grid-cols-[auto_1fr_auto] gap-2 items-center">
                                <span className="font-mono text-muted uppercase">{kind}</span>
                                <div className="h-1.5 rounded bg-bg-1 overflow-hidden">
                                  <div
                                    className="h-full"
                                    style={{
                                      width: `${pct}%`,
                                      backgroundColor: MEMBER_KIND_COLORS[kind] ?? "#f8f8f2",
                                    }}
                                  />
                                </div>
                                <span className="font-mono text-muted">{count}</span>
                              </div>
                            );
                          })}
                        </div>
                      </div>

                      <div className="rounded-md border border-[var(--line)] bg-[rgba(39,40,34,0.78)] p-2">
                        <p className="text-[11px] uppercase tracking-wide text-muted mb-2">
                          Extensions / 拡張子
                        </p>
                        <div className="space-y-1 text-[11px] font-mono text-muted">
                          {topExtensions.map((entry) => (
                            <div key={`${zip.id}-ext-${entry.ext}`} className="flex items-center justify-between">
                              <span>{entry.ext}</span>
                              <span>{entry.count}</span>
                            </div>
                          ))}
                        </div>
                      </div>

                      <div className="rounded-md border border-[var(--line)] bg-[rgba(39,40,34,0.78)] p-2">
                        <p className="text-[11px] uppercase tracking-wide text-muted mb-2 flex items-center gap-1">
                          <FolderTree size={12} />
                          Top Folders / 先頭階層
                        </p>
                        <div className="space-y-1 text-[11px] font-mono text-muted">
                          {topFolders.map((entry) => (
                            <div key={`${zip.id}-top-${entry.name}`} className="flex items-center justify-between gap-2">
                              <span className="truncate" title={entry.name}>{entry.name}</span>
                              <span>{entry.count}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </details>
            );
          })}
        </div>
      </section>

      <div className="grid grid-cols-[repeat(auto-fill,minmax(260px,1fr))] gap-3">
        {visibleItems.map((item) => (
          <article
            key={item.rel_path}
            className="border border-[var(--line)] rounded-xl p-3 bg-[rgba(39,40,34,0.84)]"
          >
            <div className="flex justify-between items-baseline mb-1">
              <strong className="text-sm font-semibold">{item.display_name.en}</strong>
              <span className="text-xs text-muted">{item.display_name.ja}</span>
            </div>

            <div className="flex items-center gap-1 text-xs text-muted mb-2">
              {getIcon(item.kind)}
              <span>
                {item.display_role.en} / {item.display_role.ja}
              </span>
            </div>

            <div className="mt-2">
              {item.kind === "audio" && (
                <audio controls src={item.url} className="w-full h-8">
                  <track kind="captions" />
                </audio>
              )}
              {item.kind === "image" && (
                <img
                  src={item.url}
                  alt={item.name}
                  loading="lazy"
                  className="w-full rounded-lg"
                />
              )}
              {item.kind === "video" && (
                <video controls src={item.url} className="w-full rounded-lg">
                  <track kind="captions" />
                </video>
              )}
              {!['audio', 'image', 'video'].includes(item.kind) && (
                <a
                  href={item.url}
                  target="_blank"
                  rel="noreferrer"
                  className="flex items-center gap-2 text-sm text-[#66d9ef] hover:underline"
                >
                  <Eye size={14} />
                  View / 表示
                </a>
              )}
            </div>

            <div className="mt-2 text-[10px] text-muted text-right">
              {item.part} | {(item.bytes / 1024).toFixed(1)} KB
            </div>
          </article>
        ))}
      </div>

      {items.length > MAX_VISIBLE_ITEMS ? (
        <div className="flex items-center justify-between rounded-lg border border-[var(--line)] bg-[rgba(45,46,39,0.86)] px-3 py-2">
          <p className="text-xs text-muted">
            showing <code>{visibleItems.length}</code> / <code>{items.length}</code> artifacts
          </p>
          <button
            type="button"
            className="border border-[var(--line)] rounded-md bg-[rgba(39,40,34,0.9)] px-3 py-1 text-xs font-semibold text-ink hover:bg-[rgba(55,56,48,0.96)]"
            onClick={() => setExpanded((value) => !value)}
          >
            {expanded
              ? "Show less / 折りたたむ"
              : `Show ${hiddenCount} more / さらに表示`}
          </button>
        </div>
      ) : null}
    </div>
  );
}
