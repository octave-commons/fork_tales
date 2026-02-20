export function relativeTime(isoText: string | undefined): string {
  const raw = String(isoText || "").trim();
  if (!raw) {
    return "n/a";
  }
  const parsed = Date.parse(raw);
  if (!Number.isFinite(parsed)) {
    return raw;
  }
  const seconds = Math.max(0, Math.round((Date.now() - parsed) / 1000));
  if (seconds < 45) {
    return `${seconds}s ago`;
  }
  if (seconds < 3600) {
    return `${Math.round(seconds / 60)}m ago`;
  }
  if (seconds < 86400) {
    return `${Math.round(seconds / 3600)}h ago`;
  }
  return `${Math.round(seconds / 86400)}d ago`;
}
