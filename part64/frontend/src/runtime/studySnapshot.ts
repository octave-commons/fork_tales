import type { StudySnapshotPayload } from "../types";
import { runtimeApiUrl, runtimeBaseUrl } from "./endpoints";

const STUDY_SNAPSHOT_CACHE_TTL_MS = 2500;

type CacheEntry = {
  fetchedAtMs: number;
  payload: StudySnapshotPayload;
};

const studySnapshotCache = new Map<string, CacheEntry>();
const studySnapshotInflight = new Map<string, Promise<StudySnapshotPayload>>();

function normalizedLimit(limit: number): number {
  return Number.isFinite(limit) ? Math.max(1, Math.floor(limit)) : 4;
}

function studySnapshotKey(limit: number): string {
  return `${runtimeBaseUrl()}|limit=${normalizedLimit(limit)}`;
}

export async function fetchStudySnapshot(limit = 4): Promise<StudySnapshotPayload> {
  const key = studySnapshotKey(limit);
  const nowMs = Date.now();
  const cached = studySnapshotCache.get(key);
  if (cached && (nowMs - cached.fetchedAtMs) <= STUDY_SNAPSHOT_CACHE_TTL_MS) {
    return cached.payload;
  }

  const inflight = studySnapshotInflight.get(key);
  if (inflight) {
    return inflight;
  }

  const request = fetch(runtimeApiUrl(`/api/study?limit=${normalizedLimit(limit)}`))
    .then(async (response) => {
      if (!response.ok) {
        throw new Error(`/api/study failed: ${response.status}`);
      }
      const payload = (await response.json()) as StudySnapshotPayload;
      studySnapshotCache.set(key, { fetchedAtMs: Date.now(), payload });
      return payload;
    })
    .finally(() => {
      studySnapshotInflight.delete(key);
    });

  studySnapshotInflight.set(key, request);
  return request;
}

export function invalidateStudySnapshot(limit?: number): void {
  if (typeof limit === "number" && Number.isFinite(limit)) {
    const key = studySnapshotKey(limit);
    studySnapshotCache.delete(key);
    studySnapshotInflight.delete(key);
    return;
  }
  studySnapshotCache.clear();
  studySnapshotInflight.clear();
}
