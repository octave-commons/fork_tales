const chokidar = require('chokidar');
const axios = require('axios');
const path = require('path');

const WORLD_API = process.env.WORLD_API || 'http://127.0.0.1:8787';
const ETA_MU_INBOX_ROOT = String(process.env.ETA_MU_INBOX_ROOT || '').trim();
const ETA_MU_SYNC_DEBOUNCE_MS = Number.parseInt(
  String(process.env.ETA_MU_SYNC_DEBOUNCE_MS || '1500'),
  10,
);
const WATCH_PATHS = [
  path.join(__dirname, '../artifacts'),
  path.join(__dirname, '../NEW_LYRICS*.md'),
  path.join(__dirname, '../GATES_OF_TRUTH*.md'),
  ...(ETA_MU_INBOX_ROOT ? [ETA_MU_INBOX_ROOT] : []),
];
const ETA_MU_INBOX_ROOT_RESOLVED = ETA_MU_INBOX_ROOT ? path.resolve(ETA_MU_INBOX_ROOT) : '';

let inboxSyncTimer = null;

console.log(`[world-io] Initializing Sensory Organ...`);
console.log(`[world-io] Watching: ${WATCH_PATHS.join(', ')}`);
if (ETA_MU_INBOX_ROOT_RESOLVED) {
  console.log(`[world-io] Eta Mu inbox: ${ETA_MU_INBOX_ROOT_RESOLVED}`);
}

const watcher = chokidar.watch(WATCH_PATHS, {
  ignored: /(^|[\/\\])\../,
  persistent: true,
  ignoreInitial: true,
});

async function signalWorld(type, data) {
  try {
    await axios.post(`${WORLD_API}/api/input-stream`, {
      type: type,
      data: data
    });
    console.log(`[world-io] Signal sent: ${type} -> ${data.path}`);
  } catch (err) {
    console.error(`[world-io] Failed to signal world: ${err.message}`);
  }
}

function normalizePath(filePath) {
  return path.resolve(String(filePath || ''));
}

function isEtaMuInboxPath(filePath) {
  if (!ETA_MU_INBOX_ROOT_RESOLVED) {
    return false;
  }
  const resolved = normalizePath(filePath);
  return resolved === ETA_MU_INBOX_ROOT_RESOLVED
    || resolved.startsWith(`${ETA_MU_INBOX_ROOT_RESOLVED}${path.sep}`);
}

async function requestEtaMuSync(triggerPath) {
  try {
    await axios.post(`${WORLD_API}/api/eta-mu/sync`, {
      wait: false,
      force: true,
    });
    console.log(`[world-io] Inbox sync scheduled -> ${triggerPath}`);
  } catch (err) {
    console.error(`[world-io] Failed to schedule inbox sync: ${err.message}`);
  }
}

function scheduleEtaMuSync(triggerPath) {
  if (!ETA_MU_INBOX_ROOT_RESOLVED) {
    return;
  }
  if (inboxSyncTimer) {
    clearTimeout(inboxSyncTimer);
  }
  inboxSyncTimer = setTimeout(() => {
    inboxSyncTimer = null;
    void requestEtaMuSync(triggerPath);
  }, Math.max(0, Number.isFinite(ETA_MU_SYNC_DEBOUNCE_MS) ? ETA_MU_SYNC_DEBOUNCE_MS : 1500));
}

async function handleFsEvent(type, filePath, event) {
  await signalWorld(type, { path: filePath, event: event });
  if (isEtaMuInboxPath(filePath)) {
    scheduleEtaMuSync(filePath);
  }
}

watcher
  .on('add', filePath => void handleFsEvent('file_added', filePath, 'add'))
  .on('change', filePath => void handleFsEvent('file_changed', filePath, 'change'))
  .on('unlink', filePath => void handleFsEvent('file_removed', filePath, 'unlink'));

console.log(`[world-io] Sensory Organ Active. Listening for ripples...`);
