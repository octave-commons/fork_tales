const chokidar = require('chokidar');
const axios = require('axios');
const path = require('path');

const WORLD_API = process.env.WORLD_API || 'http://127.0.0.1:8787';
const WATCH_PATHS = [
  path.join(__dirname, '../artifacts'),
  path.join(__dirname, '../NEW_LYRICS*.md'),
  path.join(__dirname, '../GATES_OF_TRUTH*.md'),
];

console.log(`[world-io] Initializing Sensory Organ...`);
console.log(`[world-io] Watching: ${WATCH_PATHS.join(', ')}`);

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

watcher
  .on('add', path => signalWorld('file_added', { path: path, event: 'add' }))
  .on('change', path => signalWorld('file_changed', { path: path, event: 'change' }))
  .on('unlink', path => signalWorld('file_removed', { path: path, event: 'unlink' }));

console.log(`[world-io] Sensory Organ Active. Listening for ripples...`);
