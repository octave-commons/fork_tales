import fs from "node:fs";
import path from "node:path";
import process from "node:process";
import { fileURLToPath } from "node:url";
import { app, BrowserWindow, shell } from "electron";

const DEFAULT_WORLD_RUNTIME_URL = "http://127.0.0.1:8787";
const DEFAULT_WEAVER_RUNTIME_URL = "http://127.0.0.1:8793";

const currentFilePath = fileURLToPath(import.meta.url);
const currentDirPath = path.dirname(currentFilePath);
const distIndexPath = path.resolve(currentDirPath, "../dist/index.html");

function createMainWindow() {
  const window = new BrowserWindow({
    width: 1680,
    height: 1024,
    minWidth: 1080,
    minHeight: 720,
    autoHideMenuBar: true,
    backgroundColor: "#0c1219",
    title: "eta-mu world client",
    webPreferences: {
      preload: path.resolve(currentDirPath, "./preload.mjs"),
      sandbox: true,
      contextIsolation: true,
      nodeIntegration: false,
      webSecurity: true,
      spellcheck: false,
    },
  });

  window.webContents.setWindowOpenHandler(({ url }) => {
    void shell.openExternal(url);
    return { action: "deny" };
  });

  window.webContents.on("will-navigate", (event, targetUrl) => {
    const currentUrl = window.webContents.getURL();
    if (targetUrl === currentUrl) {
      return;
    }
    event.preventDefault();
    void shell.openExternal(targetUrl);
  });

  window.webContents.session.setPermissionRequestHandler((_webContents, permission, callback) => {
    callback(permission === "media");
  });

  const devServerUrl = String(process.env.VITE_DEV_SERVER_URL || "").trim();
  if (devServerUrl) {
    void window.loadURL(devServerUrl);
    return;
  }

  if (!fs.existsSync(distIndexPath)) {
    throw new Error("Missing dist/index.html. Run `npm run build` before launching Electron.");
  }

  void window.loadFile(distIndexPath);
}

app.whenReady().then(() => {
  process.env.ETA_MU_WORLD_BASE_URL =
    String(process.env.WORLD_RUNTIME_URL || "").trim() || DEFAULT_WORLD_RUNTIME_URL;
  process.env.ETA_MU_WEAVER_BASE_URL =
    String(process.env.WEAVER_RUNTIME_URL || "").trim() || DEFAULT_WEAVER_RUNTIME_URL;

  createMainWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createMainWindow();
    }
  });
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});
