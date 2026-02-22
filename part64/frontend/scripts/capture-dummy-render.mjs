import { mkdir } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { chromium } from "@playwright/test";

const targetUrl = process.argv[2] || "http://127.0.0.1:5173";
const outputPath = resolve(process.argv[3] || "part64/frontend/artifacts/webgl-dummy-render.png");

await mkdir(dirname(outputPath), { recursive: true });

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1536, height: 1024 } });

await page.goto(targetUrl, { waitUntil: "domcontentloaded" });

await page.waitForSelector("canvas", { timeout: 15000 });
await page.waitForFunction(() => {
  const meta = document.querySelector("p");
  const text = String(meta?.textContent || "");
  return text.includes("webgl overlay particles:") || document.querySelectorAll("canvas").length >= 2;
}, { timeout: 15000 }).catch(() => {});

await page.waitForTimeout(2200);
await page.screenshot({ path: outputPath, fullPage: true });

const renderStatus = await page.evaluate(() => {
  const paragraphs = Array.from(document.querySelectorAll("p"));
  const metaText = paragraphs
    .map((row) => String(row.textContent || ""))
    .find((text) => text.includes("webgl overlay particles:")) || "";
  const canvasSizes = Array.from(document.querySelectorAll("canvas")).map((canvas) => ({
    width: canvas.width,
    height: canvas.height,
  }));
  return {
    metaText,
    canvasCount: canvasSizes.length,
    canvasSizes,
  };
});

await browser.close();

process.stdout.write(`${JSON.stringify({ outputPath, targetUrl, renderStatus }, null, 2)}\n`);
