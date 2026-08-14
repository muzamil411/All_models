#!/usr/bin/env node
/*
 * Offline renderer: drives a scene's seek(t) frame by frame in headless
 * Chromium, screenshots each frame with alpha, then muxes with ffmpeg.
 *
 *   node render.mjs <scene> [--out name] [--params '<json>'] [--fps 30]
 *                           [--scale 1] [--preview]
 *
 * Outputs into out/<name>/:
 *   frames/*.png   universal alpha sequence — import straight into any NLE
 *   <name>.webm    VP9 + alpha, for quick review or web use
 *   <name>.mov     ProRes 4444 with alpha, for Premiere / Resolve / FCP
 *   <name>.mp4     flattened preview over a dark ground (optional, --preview)
 */

import { chromium } from "playwright-core";
import { spawn } from "node:child_process";
import { mkdir, rm, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const CHROME = process.env.CHROMIUM_PATH || "/opt/pw-browsers/chromium";
const FFMPEG = process.env.FFMPEG_PATH || "ffmpeg";

function arg(name, fallback = null) {
  const i = process.argv.indexOf(`--${name}`);
  if (i === -1) return fallback;
  const next = process.argv[i + 1];
  return next && !next.startsWith("--") ? next : true;
}

function run(cmd, args) {
  return new Promise((resolve, reject) => {
    const p = spawn(cmd, args, { stdio: ["ignore", "ignore", "pipe"] });
    let err = "";
    p.stderr.on("data", (d) => (err += d));
    p.on("close", (code) =>
      code === 0 ? resolve() : reject(new Error(`${cmd} exited ${code}\n${err.slice(-1500)}`))
    );
  });
}

const sceneName = process.argv[2];
if (!sceneName) {
  console.error("usage: node render.mjs <scene> [--params '<json>'] [--out name] [--preview]");
  process.exit(1);
}

const scenePath = path.join(HERE, "scenes", `${sceneName}.html`);
if (!existsSync(scenePath)) {
  console.error(`scene not found: ${scenePath}`);
  process.exit(1);
}

const outName = arg("out", sceneName);
const scale = Number(arg("scale", 1));
const outDir = path.join(HERE, "out", outName);
const framesDir = path.join(outDir, "frames");

await rm(outDir, { recursive: true, force: true });
await mkdir(framesDir, { recursive: true });

// Scene params travel as base64 JSON on the URL.
let url = `file://${scenePath}`;
const rawParams = arg("params", null);
if (rawParams && rawParams !== true) {
  JSON.parse(rawParams); // fail loudly here rather than silently in the page
  url += `?p=${encodeURIComponent(Buffer.from(rawParams, "utf8").toString("base64"))}`;
}

const browser = await chromium.launch({
  executablePath: CHROME,
  args: ["--force-color-profile=srgb", "--disable-lcd-text", "--font-render-hinting=none"],
});
const page = await browser.newPage({
  viewport: { width: 1920, height: 1080 },
  deviceScaleFactor: scale,
});

const pageErrors = [];
page.on("pageerror", (e) => pageErrors.push(e.message));

await page.goto(url, { waitUntil: "load" });
await page.waitForFunction(() => window.SCENE_READY !== undefined);
await page.evaluate(() => window.SCENE_READY);

if (pageErrors.length) {
  console.error("scene threw:\n  " + pageErrors.join("\n  "));
  await browser.close();
  process.exit(1);
}

const fps = Number(arg("fps", 0)) || (await page.evaluate(() => window.SCENE.fps || 30));
const duration = await page.evaluate(() => window.SCENE.duration);
const total = Math.max(1, Math.round(duration * fps));

process.stdout.write(`rendering ${sceneName}: ${duration}s @ ${fps}fps = ${total} frames\n`);

for (let f = 0; f < total; f++) {
  const t = f / fps;
  await page.evaluate((time) => window.SCENE.seek(time), t);
  await page.screenshot({
    path: path.join(framesDir, `f${String(f).padStart(5, "0")}.png`),
    omitBackground: true,
  });
  if (f % 15 === 0 || f === total - 1) {
    process.stdout.write(`\r  frame ${f + 1}/${total}`);
  }
}
process.stdout.write("\n");

if (pageErrors.length) {
  console.error("scene threw during seek:\n  " + pageErrors.join("\n  "));
}
await browser.close();

const seq = path.join(framesDir, "f%05d.png");

// VP9 + alpha: small, quick to review.
await run(FFMPEG, [
  "-y", "-loglevel", "error", "-framerate", String(fps), "-i", seq,
  "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-b:v", "0", "-crf", "24",
  "-auto-alt-ref", "0", path.join(outDir, `${outName}.webm`),
]);

// ProRes 4444 + alpha: the file you actually drop on a timeline.
await run(FFMPEG, [
  "-y", "-loglevel", "error", "-framerate", String(fps), "-i", seq,
  "-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le",
  "-alpha_bits", "16", "-vendor", "apl0", path.join(outDir, `${outName}.mov`),
]);

if (arg("preview", false)) {
  // Flatten over the reference's near-black ground so alpha edges are visible.
  await run(FFMPEG, [
    "-y", "-loglevel", "error",
    "-f", "lavfi", "-i", `color=c=0x0B0B14:s=1920x1080:r=${fps}:d=${duration}`,
    "-framerate", String(fps), "-i", seq,
    "-filter_complex", "[0:v][1:v]overlay=shortest=1,format=yuv420p",
    "-c:v", "libx264", "-crf", "18", path.join(outDir, `${outName}.mp4`),
  ]);
}

console.log(`done -> ${path.relative(HERE, outDir)}/`);
