#!/usr/bin/env node

import process from "node:process";
import { mkdir, writeFile } from "node:fs/promises";

function parseArgs(argv) {
  const options = {
    debugPort: 9335,
    firstFrame: 0,
    frameCount: 300,
    periodMs: 100,
    settleMs: 80,
    outputDir: null,
    cameraView: "overview",
    displayMode: "simulation",
    imageFormat: "png",
    imageQuality: 97,
  };
  for (let index = 0; index < argv.length; index += 2) {
    const name = argv[index];
    const rawValue = argv[index + 1];
    if (
      name === "--output-dir"
      || name === "--camera-view"
      || name === "--display-mode"
      || name === "--image-format"
    ) {
      if (!rawValue) throw new Error(`${name} 缺少参数值`);
      if (name === "--output-dir") options.outputDir = rawValue;
      else if (name === "--camera-view") options.cameraView = rawValue;
      else if (name === "--image-format") options.imageFormat = rawValue;
      else options.displayMode = rawValue;
      continue;
    }
    const value = Number(rawValue);
    if (!Number.isFinite(value)) throw new Error(`参数 ${name} 缺少有效数值`);
    if (name === "--debug-port") options.debugPort = value;
    else if (name === "--first-frame") options.firstFrame = value;
    else if (name === "--frame-count") options.frameCount = value;
    else if (name === "--period-ms") options.periodMs = value;
    else if (name === "--settle-ms") options.settleMs = value;
    else if (name === "--image-quality") options.imageQuality = value;
    else throw new Error(`未知参数：${name}`);
  }
  if (!["near", "overview"].includes(options.cameraView)) {
    throw new Error("--camera-view 必须是 near 或 overview");
  }
  if (!["simulation", "page"].includes(options.displayMode)) {
    throw new Error("--display-mode 必须是 simulation 或 page");
  }
  if (!["png", "jpeg"].includes(options.imageFormat)) {
    throw new Error("--image-format 必须是 png 或 jpeg");
  }
  if (options.imageQuality < 1 || options.imageQuality > 100) {
    throw new Error("--image-quality 必须在 [1, 100]");
  }
  return options;
}

function wait(milliseconds) {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

async function findPage(debugPort) {
  const response = await fetch(`http://127.0.0.1:${debugPort}/json`);
  if (!response.ok) throw new Error(`无法读取 Chromium 调试端口：HTTP ${response.status}`);
  const targets = await response.json();
  const target = targets.find(
    (item) => item.type === "page" && item.title === "Nexus Simulator",
  ) || targets.find((item) => item.type === "page" && item.url.includes(":8766/"));
  if (!target?.webSocketDebuggerUrl) throw new Error("没有找到 Nexus 录制页面");
  return target;
}

class CdpClient {
  constructor(url) {
    this.nextId = 1;
    this.pending = new Map();
    this.socket = new WebSocket(url);
  }

  async connect() {
    await new Promise((resolve, reject) => {
      this.socket.addEventListener("open", resolve, { once: true });
      this.socket.addEventListener("error", reject, { once: true });
    });
    this.socket.addEventListener("message", (event) => {
      const message = JSON.parse(event.data);
      const pending = this.pending.get(message.id);
      if (!pending) return;
      this.pending.delete(message.id);
      if (message.error) pending.reject(new Error(message.error.message));
      else pending.resolve(message.result);
    });
  }

  command(method, params = {}) {
    const id = this.nextId++;
    this.socket.send(JSON.stringify({ id, method, params }));
    return new Promise((resolve, reject) => this.pending.set(id, { resolve, reject }));
  }

  async evaluate(expression) {
    const response = await this.command("Runtime.evaluate", {
      expression,
      awaitPromise: true,
      returnByValue: true,
    });
    if (response.exceptionDetails) {
      throw new Error(response.exceptionDetails.exception?.description || "页面脚本执行失败");
    }
    return response.result?.value;
  }

  close() {
    this.socket.close();
  }
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  const target = await findPage(options.debugPort);
  const client = new CdpClient(target.webSocketDebuggerUrl);
  await client.connect();
  try {
    await client.evaluate("pauseReplay(); true");
    if (options.outputDir) {
      await mkdir(options.outputDir, { recursive: true });
      await client.command("Page.enable");
      await client.command("Emulation.setDeviceMetricsOverride", {
        width: 1920,
        height: 1080,
        deviceScaleFactor: 1,
        mobile: false,
      });
      await client.evaluate(
        `document.body.classList.toggle("is-recording", ${options.displayMode === "simulation"}); true`,
      );
      await client.evaluate(
        `setCameraView(${JSON.stringify(options.cameraView)}, {silent: true}).then(() => true)`,
      );
      await wait(300);
    }
    for (let offset = 0; offset < options.frameCount; offset += 1) {
      const frame = options.firstFrame + offset;
      const startedAt = performance.now();
      await client.evaluate(`requestReplayFrame(${frame}).then(() => ${frame})`);
      if (options.outputDir) {
        await wait(options.settleMs);
        const screenshot = await client.command("Page.captureScreenshot", {
          format: options.imageFormat,
          quality: options.imageFormat === "jpeg" ? options.imageQuality : undefined,
          fromSurface: true,
          captureBeyondViewport: false,
          optimizeForSpeed: true,
        });
        const extension = options.imageFormat === "jpeg" ? "jpg" : "png";
        const fileName = `frame-${String(offset).padStart(6, "0")}.${extension}`;
        await writeFile(`${options.outputDir}/${fileName}`, Buffer.from(screenshot.data, "base64"));
      }
      const remainingMs = options.periodMs - (performance.now() - startedAt);
      if (remainingMs > 0) await wait(remainingMs);
      if (offset % 30 === 0 || offset === options.frameCount - 1) {
        process.stdout.write(`\r已渲染 ${offset + 1}/${options.frameCount} 帧`);
      }
    }
    process.stdout.write("\n");
  } finally {
    client.close();
  }
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error.message}\n`);
  process.exitCode = 1;
});
