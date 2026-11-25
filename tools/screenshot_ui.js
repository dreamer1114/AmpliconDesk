/**
 * 用 CDP 驱动 Edge，走完整个流程并逐步截图。
 *   node tools/_shoot.js
 * 输出到 preview/_shot-<name>.png
 */
const { spawn } = require("child_process");
const fs = require("fs");
const path = require("path");
const http = require("http");

const EDGE_CANDIDATES = [
  process.env.EDGE_PATH,
  "C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe",
  "C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe",
].filter(Boolean);
const EDGE = EDGE_CANDIDATES.find((candidate) => fs.existsSync(candidate));
const PORT = 9333;
const URL = "http://127.0.0.1:8765/";
const OUT = path.join(__dirname, "..", "preview");
const PROFILE = path.join(process.env.TEMP || ".", "edge-cdp-shot");

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function getJSON(url) {
  return new Promise((resolve, reject) => {
    http.get(url, (res) => {
      let d = "";
      res.on("data", (c) => (d += c));
      res.on("end", () => {
        try { resolve(JSON.parse(d)); } catch (e) { reject(e); }
      });
    }).on("error", reject);
  });
}

class CDP {
  constructor(ws) { this.ws = ws; this.id = 0; this.waiting = new Map(); }
  static async connect(wsUrl) {
    const ws = new WebSocket(wsUrl);
    const c = new CDP(ws);
    await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });
    ws.onmessage = (ev) => {
      const m = JSON.parse(ev.data);
      if (m.id && c.waiting.has(m.id)) {
        const { resolve, reject } = c.waiting.get(m.id);
        c.waiting.delete(m.id);
        m.error ? reject(new Error(JSON.stringify(m.error))) : resolve(m.result);
      }
    };
    return c;
  }
  send(method, params = {}) {
    const id = ++this.id;
    return new Promise((resolve, reject) => {
      this.waiting.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
      setTimeout(() => {
        if (this.waiting.has(id)) { this.waiting.delete(id); reject(new Error("timeout " + method)); }
      }, 180000);
    });
  }
  async eval(expr) {
    const r = await this.send("Runtime.evaluate", {
      expression: expr, awaitPromise: true, returnByValue: true,
    });
    if (r.exceptionDetails) throw new Error(r.exceptionDetails.text + " " +
      (r.exceptionDetails.exception && r.exceptionDetails.exception.description || ""));
    return r.result.value;
  }
  async shot(name) {
    const r = await this.send("Page.captureScreenshot", { format: "png" });
    const f = path.join(OUT, `_shot-${name}.png`);
    fs.writeFileSync(f, Buffer.from(r.data, "base64"));
    console.log(`  [截图] ${path.basename(f)}`);
  }
}

(async () => {
  if (!EDGE) {
    console.error("未找到 Edge。请安装 Edge，或通过 EDGE_PATH 指定 msedge.exe 的路径。");
    process.exit(1);
  }
  fs.mkdirSync(OUT, { recursive: true });
  console.log("启动 Edge（远程调试）...");
  const edge = spawn(EDGE, [
    "--headless=new", "--disable-gpu", "--hide-scrollbars",
    "--force-device-scale-factor=1",
    `--remote-debugging-port=${PORT}`,
    `--user-data-dir=${PROFILE}`,
    "--window-size=1400,1000",
    "about:blank",
  ], { stdio: "ignore", detached: false });

  let target = null;
  for (let i = 0; i < 40; i++) {
    await sleep(500);
    try {
      const list = await getJSON(`http://127.0.0.1:${PORT}/json/list`);
      target = list.find((t) => t.type === "page");
      if (target) break;
    } catch (_) { /* 还没起来 */ }
  }
  if (!target) { console.error("连不上 Edge 调试端口"); edge.kill(); process.exit(1); }

  const cdp = await CDP.connect(target.webSocketDebuggerUrl);
  await cdp.send("Page.enable");
  await cdp.send("Runtime.enable");
  await cdp.send("Emulation.setDeviceMetricsOverride", {
    width: 1400, height: 1000, deviceScaleFactor: 1, mobile: false,
  });

  console.log("打开页面...");
  await cdp.send("Page.navigate", { url: URL });
  await sleep(3500);

  console.log("① 选择数据");
  await cdp.shot("1-data");

  console.log("载入演示数据 + 数据体检");
  await cdp.eval("loadDemo()");
  await sleep(1800);
  await cdp.eval("doInspect()");
  await sleep(3000);
  await cdp.eval("goStep(2)");
  await sleep(900);
  await cdp.shot("2-group");

  console.log("③ 分析内容");
  await cdp.eval("goStep(3)");
  await sleep(900);
  await cdp.shot("3-analysis");

  console.log("④ 绘图样式（等预览图）");
  await cdp.eval("goStep(4)");
  await sleep(9000);
  await cdp.shot("4-style");

  console.log("提交分析任务（全量，约 90-140 秒）...");
  // 用 void 让表达式立即返回，任务在后台跑；否则 awaitPromise 会一直挂到跑完
  await cdp.eval("void quickRun(); 'started'");
  let done = false;
  for (let i = 0; i < 90; i++) {
    await sleep(5000);
    let step = "?";
    try {
      step = await cdp.eval("(document.querySelector('.panel:not(.hidden)')||{}).dataset?.panel || '?'");
    } catch (_) { /* 页面忙，忽略 */ }
    if (i % 2 === 0) console.log(`    ...第 ${i * 5}s，当前步骤 ${step}`);
    if (step === "6") { done = true; break; }
  }
  if (!done) console.log("    警告：等待超时，仍尝试截图");
  await sleep(5000);
  await cdp.eval("goStep(6)");
  await sleep(3000);
  await cdp.shot("6-result-top");
  await cdp.eval("window.scrollTo(0, 2600)");
  await sleep(1000);
  await cdp.shot("6-result-tables");
  await cdp.eval("window.scrollTo(0, 900)");
  await sleep(1200);
  await cdp.shot("6-result-mid");
  await cdp.eval("window.scrollTo(0, document.body.scrollHeight)");
  await sleep(1500);
  await cdp.shot("6-result-end");

  cdp.ws.close();
  edge.kill();
  await sleep(600);
  console.log("完成。");
  process.exit(0);
})().catch((e) => { console.error("失败:", e.message); process.exit(1); });
