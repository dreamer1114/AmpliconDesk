/* AmpliconDesk 前端逻辑 —— 无框架，直接操作 DOM */

const S = {
  files: { abundance: null, metadata: null, taxonomy: null, env: null },
  inspect: null,
  config: null,
  jobId: null,
  pollTimer: null,
  result: null,
  selectedPairs: new Set(),
  style: {},          // 扁平样式键值（含 fig.<id>.<key> 覆盖）
  styleScope: "",     // "" = 全局，否则为图 ID
  styleTab: "全局",
  previewTimer: null,
  previewBusy: false,
  chosenOrd: null,
  chosenStats: null,
};

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

/* ---------------- 通用 ---------------- */
function toast(msg, ms = 2600) {
  const t = $("#toast");
  t.textContent = msg;
  t.classList.remove("hidden");
  clearTimeout(t._t);
  t._t = setTimeout(() => t.classList.add("hidden"), ms);
}

async function api(url, opts) {
  const res = await fetch(url, opts);
  let data = null;
  try { data = await res.json(); } catch { /* 非 JSON */ }
  if (!res.ok || (data && data.ok === false)) {
    throw new Error((data && (data.error || data.message)) || `请求失败（${res.status}）`);
  }
  return data;
}

function goStep(n) {
  $$(".panel").forEach((p) => p.classList.toggle("hidden", p.dataset.panel !== String(n)));
  $$(".step").forEach((s) => {
    const k = Number(s.dataset.step);
    s.classList.toggle("active", k === n);
    s.classList.toggle("done", k < n);
  });
  window.scrollTo({ top: 0, behavior: "smooth" });
  if (n === 4 && !$("#previewFrame").dataset.loaded) queuePreview();
}

/* ---------------- 启动 ---------------- */
async function boot() {
  try {
    const cfg = await api("/api/config");
    S.config = cfg;
    S.chosenOrd = null;
    S.chosenStats = null;
    $("#ver").textContent = "v" + cfg.version;
    $("#platformPill").textContent = cfg.platform === "Windows" ? "本地运行 · Windows" : "本地运行";

    fillSelect($("#rankSelect"), [{ v: "", t: "自动（推荐）" },
      ...cfg.ranks.map((r) => ({ v: r, t: r }))]);
    fillSelect($("#ordMetric"), cfg.beta_metrics.map((m) => ({ v: m.id, t: m.label })));
    fillSelect($("#hypMethod"), (cfg.hypothesis_methods || []).map((m) => ({ v: m.id, t: m.label })));
    fillSelect($("#stylePreset"), cfg.style_presets.map((p) => ({ v: p, t: p })));
    fillSelect($("#previewFigure"), cfg.previewable.map((p) => ({ v: p.id, t: p.label })));

    $("#betaMetrics").innerHTML = cfg.beta_metrics
      .map((m) => `<label class="switch"><input type="checkbox" value="${m.id}"><span>${m.label}</span></label>`)
      .join("");
    $("#ordChecks").innerHTML = cfg.ordinations
      .map((o) => `<label class="switch"><input type="checkbox" value="${o.id}"
        data-group="${esc(o.group)}"><span>${o.label}</span></label>`).join("");
    $("#statChecks").innerHTML = cfg.stats_tests
      .map((o) => `<label class="switch"><input type="checkbox" value="${o.id}"><span>${o.label}</span></label>`)
      .join("");

    applyAnalysisType();
    buildStyleScopeOptions();
    wire();
    renderStyleForm();
    loadThemes();
    loadHistory();
  } catch (e) {
    toast("初始化失败：" + e.message, 6000);
  }
}

function fillSelect(sel, items) {
  sel.innerHTML = items.map((i) => `<option value="${esc(i.v)}">${esc(i.t)}</option>`).join("");
}

function applyAnalysisType() {
  // 本工具只做 16S 扩增子，分析类型固定
  const list = (S.config && S.config.analysis_types) || [];
  const t = list[0];
  if (!t) return;
  $("#analysisNote").textContent = t.note;

  const defaults = t.default_beta || ["braycurtis"];
  $$("#betaMetrics input").forEach((i) => { i.checked = defaults.includes(i.value); });

  if (S.chosenOrd === null) {
    const ord = t.default_ord && t.default_ord.length ? t.default_ord : ["pcoa", "nmds", "pca"];
    $$("#ordChecks input").forEach((i) => { i.checked = ord.includes(i.value); });
  }
  if (S.chosenStats === null) {
    $$("#statChecks input").forEach((i) => {
      i.checked = ["permanova", "anosim", "mrpp", "permdisp"].includes(i.value);
    });
  }
}

/* ---------------- 事件绑定 ---------------- */
function wire() {
  $$(".dropzone").forEach((dz) => {
    const kind = dz.dataset.kind;
    const input = dz.querySelector("input[type=file]");
    dz.onclick = () => input.click();
    input.onchange = () => { if (input.files[0]) upload(kind, input.files[0]); };
    dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("dragover"); });
    dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
    dz.addEventListener("drop", (e) => {
      e.preventDefault();
      dz.classList.remove("dragover");
      const f = e.dataTransfer.files[0];
      if (f) upload(kind, f);
    });
  });

  $$("[data-goto]").forEach((b) => { b.onclick = () => goStep(Number(b.dataset.goto)); });
  $("#btnInspect").onclick = doInspect;
  $("#btnQuickRun").onclick = quickRun;
  $$("[data-preset]").forEach((b) => { b.onclick = () => applyQuickPreset(b.dataset.preset); });
  $("#btnUsePaste").onclick = usePastedMeta;
  $("#btnRun").onclick = runAnalysis;
  $("#btnToResult").onclick = () => goStep(6);
  $("#btnNewRun").onclick = () => { resetRun(); goStep(1); };
  $("#btnHistory").onclick = () => { loadHistory(); $("#historyModal").classList.remove("hidden"); };
  $("#btnCloseHistory").onclick = () => $("#historyModal").classList.add("hidden");
  $("#historyModal").onclick = (e) => { if (e.target.id === "historyModal") $("#historyModal").classList.add("hidden"); };
  $("#btnDemo").onclick = loadDemo;
  $("#groupColumn").onchange = () => { S.selectedPairs.clear(); renderGroupPreview(); renderStyleForm(); };

  $$("#ordChecks input").forEach((i) => {
    i.onchange = () => {
      S.chosenOrd = $$("#ordChecks input:checked").map((x) => x.value);
      queuePreview();
    };
  });
  $$("#statChecks input").forEach((i) => { i.onchange = () => { S.chosenStats = $$("#statChecks input:checked").map((x) => x.value); }; });
  $$("#betaMetrics input").forEach((i) => { i.onchange = queuePreview; });

  // 样式面板
  $("#stylePreset").onchange = applyPreset;
  $("#styleScope").onchange = () => {
    S.styleScope = $("#styleScope").value;
    $("#scopeTip").textContent = S.styleScope
      ? `仅影响「${$("#previewFigure").selectedOptions[0]?.text || S.styleScope}」这一张图，其余图沿用全局设置`
      : "全局：影响本次分析的所有图";
    renderStyleForm();
  };
  $("#previewFigure").onchange = () => { syncScopeOptions(); queuePreview(); };
  $("#btnRefreshPreview").onclick = () => queuePreview(true);
  $("#btnResetStyle").onclick = () => {
    S.style = {};
    $("#stylePreset").value = "默认";
    renderStyleForm();
    queuePreview(true);
    toast("已恢复默认样式");
  };
  $("#btnSaveTheme").onclick = () => $("#saveThemeModal").classList.remove("hidden");
  $("#btnCancelTheme").onclick = () => $("#saveThemeModal").classList.add("hidden");
  $("#btnConfirmTheme").onclick = saveTheme;
  $("#btnLoadTheme").onclick = loadTheme;

  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeLightbox(); });
}

/* ---------------- 上传 ---------------- */
async function upload(kind, file) {
  const dz = document.querySelector(`.dropzone[data-kind="${kind}"]`);
  dz.querySelector(".dz-file").textContent = "上传中…";
  const fd = new FormData();
  fd.append("file", file);
  try {
    const res = await api("/api/upload", { method: "POST", body: fd });
    S.files[kind] = { path: res.path, name: res.name };
    dz.classList.add("filled");
    dz.querySelector(".dz-file").textContent = `✓ ${res.name}`;
    const prev = res.preview;
    if (prev && prev.ok) {
      dz.querySelector(".dz-desc").innerHTML = `已识别：<b>${prev.shape[0]}</b> 行 × <b>${prev.shape[1]}</b> 列`;
    }
    updateInspectButton();
  } catch (e) {
    dz.querySelector(".dz-file").textContent = "";
    toast("上传失败：" + e.message, 5000);
  }
}

function updateInspectButton() {
  $("#btnInspect").disabled = !(S.files.abundance && S.files.metadata);
}

async function usePastedMeta() {
  const text = $("#pasteMeta").value.trim();
  if (!text) { toast("请先粘贴分组内容"); return; }
  try {
    const res = await api("/api/metadata/from_text", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    S.files.metadata = { path: res.path, name: "粘贴的分组" };
    const dz = document.querySelector('.dropzone[data-kind="metadata"]');
    dz.classList.add("filled");
    dz.querySelector(".dz-file").textContent = `✓ 已解析 ${res.n_samples} 个样本`;
    dz.querySelector(".dz-desc").innerHTML = `分组列：<b>${res.columns.join(" / ")}</b>`;
    $("#pasteStatus").textContent = `✓ 已解析 ${res.n_samples} 个样本，${res.columns.length} 列`;
    updateInspectButton();
  } catch (e) {
    $("#pasteStatus").textContent = "";
    toast("解析失败：" + e.message, 5000);
  }
}

/* ---------------- 数据体检 ---------------- */
async function doInspect() {
  if (!S.files.abundance || !S.files.metadata) { toast("请先选择丰度表和分组表"); return; }
  const btn = $("#btnInspect");
  btn.disabled = true;
  btn.textContent = "检查中…";
  try {
    const res = await api("/api/inspect", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildInspectPayload()),
    });
    S.inspect = res;
    renderInspect(res);
    const sel = $("#groupColumn");
    const cands = res.group_candidates || [];
    sel.innerHTML = cands.length
      ? cands.map((g) => `<option value="${esc(g.column)}">${esc(g.column)}（${g.n_levels} 组）</option>`).join("")
      : res.metadata_columns.map((c) => `<option value="${esc(c)}">${esc(c)}</option>`).join("");
    renderGroupPreview();
    renderStyleForm();
    goStep(2);
  } catch (e) {
    toast("数据检查失败：" + e.message, 6000);
  } finally {
    btn.disabled = false;
    btn.textContent = "下一步：检查数据";
  }
}

function buildInspectPayload() {
  return {
    abundance_path: S.files.abundance?.path || "",
    metadata_path: S.files.metadata?.path || "",
    taxonomy_path: S.files.taxonomy?.path || "",
    env_path: S.files.env?.path || "",
  };
}

function renderInspect(r) {
  const matched = r.n_samples_matched;
  const cls = matched >= 6 ? "ok" : "bad";
  const extra = (r.n_only_in_table || 0) + (r.n_only_in_metadata || 0);
  const matchBox = extra === 0
    ? `<div class="alert ok">✓ 丰度表与分组表的样本名 <b>完全匹配</b>（${matched} 个样本）。</div>`
    : `<div class="alert warn"><b>样本名未完全匹配</b>，已按交集分析。
       <ul><li>仅在丰度表中（${r.n_only_in_table}）：${(r.only_in_table || []).slice(0, 12).join("、") || "无"}</li>
       <li>仅在分组表中（${r.n_only_in_metadata}）：${(r.only_in_metadata || []).slice(0, 12).join("、") || "无"}</li></ul></div>`;

  let enrichHtml = "";

  $("#inspectBox").innerHTML = `
    <div class="stat"><div class="k">匹配样本</div><div class="v ${cls}">${matched}</div>
      <div class="n">两组数据共有的样本数</div></div>
    <div class="stat"><div class="k">特征数</div><div class="v">${r.n_features}</div>
      <div class="n">OTU / 物种 / 基因数</div></div>
    <div class="stat"><div class="k">表格方向</div><div class="v sm">${r.transposed ? "已自动转置" : "无需转置"}</div>
      <div class="n">${esc(r.orient_reason || "")}</div></div>
    <div class="stat"><div class="k">分类注释</div><div class="v sm">${r.taxonomy_ok ? "已提供" : "未提供"}</div>
      <div class="n">${r.taxonomy_ok ? "可做分类级组成图" : "将按特征 ID 直接展示"}</div></div>
    <div class="stat"><div class="k">环境因子</div>
      <div class="v sm ${r.env_ok ? "ok" : ""}">${r.env_ok ? "已提供" : "未提供"}</div>
      <div class="n">${r.env_ok ? (r.env_columns || []).join("、") : "RDA/CCA/envfit/Mantel 将用分组构造"}</div></div>
    ${enrichHtml}
    ${matchBox}`;
}

function renderGroupPreview() {
  const col = $("#groupColumn").value;
  const g = (S.inspect?.group_candidates || []).find((x) => x.column === col);
  const box = $("#groupPreview");
  const pairsBox = $("#comparePairs");
  if (!g) { box.innerHTML = ""; pairsBox.innerHTML = ""; return; }

  const colors = paletteColors();
  box.innerHTML = `<div class="muted small" style="width:100%">分组构成（${g.n_levels} 组）：</div>` +
    g.levels.map((lv, i) =>
      `<div class="gchip"><span class="dot" style="background:${colors[i % colors.length]}"></span>
        ${esc(lv)}<span class="cnt">n=${g.counts[lv]}</span></div>`).join("");

  const levels = g.levels;
  if (levels.length < 2) {
    pairsBox.innerHTML = `<div class="muted small">该列只有 1 个水平，无法进行组间比较。</div>`;
    return;
  }
  const pairs = [];
  for (let i = 0; i < levels.length; i++)
    for (let j = i + 1; j < levels.length; j++) pairs.push([levels[i], levels[j]]);

  pairsBox.innerHTML = `<div class="muted small" style="margin-bottom:4px">勾选要执行的比较（不勾 = 全部执行）：</div>` +
    pairs.map(([a, b]) => `<label class="pair"><input type="checkbox" data-pair="${esc(a)}||${esc(b)}">
        <span>${esc(a)}</span><span class="arrow">vs</span><span>${esc(b)}</span></label>`).join("");
  $$("#comparePairs input").forEach((c) => {
    c.onchange = () => {
      if (c.checked) S.selectedPairs.add(c.dataset.pair);
      else S.selectedPairs.delete(c.dataset.pair);
    };
  });
}

function paletteColors() {
  const custom = (S.style["palette.custom"] || []).filter((x) => String(x).trim());
  if (custom.length) return custom;
  const preset = S.style["palette.preset"] || "default";
  const pal = (S.config?.palettes || {})[preset] || (S.config?.palettes || {}).default || [];
  return pal.length ? pal : ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"];
}

function currentGroups() {
  const col = $("#groupColumn").value;
  const g = (S.inspect?.group_candidates || []).find((x) => x.column === col);
  return g ? g.levels : [];
}

/* ==================================================================
   一键出图 与 快捷预设
   ================================================================== */
const QUICK_PRESETS = {
  simple: {
    label: "精简",
    modules: { runAlpha: true, runBeta: true, runOrd: false, runMulti: false,
               runWithin: false, runComp: true, runDiff: true, runLda: false,
               runRf: false, runAdv: true },
    ordinations: [],
  },
  standard: {
    label: "标准",
    modules: { runAlpha: true, runBeta: true, runOrd: true, runMulti: true,
               runWithin: true, runComp: true, runDiff: true, runLda: true,
               runRf: true, runAdv: true },
    ordinations: ["pcoa", "nmds", "pca", "tsne", "umap", "plsda"],
  },
  full: {
    label: "全量",
    modules: { runAlpha: true, runBeta: true, runOrd: true, runMulti: true,
               runWithin: true, runComp: true, runDiff: true, runLda: true,
               runRf: true, runAdv: true },
    ordinations: ["pcoa", "nmds", "pca", "tsne", "umap", "plsda",
                  "rda", "dbrda", "cca", "cap"],
  },
  diff: {
    label: "侧重差异",
    modules: { runAlpha: true, runBeta: false, runOrd: false, runMulti: false,
               runWithin: false, runComp: true, runDiff: true, runLda: true,
               runRf: true, runAdv: true },
    ordinations: [],
  },
  ordination: {
    label: "侧重排序",
    modules: { runAlpha: true, runBeta: true, runOrd: true, runMulti: true,
               runWithin: true, runComp: false, runDiff: false, runLda: false,
               runRf: false, runAdv: true },
    ordinations: ["pcoa", "nmds", "pca", "tsne", "umap", "plsda",
                  "rda", "dbrda", "cca", "cap"],
  },
};

function applyQuickPreset(name) {
  const p = QUICK_PRESETS[name];
  if (!p) return;
  Object.entries(p.modules).forEach(([id, v]) => {
    const el = document.getElementById(id);
    if (el) el.checked = v;
  });
  if (p.ordinations.length) {
    S.chosenOrd = p.ordinations.slice();
    $$("#ordChecks input").forEach((i) => { i.checked = p.ordinations.includes(i.value); });
  }
  S.chosenStats = ["permanova", "anosim", "mrpp", "permdisp", "mantel", "envfit"];
  $$("#statChecks input").forEach((i) => { i.checked = true; });
  toast(`已应用「${p.label}」预设`);
}

async function quickRun() {
  if (!S.files.abundance || !S.files.metadata) {
    toast("请先选择丰度表和分组表（分组表为必填）", 4200);
    goStep(1);
    return;
  }
  try {
    const res = await api("/api/inspect", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildInspectPayload()),
    });
    S.inspect = res;
    if (!res.group_candidates || !res.group_candidates.length) {
      toast("分组表里没有可用的分组列，请手动检查", 5000);
      return;
    }
    const sel = $("#groupColumn");
    sel.innerHTML = res.group_candidates
      .map((g) => `<option value="${esc(g.column)}">${esc(g.column)}（${g.n_levels} 组）</option>`).join("");
    renderGroupPreview();
  } catch (e) {
    toast("数据检查失败：" + e.message, 6000);
    return;
  }
  applyQuickPreset("full");
  toast("正在用全量默认设置出图…", 3200);
  await runAnalysis();
}

/* ==================================================================
   绘图样式编辑器
   ================================================================== */
function styleKey(key) {
  return S.styleScope ? `fig.${S.styleScope}.${key}` : key;
}

function styleValue(key) {
  const scoped = styleKey(key);
  if (S.style[scoped] !== undefined) return S.style[scoped];
  if (S.style[key] !== undefined) return S.style[key];
  const item = (S.config?.style_schema || []).find((s) => s.key === key);
  return item ? item.default : "";
}

function isOverridden(key) {
  return S.style[styleKey(key)] !== undefined && !!S.styleScope;
}

function setStyleValue(key, value) {
  const scoped = styleKey(key);
  const item = (S.config?.style_schema || []).find((s) => s.key === key && s.type === "number");
  if (item && typeof value === "number") {
    value = Math.min(item.max, Math.max(item.min, value));
  }
  S.style[scoped] = value;
  queuePreview();
}

function setStyleValueSilent(key, value) {
  S.style[styleKey(key)] = value;
}

function buildStyleScopeOptions() {
  const sel = $("#styleScope");
  const figs = S.config.previewable || [];
  sel.innerHTML = `<option value="">全局（所有图）</option>` +
    figs.map((f) => `<option value="${esc(f.id)}">单图：${esc(f.label)}</option>`).join("");
}

function syncScopeOptions() {
  // 图选择变化时，若正处于"单图"范围，则把范围同步到该图
  if (S.styleScope) {
    S.styleScope = $("#previewFigure").value;
    $("#styleScope").value = S.styleScope;
    renderStyleForm();
  }
}

function applyPreset() {
  const name = $("#stylePreset").value;
  const presetMap = S.config.preset_values || {};
  // 预设值由后端 style_schema 的默认值 + 前端内置映射决定
  const builtin = {
    "默认": {},
    "论文投稿": {
      "font.scale": 0.95, "figure.dpi": 600, "font.title.weight": "bold",
      "font.title.size": 11, "font.axis_label.size": 10, "font.tick_label.size": 8.5,
      "font.legend_text.size": 8.5, "font.annotation.size": 8,
      "element.grid": false, "element.point_alpha": 1.0,
      "palette.preset": "okabe_ito", "element.axis_rotate": 45,
    },
    "演示大屏": {
      "font.scale": 1.45, "figure.scale": 1.15, "figure.dpi": 200,
      "font.title.weight": "bold", "element.point_size": 130,
      "element.grid_alpha": 0.3, "palette.preset": "tableau10", "legend.ncol": 2,
    },
    "极简黑白": {
      "palette.preset": "grayscale", "element.grid": false,
      "element.ellipse": false, "font.title.weight": "bold",
    },
    "答辩配色": {
      "palette.preset": "nejm", "font.scale": 1.2, "figure.scale": 1.1,
      "element.point_size": 100, "legend.ncol": 2,
    },
  };
  const values = builtin[name] || {};
  // 清除同范围的旧键，再写入预设
  Object.keys(S.style).forEach((k) => {
    if (!k.startsWith("fig.")) delete S.style[k];
  });
  Object.assign(S.style, values);
  renderStyleForm();
  queuePreview(true);
}

function renderStyleForm() {
  const schema = S.config?.style_schema || [];
  const tabs = S.config?.style_groups || [];
  const body = $("#styleBody");
  const tabBar = $("#styleTabs");

  // 标签栏（跳过"位置"——它并入"文字"标签展示）
  const visibleTabs = tabs.filter((t) => t !== "位置");
  if (!visibleTabs.includes(S.styleTab)) S.styleTab = visibleTabs[0];
  tabBar.innerHTML = visibleTabs.map((t) =>
    `<button class="style-tab ${t === S.styleTab ? "active" : ""}" data-tab="${esc(t)}">${esc(t)}</button>`).join("");
  $$("#styleTabs .style-tab").forEach((b) => {
    b.onclick = () => { S.styleTab = b.dataset.tab; renderStyleForm(); };
  });

  const items = schema.filter((s) => s.group === S.styleTab ||
    (S.styleTab === "文字" && s.group === "位置"));
  if (!items.length) { body.innerHTML = '<div class="muted">该分组暂无可调项。</div>'; return; }

  const scopeNote = S.styleScope
    ? `<div class="scope-banner">正在编辑单图覆盖：<b>${esc(S.styleScope)}</b>
       <button class="btn small ghost" id="btnClearScope">清除该图覆盖</button></div>`
    : "";

  const picker = (S.styleTab === "配色") ? renderPalettePicker() : "";
  body.innerHTML = scopeNote + picker + items.map((item) => renderStyleField(item)).join("");

  // 绑定
  $$("#styleBody [data-skey]").forEach((el) => {
    const key = el.dataset.skey;
    const type = el.dataset.stype;
    const handler = () => {
      let v;
      if (type === "bool") v = el.checked;
      else if (type === "number") v = Number(el.value);
      else if (type === "colorlist") v = el.value.split(/[,\s]+/).map((s) => s.trim()).filter(Boolean);
      else if (type === "colormap") {
        try { v = JSON.parse(el.value || "{}"); } catch { return; }
      } else v = el.value;
      setStyleValue(key, v);
      if (type === "number" || type === "color") {
        const out = $("#out_" + cssId(key));
        if (out) out.textContent = el.value;
      }
      if (key === "palette.preset" || key === "palette.custom" || key === "palette.group_map") {
        renderGroupPreview();
      }
    };
    el.addEventListener(type === "number" ? "input" : "change", handler);
    if (type === "text" || type === "colorlist") el.addEventListener("input", debounce(handler, 500));
  });
  const clearBtn = $("#btnClearScope");
  if (clearBtn) {
    clearBtn.onclick = () => {
      const prefix = `fig.${S.styleScope}.`;
      Object.keys(S.style).forEach((k) => { if (k.startsWith(prefix)) delete S.style[k]; });
      renderStyleForm();
      queuePreview(true);
    };
  }

  // 分组配色：每个颜色选择器写回 palette.group_map
  const gcolors = $$("#styleBody [data-gcolor]");
  if (gcolors.length) {
    gcolors.forEach((el) => {
      el.addEventListener("input", () => {
        const map = {};
        $$("#styleBody [data-gcolor]").forEach((x) => { map[x.dataset.gcolor] = x.value; });
        const hidden = $("#groupMapInput");
        if (hidden) hidden.value = JSON.stringify(map);
        setStyleValue("palette.group_map", map);
        renderGroupPreview();
      });
    });
  }

  // 科研配色选择器
  if (S.styleTab === "配色") bindPalettePicker();
}

function cssId(s) { return s.replace(/[^\w]/g, "_"); }

function renderStyleField(item) {
  const key = item.key;
  const val = styleValue(key);
  const overridden = isOverridden(key);
  const badge = overridden ? '<span class="ovr" title="该图已单独覆盖">单图</span>' : "";
  const help = item.help ? `<div class="tip">${esc(item.help)}</div>` : "";
  const label = `<label>${esc(item.label)} ${badge}</label>`;

  if (item.type === "bool") {
    return `<div class="sf"><label class="switch"><input type="checkbox" data-skey="${esc(key)}"
      data-stype="bool" ${val ? "checked" : ""}><span>${esc(item.label)}</span></label>${help}</div>`;
  }

  let control = "";
  if (item.type === "number") {
    control = `<div class="num-row">
      <input type="range" data-skey="${esc(key)}" data-stype="number"
        min="${item.min}" max="${item.max}" step="${item.step}" value="${val}">
      <output id="out_${cssId(key)}">${val}</output></div>`;
  } else if (item.type === "select") {
    control = `<select data-skey="${esc(key)}" data-stype="select">` +
      item.options.map((o) => `<option value="${esc(o)}" ${String(o) === String(val) ? "selected" : ""}>${esc(o)}</option>`).join("") +
      `</select>`;
  } else if (item.type === "color") {
    control = `<div class="color-row">
      <input type="color" data-skey="${esc(key)}" data-stype="color" value="${esc(val || "#000000")}">
      <input type="text" data-skey="${esc(key)}" data-stype="color" value="${esc(val || "")}" style="flex:1">
    </div>`;
  } else if (item.type === "font") {
    const fonts = S.fonts || [];
    control = `<select data-skey="${esc(key)}" data-stype="font">
      <option value="">自动（推荐）</option>
      ${fonts.map((f) => `<option value="${esc(f)}" ${f === val ? "selected" : ""}>${esc(f)}</option>`).join("")}
    </select>`;
  } else if (item.type === "colorlist") {
    control = `<input type="text" data-skey="${esc(key)}" data-stype="colorlist"
      value="${esc((val || []).join(", "))}" placeholder="#4C72B0, #DD8452, #55A868">`;
  } else if (item.type === "colormap") {
    control = renderGroupColorEditor(val);
  } else {
    control = `<input type="text" data-skey="${esc(key)}" data-stype="text" value="${esc(val || "")}">`;
  }
  return `<div class="sf">${label}${control}${help}</div>`;
}

/* ---------------- 科研配色选择器 ---------------- */
S.schemeName = null;
S.pickedColors = [];

function currentSchemeName() {
  const research = S.config?.research_palette_names || [];
  if (S.schemeName && research.includes(S.schemeName)) return S.schemeName;
  const preset = String(S.style["palette.preset"] || "");
  if (research.includes(preset)) return preset;
  // 自定义颜色是否能完整匹配某一套方案
  const custom = (S.style["palette.custom"] || []).map((c) => String(c).toUpperCase());
  if (custom.length) {
    const all = S.config?.research_palettes || {};
    for (const [name, cols] of Object.entries(all)) {
      const norm = cols.map((c) => String(c).toUpperCase());
      if (norm.length === custom.length && norm.every((c, i) => c === custom[i])) return name;
    }
  }
  return research[0] || null;
}

function schemeColors(name) {
  const all = S.config?.research_palettes || {};
  return (all[name] || []).map((c) => String(c).toUpperCase());
}

function renderPalettePicker() {
  const names = S.config?.research_palette_names || [];
  if (!names.length) return "";
  const cur = currentSchemeName();
  const colors = schemeColors(cur);
  const picked = new Set(S.pickedColors.map((c) => String(c).toUpperCase()));

  const swatches = colors.map((c, i) => {
    const on = picked.has(c);
    return `<button class="swatch ${on ? "on" : ""}" data-color="${esc(c)}"
      title="${esc(c)}　点击${on ? "取消" : "选中"}"
      style="background:${esc(c)}"><span class="sw-num">${i + 1}</span>
      ${on ? '<span class="sw-tick">✓</span>' : ""}</button>`;
  }).join("");

  const preview = S.pickedColors.length
    ? S.pickedColors.map((c) => `<span class="pv-chip" style="background:${esc(c)}"></span>`).join("")
    : '<span class="muted small">（尚未选择，将使用整套方案）</span>';

  return `
  <div class="palette-picker">
    <div class="pp-head">
      <label>科研配色方案</label>
      <select id="schemeSelect">
        ${names.map((n) => `<option value="${esc(n)}" ${n === cur ? "selected" : ""}>${esc(n)}（${schemeColors(n).length} 色）</option>`).join("")}
      </select>
      <span class="muted small">共 ${names.length} 套，来自你的配色方案表</span>
    </div>
    <div class="pp-swatches" id="schemeSwatches">${swatches}</div>
    <div class="pp-actions">
      <button class="btn small primary" id="btnUseWholeScheme">整套使用「${esc(cur || "")}」</button>
      <button class="btn small" id="btnUsePicked" ${S.pickedColors.length ? "" : "disabled"}>
        使用选中的 ${S.pickedColors.length} 种颜色</button>
      <button class="btn small ghost" id="btnClearPicked" ${S.pickedColors.length ? "" : "disabled"}>清空选择</button>
    </div>
    <div class="pp-preview">当前自定义配色：${preview}</div>
    <div class="tip">点色块可多选。选中的颜色会按顺序分配给各分组，也可在下面「分组指定颜色」里逐个微调。</div>
  </div>`;
}

function bindPalettePicker() {
  const sel = $("#schemeSelect");
  if (sel) {
    sel.onchange = () => { S.schemeName = sel.value; S.pickedColors = []; renderStyleForm(); };
  }
  const sw = $$("#schemeSwatches .swatch");
  sw.forEach((b) => {
    b.onclick = () => {
      const c = String(b.dataset.color).toUpperCase();
      const idx = S.pickedColors.findIndex((x) => String(x).toUpperCase() === c);
      if (idx >= 0) S.pickedColors.splice(idx, 1);
      else S.pickedColors.push(c);
      renderStyleForm();
    };
  });
  const whole = $("#btnUseWholeScheme");
  if (whole) {
    whole.onclick = () => {
      const name = currentSchemeName();
      S.schemeName = name;
      S.pickedColors = [];
      delete S.style["palette.custom"];
      S.style["palette.preset"] = name;
      renderStyleForm();
      renderGroupPreview();
      queuePreview(true);
      toast(`已整套使用「${name}」`);
    };
  }
  const usePicked = $("#btnUsePicked");
  if (usePicked) {
    usePicked.onclick = () => {
      S.style["palette.custom"] = S.pickedColors.slice();
      renderStyleForm();
      renderGroupPreview();
      queuePreview(true);
      toast(`已使用选中的 ${S.pickedColors.length} 种颜色`);
    };
  }
  const clearPicked = $("#btnClearPicked");
  if (clearPicked) {
    clearPicked.onclick = () => { S.pickedColors = []; renderStyleForm(); };
  }
}

function renderGroupColorEditor(current) {
  const groups = currentGroups();
  const map = current || {};
  if (!groups.length) {
    return `<div class="muted small">先在「设置分组」里选定分组列，这里就能为每组指定颜色。</div>
      <input type="hidden" data-skey="palette.group_map" data-stype="colormap" value="${esc(JSON.stringify(map))}">`;
  }
  const auto = paletteColors();
  const rows = groups.map((g, i) => {
    const c = map[g] || auto[i % auto.length];
    return `<div class="gcolor-row"><span class="gcolor-name">${esc(g)}</span>
      <input type="color" value="${esc(c)}" data-gcolor="${esc(g)}"></div>`;
  }).join("");
  return `<div class="gcolor-list">${rows}</div>
    <input type="hidden" data-skey="palette.group_map" data-stype="colormap"
      id="groupMapInput" value="${esc(JSON.stringify(map))}">`;
}

/* ---------------- 样式预览 ---------------- */
function debounce(fn, ms) {
  let t;
  return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

function queuePreview(force = false) {
  clearTimeout(S.previewTimer);
  S.previewTimer = setTimeout(() => doPreview(force), force ? 0 : 420);
}

async function doPreview(force = false) {
  if (!$("#previewFigure")) return;
  if (!S.files.abundance || !S.files.metadata) {
    $("#previewFrame").innerHTML = '<div class="preview-empty">请先在「选择数据」里指定丰度表和分组表</div>';
    return;
  }
  if (S.previewBusy) { queuePreview(force); return; }
  S.previewBusy = true;

  const frame = $("#previewFrame");
  frame.innerHTML = '<div class="preview-empty">渲染中…</div>';
  try {
    const res = await api("/api/style/preview", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        config: buildAnalysisConfig(),
        style: { values: S.style },
        figure_id: $("#previewFigure").value,
        force,
      }),
    });
    frame.dataset.loaded = "1";
    frame.innerHTML = `<img src="${res.url}" alt="预览">`;
    $("#previewCaption").textContent = res.caption || "";
  } catch (e) {
    frame.innerHTML = `<div class="preview-empty err">预览失败：${esc(e.message)}</div>`;
    $("#previewCaption").textContent = "";
  } finally {
    S.previewBusy = false;
  }
}

async function loadThemes() {
  try {
    const { themes } = await api("/api/style/themes");
    const sel = $("#themeSelect");
    sel.innerHTML = '<option value="">（未保存）</option>' +
      themes.map((t) => `<option value="${esc(t.name)}">${esc(t.name)}</option>`).join("");
  } catch { /* 忽略 */ }
}

async function saveTheme() {
  const name = $("#themeName").value.trim();
  if (!name) { toast("请输入主题名称"); return; }
  try {
    await api("/api/style/save", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, style: { values: S.style } }),
    });
    $("#saveThemeModal").classList.add("hidden");
    $("#themeName").value = "";
    await loadThemes();
    toast("主题已保存");
  } catch (e) { toast("保存失败：" + e.message, 5000); }
}

async function loadTheme() {
  const name = $("#themeSelect").value;
  if (!name) { toast("请先选择一个已保存主题"); return; }
  try {
    const { style } = await api(`/api/style/theme/${encodeURIComponent(name)}`);
    S.style = style.values || {};
    renderStyleForm();
    queuePreview(true);
    toast(`已载入主题：${name}`);
  } catch (e) { toast("载入失败：" + e.message, 5000); }
}

/* ==================================================================
   运行
   ================================================================== */
function buildAnalysisConfig() {
  const metrics = $$("#betaMetrics input:checked").map((i) => i.value);
  const ords = $$("#ordChecks input:checked").map((i) => i.value);
  const stats = $$("#statChecks input:checked").map((i) => i.value);
  const pairs = Array.from(S.selectedPairs).map((k) => k.split("||"));
  return {
    analysis_type: "amplicon16s",
    abundance_path: S.files.abundance?.path || "",
    metadata_path: S.files.metadata?.path || "",
    taxonomy_path: S.files.taxonomy?.path || "",
    env_path: S.files.env?.path || "",
    group_column: $("#groupColumn").value,
    compare_pairs: pairs,
    rank: $("#rankSelect").value || null,
    filter: {
      min_count: Number($("#minCount").value) || 0,
      min_prevalence: Number($("#minPrev").value) || 0,
      min_total_frac: Number($("#minFrac").value) || 0,
    },
    rarefy: $("#rarefy").checked,
    beta_metrics: metrics.length ? metrics : null,
    ordinations: ords.length ? ords : null,
    stats_tests: stats.length ? stats : null,
    ordination_transform: $("#ordTransform").value,
    ordination_metric: $("#ordMetric").value,
    umap_neighbors: Number($("#umapNeighbors").value) || 15,
    umap_min_dist: Number($("#umapMinDist").value) || 0.1,
    permutations: Number($("#perms").value) || 999,
    top_n_composition: Number($("#topComp").value) || 10,
    top_n_heatmap: Number($("#topHeat").value) || 30,
    top_n_diff_features: Number($("#topDiff").value) || 9,
    venn_threshold: Number($("#vennThreshold").value) || 0.5,
    hypothesis_method: $("#hypMethod").value || "auto",
    p_adjust: $("#pAdjust").value || "bh",
    posthoc: $("#posthoc").checked,
    run_alpha: $("#runAlpha").checked,
    run_beta: $("#runBeta").checked,
    run_ordination: $("#runOrd").checked,
    run_multivariate: $("#runMulti").checked,
    run_within_group: $("#runWithin").checked,
    run_composition: $("#runComp").checked,
    run_diff: $("#runDiff").checked,
    run_lda: $("#runLda").checked,
    run_rf: $("#runRf").checked,
    run_advanced: $("#runAdv").checked,
    style: { values: S.style },
  };
}

async function runAnalysis() {
  const cfg = buildAnalysisConfig();
  if (!cfg.abundance_path || !cfg.metadata_path) { toast("请先选择数据文件"); goStep(1); return; }
  if (!cfg.group_column) { toast("请选择分组列"); goStep(2); return; }
  if (cfg.run_beta && !cfg.beta_metrics) { toast("请至少选择一个距离指标"); goStep(3); return; }

  $("#progressBar").style.width = "0%";
  $("#progressStage").textContent = "提交任务…";
  $("#progressPct").textContent = "0%";
  $("#logBox").textContent = "";
  $("#btnToResult").classList.add("hidden");
  goStep(5);

  try {
    const res = await api("/api/run", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(cfg),
    });
    S.jobId = res.job_id;
    pollJob();
  } catch (e) {
    $("#progressStage").textContent = "启动失败";
    $("#logBox").textContent = e.message;
    toast("启动失败：" + e.message, 6000);
  }
}

function pollJob() {
  clearTimeout(S.pollTimer);
  const seenLines = new Set();
  const tick = async () => {
    try {
      const { job } = await api(`/api/job/${S.jobId}`);
      $("#progressBar").style.width = job.percent + "%";
      $("#progressPct").textContent = Math.round(job.percent) + "%";
      $("#progressStage").textContent = job.stage + (job.message ? " · " + job.message : "");

      if (job.status === "running") {
        const line = `[${String(job.percent).padStart(5)}%] ${job.stage}${job.message ? " · " + job.message : ""}`;
        if (!seenLines.has(line) && (job.message || job.stage)) {
          seenLines.add(line);
          $("#logBox").textContent += line + "\n";
          $("#logBox").scrollTop = $("#logBox").scrollHeight;
        }
        S.pollTimer = setTimeout(tick, 900);
      } else if (job.status === "done") {
        S.result = job.result;
        $("#progressStage").textContent = "完成";
        $("#progressBar").style.width = "100%";
        $("#progressPct").textContent = "100%";
        $("#logBox").textContent = (job.logs || []).join("\n");
        $("#btnToResult").classList.remove("hidden");
        renderResult(job.result);
        loadHistory();
        setTimeout(() => goStep(6), 700);
      } else {
        $("#progressStage").textContent = "运行失败";
        $("#logBox").textContent = (job.logs || []).join("\n") || job.error || "未知错误";
        toast("分析失败：" + (job.error || "请查看日志"), 8000);
      }
    } catch (e) {
      $("#progressStage").textContent = "任务状态读取失败";
      toast("读取任务状态失败：" + e.message, 5000);
    }
  };
  tick();
}

function resetRun() {
  clearTimeout(S.pollTimer);
  S.jobId = null; S.result = null;
  $("#resultBox").innerHTML = "";
  $("#logBox").textContent = "";
}

/* ---------------- 结果 ---------------- */
function renderNarrative(nar) {
  const box = $("#narrativeBox");
  if (!nar || !nar.sections) { box.innerHTML = ""; return; }
  const blocks = nar.sections.map(([title, sents]) => `
    <div class="nar-block">
      <h3>${esc(title)}</h3>
      <ul>${sents.map((s) => {
        const sub = String(s).startsWith("　·") || String(s).startsWith("·");
        const txt = esc(String(s).replace(/^[　\s]*·\s*/, ""))
          .replace(/&lt;b&gt;(.*?)&lt;\/b&gt;/g, "<b>$1</b>")
          .replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");
        return `<li class="${sub ? "nar-sub" : ""}">${txt}</li>`;
      }).join("")}</ul>
    </div>`).join("");

  box.innerHTML = `
    <div class="nar-head">
      <h3>分析结论</h3>
      <button class="btn small ghost" id="btnCopyNar">复制全文</button>
    </div>
    <div class="alert ok nar-headline">${esc(nar.headline || "").replace(/\*\*(.+?)\*\*/g, "<b>$1</b>")}</div>
    ${blocks}
    <div class="muted small" style="margin-bottom:18px">
      以上文字由程序根据本次实际统计结果自动生成；完整版见结果目录下的
      <code>结论描述.md</code> 与 <code>conclusions.txt</code>。
    </div>`;

  const btn = $("#btnCopyNar");
  if (btn) {
    btn.onclick = async () => {
      const plain = [nar.headline, ""].concat(
        (nar.sections || []).flatMap(([t, ss]) => [t, ...ss.map((s) => "  " + s)]).concat([""])
      ).join("\n");
      try {
        await navigator.clipboard.writeText(plain);
        toast("结论已复制到剪贴板");
      } catch {
        toast("浏览器不允许自动复制，请手动选中", 4000);
      }
    };
  }
}

function renderResult(r) {
  if (!r) return;
  renderNarrative(r.narrative);
  const cards = [];
  const info = r.input_info || {};
  const add = (k, v, n = "") =>
    cards.push(`<div class="card"><div class="k">${k}</div><div class="v">${v}</div>${n ? `<div class="n">${n}</div>` : ""}</div>`);

  add("数据类型", r.analysis_label || "-");
  add("样本 / 特征", `${info.n_samples ?? "-"} / ${info.n_features_used ?? info.n_features ?? "-"}`);
  add("分组", Object.keys(r.groups || {}).length,
    Object.entries(r.groups || {}).map(([k, v]) => `${esc(k)}=${v}`).join("  "));
  if (r.random_forest) {
    const rf = r.random_forest;
    add("RF 准确率", (rf.accuracy * 100).toFixed(1) + "%",
      rf.auc ? `AUC = ${rf.auc.toFixed(3)}` : `${rf.n_splits} 折交叉验证`);
  }
  add("生成图表", (r.figures || []).length, `耗时 ${r.duration_sec ?? "-"} 秒`);

  const betaRows = Object.values(r.beta || {}).map((b) => {
    const p = b.permanova || {}, a = b.anosim || {}, m = b.mrpp || {}, d = b.permdisp || {};
    return `<tr><td>${esc(b.label)}</td>
      <td>${num3(p.R2)}</td><td>${num3(p.F)}</td><td class="${sigCls(p.p)}">${fmtP(p.p)}${stars(p.p)}</td>
      <td>${num3(a.R)}</td><td class="${sigCls(a.p)}">${fmtP(a.p)}</td>
      <td>${num3(m.chance_corrected_within_group_agreement)}</td>
      <td class="${sigCls(m.p)}">${fmtP(m.p)}</td>
      <td>${num3(d.F)}</td><td class="${sigCls(d.p)}">${fmtP(d.p)}</td></tr>`;
  }).join("");

  const alphaRows = Object.entries(r.alpha_tests || {}).map(([k, t]) =>
    `<tr><td>${esc(k)}</td><td>${esc(t.test)}</td>
     <td class="${sigCls(t.p)}">${fmtP(t.p)}${stars(t.p)}</td></tr>`).join("");

  const envRows = (r.envfit || []).map((v) =>
    `<tr><td>${esc(v.variable)}</td><td>${esc(v.type || "-")}</td><td>${num3(v.r2)}</td>
     <td class="${sigCls(v.p)}">${fmtP(v.p)}</td>
     <td class="${sigCls(v.p_adj)}">${fmtP(v.p_adj)}</td></tr>`).join("");

  const warn = (r.warnings || []).length
    ? `<div class="alert warn"><b>运行提示</b><ul>${r.warnings.map((w) => `<li>${esc(w)}</li>`).join("")}</ul></div>`
    : "";
  const dir = (r.output_dir || "").replace(/\\/g, "\\\\");

  $("#resultBox").innerHTML = `
    <div class="cards">${cards.join("")}</div>
    ${warn}
    <div class="result-actions">
      <a class="btn primary" href="/output/${r.run_id}/report.html" target="_blank" rel="noopener">打开完整 HTML 报告</a>
      <button class="btn" onclick="openFolder('${dir}')">打开结果文件夹</button>
      <a class="btn ghost" href="/output/${r.run_id}/tables/analysis_summary.xlsx" download>下载 Excel 汇总</a>
    </div>

    <div class="sec-title">Beta 多样性多元检验</div>
    <table class="stat-table">
      <tr><th>距离指标</th><th>PERMANOVA R²</th><th>F</th><th>p</th>
          <th>ANOSIM R</th><th>p</th><th>MRPP A</th><th>p</th><th>PERMDISP F</th><th>p</th></tr>
      ${betaRows || '<tr><td colspan="10" class="muted">未执行</td></tr>'}
    </table>
    ${alphaRows ? `<div class="sec-title">Alpha 多样性组间检验</div>
      <table class="stat-table"><tr><th>指标</th><th>检验方法</th><th>p 值</th></tr>${alphaRows}</table>` : ""}
    ${envRows ? `<div class="sec-title">envfit 因子拟合</div>
      <table class="stat-table"><tr><th>变量</th><th>类型</th><th>R²</th><th>p</th><th>p (校正)</th></tr>${envRows}</table>` : ""}
    ${renderGallery(r.figures || [])}
  `;
}

function renderGallery(figs) {
  const bySec = {};
  figs.forEach((f) => { (bySec[f.section] = bySec[f.section] || []).push(f); });
  const secs = Object.keys(bySec);

  const chips = [`<button class="gchip-filter active" data-sec="">全部 (${figs.length})</button>`]
    .concat(secs.map((s) => `<button class="gchip-filter" data-sec="${esc(s)}">${esc(s)} (${bySec[s].length})</button>`))
    .join("");

  const groups = secs.map((sec) => `
    <div class="gal-section" data-sec="${esc(sec)}">
      <div class="sec-title">${esc(sec)}
        <span class="muted small" style="font-weight:400;margin-left:8px">${bySec[sec].length} 图</span>
      </div>
      <div class="gallery">
        ${bySec[sec].map((f) => `
          <div class="figcard">
            <img src="${f.url}" alt="${esc(f.name)}" loading="lazy" onclick="showLightbox('${f.url}')">
            <div class="cap"><b>${esc(f.title || f.name)}</b>${esc(f.caption || "")}
              <div class="fig-actions">
                <a class="mini-btn" href="${f.url}" download>下载 PNG</a>
                <a class="mini-btn" href="${f.url.replace(/\.png$/i, ".pdf")}" download>下载 PDF</a>
                <a class="mini-btn" href="${f.url}" target="_blank" rel="noopener">原图</a>
              </div>
            </div>
          </div>`).join("")}
      </div>
    </div>`).join("");

  setTimeout(() => {
    $$(".gchip-filter").forEach((c) => {
      c.onclick = () => {
        $$(".gchip-filter").forEach((x) => x.classList.remove("active"));
        c.classList.add("active");
        const sec = c.dataset.sec;
        $$(".gal-section").forEach((g) => {
          g.style.display = (!sec || g.dataset.sec === sec) ? "" : "none";
        });
      };
    });
  }, 0);

  return `<div class="gal-filter">${chips}</div>${groups}`;
}

function showLightbox(url) {
  const lb = document.createElement("div");
  lb.className = "lightbox";
  lb.id = "lightbox";
  lb.innerHTML = `<img src="${url}" alt="">`;
  lb.onclick = closeLightbox;
  document.body.appendChild(lb);
}
function closeLightbox() {
  const lb = $("#lightbox");
  if (lb) lb.remove();
}

async function openFolder(path) {
  try {
    await api("/api/open_folder", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path: path.replace(/\\\\/g, "\\") }),
    });
    toast("已打开结果文件夹");
  } catch (e) { toast("打开失败：" + e.message, 5000); }
}

/* ---------------- 演示数据 / 历史 ---------------- */
async function loadDemo() {
  try {
    const d = await api("/api/demo");
    if (!d.datasets.length) { toast("没有可用的演示数据"); return; }
    const pick = d.datasets[0];
    applyDataset(pick);
    toast("已载入演示数据：" + pick.label, 3400);
  } catch (e) { toast("演示数据不可用：" + e.message, 5000); }
}

function applyDataset(ds) {
  const map = {
    abundance: ds.abundance_path,
    metadata: ds.metadata_path,
    taxonomy: ds.taxonomy_path,
    env: ds.env_path,
  };
  Object.entries(map).forEach(([kind, p]) => {
    const dz = document.querySelector(`.dropzone[data-kind="${kind}"]`);
    if (!dz) return;
    if (p) {
      S.files[kind] = { path: p, name: p.split(/[\\/]/).pop() };
      dz.classList.add("filled");
      dz.querySelector(".dz-file").textContent = `✓ ${S.files[kind].name}`;
    } else {
      S.files[kind] = null;
      dz.classList.remove("filled");
      dz.querySelector(".dz-file").textContent = "";
    }
  });
  applyAnalysisType();
  updateInspectButton();
  toast("演示数据已就绪，点击「下一步：检查数据」", 3200);
}

async function loadHistory() {
  try {
    const { runs } = await api("/api/runs");
    const box = $("#historyList");
    if (!runs.length) { box.innerHTML = '<div class="muted">还没有分析记录。</div>'; return; }
    box.innerHTML = runs.map((r) => `
      <div class="hitem">
        <div>
          <div><b>${esc(r.analysis_label || "-")}</b>
            <span class="muted small"> · ${esc(r.run_id)}</span></div>
          <div class="meta">${esc(r.finished_at || "")} · 样本 ${r.n_samples ?? "-"} ·
            ${Object.entries(r.groups || {}).map(([k, v]) => `${esc(k)}=${v}`).join(" ")} ·
            ${r.n_figures} 图 · ${r.duration_sec ?? "-"} 秒</div>
        </div>
        <div style="display:flex;gap:8px;flex-shrink:0">
          <a class="btn small" href="${r.report_url}" target="_blank" rel="noopener">报告</a>
          <button class="btn small ghost" onclick="openFolder('${(r.dir || "").replace(/\\/g, "\\\\")}')">文件夹</button>
        </div>
      </div>`).join("");
  } catch { /* 静默 */ }
}

/* ---------------- 小工具 ---------------- */
function esc(s) {
  return String(s == null ? "" : s)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}
function num3(v) {
  if (v == null || !isFinite(v)) return "-";
  return Number(v).toFixed(3);
}
function fmtP(p) {
  if (p == null || !isFinite(p)) return "n/a";
  return p < 0.001 ? "< 0.001" : Number(p).toFixed(3);
}
function stars(p) {
  if (p == null || !isFinite(p)) return "";
  return p < 0.001 ? " ***" : p < 0.01 ? " **" : p < 0.05 ? " *" : "";
}
function sigCls(p) {
  return (p != null && isFinite(p) && p < 0.05) ? "sig" : "";
}

/* 字体列表按需加载 */
(async () => {
  try {
    const f = await api("/api/fonts");
    S.fonts = f.all || [];
  } catch { S.fonts = []; }
})();

boot();
