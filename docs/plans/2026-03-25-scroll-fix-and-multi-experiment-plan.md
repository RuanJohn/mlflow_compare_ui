# Scroll Fix + Multi-Experiment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix auto-refresh scroll drift by updating chart data in-place, and add multi-experiment support so users can compare runs across experiments.

**Architecture:** Single-file vanilla JS app (`templates/index.html`). All changes are frontend-only except for a minor state model update in the view serialization. Backend endpoints already support arbitrary run IDs across experiments.

**Tech Stack:** Vanilla JS, uPlot 1.6.31, Flask (backend unchanged)

---

### Task 1: Fix auto-refresh scroll — extract data builder and update pollMetrics

**Files:**
- Modify: `templates/index.html` (lines ~868-936, ~982-1246)

**Step 1: Add `lastChartGroups` state variable**

At line 869, after `let lastChartMetrics = [];`, add:

```js
let lastChartGroups = [];  // [{name, metrics}] — preserved for in-place poll updates
```

**Step 2: Extract `buildGroupUData` function**

Insert before `drawChartsFromData` (before line 982). This function takes one chart group's entries and returns the uPlot data arrays + series config without touching the DOM:

```js
function buildGroupUData(allEntries, runColorMap, useWall, smoothAlpha) {
  const multiMetric = new Set(allEntries.map(e => e._metric)).size > 1;
  let xArr, series, uData, xLabel;

  if (useWall && allEntries[0].timestamps && allEntries[0].timestamps.length) {
    xLabel = "Hours";
    const relData = allEntries.map(entry => {
      const t0 = entry.timestamps[0] || 0;
      return {
        xs: entry.timestamps.map(ts => (ts - t0) / 3600000),
        ys: entry.values,
        entry,
      };
    });
    const allX = new Set();
    relData.forEach(d => d.xs.forEach(x => allX.add(x)));
    xArr = [...allX].sort((a, b) => a - b);
    const xIdx = {};
    xArr.forEach((v, i) => { xIdx[v] = i; });

    series = [{ label: xLabel }];
    uData = [new Float64Array(xArr)];

    relData.forEach((d, i) => {
      const metricSuffix = multiMetric ? ` [${d.entry._metric.split("/").pop()}]` : "";
      const label = getRunLabel(d.entry.run_id) + metricSuffix;
      series.push({
        label,
        stroke: runColorMap[d.entry.run_id] || COLORS[i % COLORS.length],
        width: 1.5,
      });
      let yArr = new Float64Array(xArr.length).fill(NaN);
      for (let j = 0; j < d.xs.length; j++) {
        yArr[xIdx[d.xs[j]]] = d.ys[j];
      }
      if (smoothAlpha > 0) yArr = emaSmooth(yArr, smoothAlpha);
      uData.push(yArr);
    });
  } else {
    xLabel = "Step";
    const allSteps = new Set();
    allEntries.forEach(e => e.steps.forEach(s => allSteps.add(s)));
    xArr = [...allSteps].sort((a, b) => a - b);
    const xIdx = {};
    xArr.forEach((v, i) => { xIdx[v] = i; });

    series = [{ label: xLabel }];
    uData = [new Float64Array(xArr)];

    allEntries.forEach((entry, i) => {
      const metricSuffix = multiMetric ? ` [${entry._metric.split("/").pop()}]` : "";
      const label = getRunLabel(entry.run_id) + metricSuffix;
      series.push({
        label,
        stroke: runColorMap[entry.run_id] || COLORS[i % COLORS.length],
        width: 1.5,
      });
      let yArr = new Float64Array(xArr.length).fill(NaN);
      for (let j = 0; j < entry.steps.length; j++) {
        yArr[xIdx[entry.steps[j]]] = entry.values[j];
      }
      if (smoothAlpha > 0) yArr = emaSmooth(yArr, smoothAlpha);
      uData.push(yArr);
    });
  }

  return { uData, series, xLabel, xArr };
}
```

**Step 3: Refactor `drawChartsFromData` to use `buildGroupUData`**

Replace the data-building sections inside the `groups.forEach` loop (the wall-time and step-mode branches, roughly lines 1044-1106) with a call to `buildGroupUData`. Also store groups in `lastChartGroups`.

The refactored `drawChartsFromData` should:
1. Set `lastChartGroups = groups;` after computing groups
2. Inside the group loop, after collecting `allEntries`, call `buildGroupUData(allEntries, runColorMap, useWall, smoothAlpha)` to get `{uData, series, xLabel, xArr}`
3. Use the returned values for the uPlot opts and construction (same as before)
4. Remove the inline data-building code that was extracted

**Step 4: Rewrite `pollMetrics` to use `setData`**

Replace the current `pollMetrics` function (lines 899-936) with:

```js
async function pollMetrics() {
  const runIds = [...selectedRunIds];
  const metrics = lastChartMetrics;
  if (!runIds.length || !metrics.length) {
    stopPolling();
    return;
  }
  try {
    const data = await api("/api/metric-history", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({run_ids: runIds, metrics, skip_cache: true}),
    });
    lastChartData = data;

    // Build lookup from new data
    const byMetric = {};
    data.results.forEach(r => {
      if (!byMetric[r.metric]) byMetric[r.metric] = [];
      byMetric[r.metric].push(r);
    });

    const runColorMap = {};
    let colorIdx = 0;
    [...selectedRunIds].forEach(id => {
      runColorMap[id] = COLORS[colorIdx % COLORS.length];
      colorIdx++;
    });

    const useWall = getXAxisMode() === "wall";
    const smoothAlpha = getSmoothingFactor();

    // If chart count matches, update in-place; otherwise full re-render
    if (lastChartGroups.length === uplotInstances.length) {
      lastChartGroups.forEach((group, i) => {
        const allEntries = [];
        group.metrics.forEach(metric => {
          const entries = (byMetric[metric] || []).filter(e => e.steps && e.steps.length > 0);
          entries.forEach(e => allEntries.push({ ...e, _metric: metric }));
        });
        if (allEntries.length) {
          const { uData } = buildGroupUData(allEntries, runColorMap, useWall, smoothAlpha);
          uplotInstances[i].setData(uData, false);  // false = preserve zoom/scales
        }
      });
    } else {
      // Fallback: full re-render (rare — metric set changed)
      const scrollEl = document.querySelector(".content");
      const savedScroll = scrollEl ? scrollEl.scrollTop : 0;
      uplotInstances.forEach(u => u.destroy());
      uplotInstances = [];
      $chartGrid.innerHTML = "";
      const cols = parseInt($colSlider.value, 10);
      $chartGrid.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
      drawChartsFromData(data, metrics);
      if (scrollEl) requestAnimationFrame(() => { scrollEl.scrollTop = savedScroll; });
    }
  } catch (err) {
    console.warn("Auto-refresh poll failed:", err);
  }
}
```

**Step 5: Verify and commit**

Run: Open browser, load experiment, select runs, click Compare. Scroll down. Wait 30s for auto-refresh. Confirm scroll position stays fixed and chart data updates.

```bash
git add templates/index.html
git commit -m "fix: update chart data in-place during auto-refresh to prevent scroll drift"
```

---

### Task 2: Multi-experiment — CSS and topbar HTML

**Files:**
- Modify: `templates/index.html` (CSS section ~lines 8-147, HTML ~lines 150-156)

**Step 1: Add experiment chip CSS**

Add after the `.btn-group` rule (after line 45):

```css
.exp-input-area{display:flex;gap:8px;align-items:center;flex:1;max-width:480px}
.exp-input-area input[type=text]{flex:1;padding:6px 10px;border:1px solid var(--border);border-radius:var(--radius);background:var(--bg);color:var(--fg);font-size:13px}
.exp-input-area input[type=text]:focus{outline:none;border-color:var(--accent)}
.exp-chips{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
.exp-chip{display:inline-flex;align-items:center;gap:4px;padding:3px 10px;background:var(--accent);color:#fff;border-radius:var(--radius);font-size:12px;white-space:nowrap}
.exp-chip-x{cursor:pointer;opacity:0.7;font-size:14px;line-height:1}
.exp-chip-x:hover{opacity:1}
```

**Step 2: Update topbar HTML**

Replace lines 150-156:

```html
<div class="topbar">
  <h1>MLflow Compare</h1>
  <div class="exp-input-area">
    <input type="text" id="expInput" placeholder="Experiment name, e.g. group/experiment">
    <button class="btn btn-primary" id="loadBtn">Add</button>
  </div>
  <div id="expChips" class="exp-chips"></div>
  <button class="btn" id="refreshBtn" title="Clear cache &amp; reload all">&#x21bb; Refresh</button>
  <span id="topStatus"></span>
</div>
```

**Step 3: Remove old topbar input CSS**

The old `.topbar input[type=text]` rules (lines 23-24) can be removed since the input is now styled via `.exp-input-area input[type=text]`.

**Step 4: Commit**

```bash
git add templates/index.html
git commit -m "feat: add experiment chip UI to topbar"
```

---

### Task 3: Multi-experiment — state model and core functions

**Files:**
- Modify: `templates/index.html` (JS section)

**Step 1: Replace `experimentId` with `loadedExperiments`**

At line 337, replace:

```js
let experimentId = null;
```

With:

```js
const loadedExperiments = new Map();  // name -> {id, runs[]}
```

Add a DOM ref for the chips container (near line 356):

```js
const $expChips = document.getElementById("expChips");
```

**Step 2: Add `rebuildAllRuns` function**

Insert after the helpers section (after `api` function, around line 395):

```js
function rebuildAllRuns() {
  allRuns = [];
  for (const [name, exp] of loadedExperiments) {
    allRuns.push(...exp.runs);
  }
  allRuns.sort((a, b) => new Date(b.start_time) - new Date(a.start_time));

  const metricSet = new Set();
  allRuns.forEach(r => (r.metric_keys || []).forEach(k => metricSet.add(k)));
  allMetricNames = [...metricSet].sort();

  selectedRunIds = new Set([...selectedRunIds].filter(id => allRuns.some(r => r.run_id === id)));
}

function getExpShortName(fullName) {
  return fullName.split("/").pop();
}

function renderExpChips() {
  $expChips.innerHTML = [...loadedExperiments.keys()].map(name =>
    `<span class="exp-chip">${esc(name)}<span class="exp-chip-x" data-exp="${esc(name)}">&times;</span></span>`
  ).join("");
}
```

**Step 3: Replace `loadExperiment` with `addExperiment` and `removeExperiment`**

Replace the `loadExperiment` function (lines 487-528) with:

```js
async function addExperiment(name) {
  if (loadedExperiments.has(name)) {
    showInfo(`Experiment "${name}" is already loaded.`, "");
    setTimeout(clearInfo, 2000);
    return;
  }

  stopPolling();
  clearInfo();
  showStatus('<span class="spinner"></span> Loading…');

  try {
    const expData = await api(`/api/experiment?name=${encodeURIComponent(name)}`);
    const runsData = await api(`/api/runs?experiment_id=${encodeURIComponent(expData.experiment_id)}`);

    const shortName = getExpShortName(name);
    runsData.runs.forEach(r => {
      r._experimentName = shortName;
      r._fullExperimentName = name;
    });

    loadedExperiments.set(name, { id: expData.experiment_id, runs: runsData.runs });
    rebuildAllRuns();
    renderExpChips();

    $expInfo.textContent = [...loadedExperiments.entries()]
      .map(([n, e]) => `${n} (${e.id})`).join(", ");

    renderTable();
    renderSelectedRuns();
    renderMetricSelector();
    $tableSection.style.display = allRuns.length ? "" : "none";
    $metricSection.style.display = allRuns.length ? "" : "none";
    clearStatus();
  } catch (err) {
    clearStatus();
    showInfo(err.message, "error");
  }
}

function removeExperiment(name) {
  if (!loadedExperiments.has(name)) return;
  stopPolling();

  const exp = loadedExperiments.get(name);
  exp.runs.forEach(r => {
    selectedRunIds.delete(r.run_id);
    customLegends.delete(r.run_id);
  });
  loadedExperiments.delete(name);
  Object.keys(runParamsCache).forEach(k => {
    if (exp.runs.some(r => r.run_id === k)) delete runParamsCache[k];
  });

  rebuildAllRuns();
  renderExpChips();

  $expInfo.textContent = [...loadedExperiments.entries()]
    .map(([n, e]) => `${n} (${e.id})`).join(", ");

  renderTable();
  renderSelectedRuns();
  renderMetricSelector();

  if (!allRuns.length) {
    $tableSection.style.display = "none";
    $metricSection.style.display = "none";
    $chartGrid.innerHTML = "";
    uplotInstances.forEach(u => u.destroy());
    uplotInstances = [];
  }
}
```

**Step 4: Commit**

```bash
git add templates/index.html
git commit -m "feat: add multi-experiment state model with add/remove functions"
```

---

### Task 4: Multi-experiment — table display and run labels

**Files:**
- Modify: `templates/index.html`

**Step 1: Update `renderTable` to show experiment/run name**

In `renderTable()` (line 541), change the run name cell from:

```js
<td>${esc(r.run_name)}</td>
```

To:

```js
<td>${r._experimentName ? esc(r._experimentName + '/' + r.run_name) : esc(r.run_name)}</td>
```

**Step 2: Update `getRunLabel` to include experiment prefix**

Replace `getRunLabel` (lines 628-632) with:

```js
function getRunLabel(runId) {
  if (customLegends.has(runId)) return customLegends.get(runId);
  const run = allRuns.find(r => r.run_id === runId);
  if (!run) return runId.slice(0, 8);
  const prefix = run._experimentName ? run._experimentName + '/' : '';
  return prefix + run.run_name;
}
```

**Step 3: Update `renderSelectedRuns` default name**

In `renderSelectedRuns()` (line 654), the `name` variable also needs the prefix:

```js
const name = run ? (run._experimentName ? run._experimentName + '/' + run.run_name : run.run_name) : id.slice(0, 8);
```

**Step 4: Update `getFilteredRuns` to filter on display name**

Replace `getFilteredRuns` (lines 453-461) with:

```js
function getFilteredRuns() {
  const nf = $nameFilter.value.toLowerCase();
  const tf = $tagFilter.value.toLowerCase();
  return allRuns.filter(r => {
    if (nf) {
      const displayName = r._experimentName ? r._experimentName + '/' + r.run_name : r.run_name;
      if (!displayName.toLowerCase().includes(nf)) return false;
    }
    if (tf && !r.tags_list.some(t => t.toLowerCase().includes(tf))) return false;
    return true;
  });
}
```

**Step 5: Commit**

```bash
git add templates/index.html
git commit -m "feat: display experiment/run name in table and chart legends"
```

---

### Task 5: Multi-experiment — event wiring

**Files:**
- Modify: `templates/index.html` (lines ~1263-1273)

**Step 1: Update load button and Enter key**

Replace lines 1265-1266:

```js
$loadBtn.addEventListener("click", loadExperiment);
$exp.addEventListener("keydown", e => { if (e.key === "Enter") loadExperiment(); });
```

With:

```js
$loadBtn.addEventListener("click", () => {
  const name = $exp.value.trim();
  if (name) { addExperiment(name); $exp.value = ""; }
});
$exp.addEventListener("keydown", e => {
  if (e.key === "Enter") {
    const name = $exp.value.trim();
    if (name) { addExperiment(name); $exp.value = ""; }
  }
});
```

**Step 2: Add chip remove handler**

Add after the load button listeners:

```js
$expChips.addEventListener("click", e => {
  const x = e.target.closest(".exp-chip-x");
  if (!x) return;
  removeExperiment(x.dataset.exp);
});
```

**Step 3: Update refresh button**

Replace lines 1268-1273:

```js
$refreshBtn.addEventListener("click", async () => {
  showStatus('<span class="spinner"></span> Clearing cache…');
  Object.keys(runParamsCache).forEach(k => delete runParamsCache[k]);
  try { await api("/api/clear-cache", {method:"POST"}); } catch(e) {}
  loadExperiment();
});
```

With:

```js
$refreshBtn.addEventListener("click", async () => {
  showStatus('<span class="spinner"></span> Clearing cache…');
  Object.keys(runParamsCache).forEach(k => delete runParamsCache[k]);
  try { await api("/api/clear-cache", {method:"POST"}); } catch(e) {}
  const names = [...loadedExperiments.keys()];
  loadedExperiments.clear();
  for (const name of names) {
    await addExperiment(name);
  }
  if (!names.length) clearStatus();
});
```

**Step 4: Commit**

```bash
git add templates/index.html
git commit -m "feat: wire up add/remove experiment events and multi-refresh"
```

---

### Task 6: Multi-experiment — state persistence (share/export/import)

**Files:**
- Modify: `templates/index.html` (lines ~1344-1542)

**Step 1: Update `serializeViewState`**

Replace `serializeViewState` (lines 1344-1362) with:

```js
function serializeViewState() {
  const legends = {};
  customLegends.forEach((v, k) => { legends[k] = v; });
  return {
    v: 2,
    experiments: [...loadedExperiments.keys()],
    // Keep single experiment field for backward compat with v1
    experiment: [...loadedExperiments.keys()][0] || "",
    runs: [...selectedRunIds],
    metrics: [...selectedMetrics],
    legends,
    settings: {
      runType: (document.querySelector('input[name="runtype"]:checked') || {}).value || "training",
      xaxis: getXAxisMode(),
      yscale: getYScale(),
      smoothing: getSmoothingFactor(),
      colsPerRow: parseInt($colSlider.value, 10),
      grouping: $groupMode.value,
    },
  };
}
```

**Step 2: Update `restoreFromState`**

Replace `restoreFromState` (lines 1469-1518) with:

```js
async function restoreFromState(state) {
  const pending = applyViewState(state);
  if (!pending) return showInfo("Invalid view state.", "error");

  clearInfo();
  showStatus('<span class="spinner"></span> Restoring view…');
  $tableSection.style.display = "none";
  $metricSection.style.display = "none";
  $chartGrid.innerHTML = "";
  Object.keys(runParamsCache).forEach(k => delete runParamsCache[k]);
  loadedExperiments.clear();

  // Support v2 (experiments array) and v1 (single experiment)
  const expNames = state.experiments || (state.experiment ? [state.experiment] : []);

  try {
    for (const name of expNames) {
      const expData = await api(`/api/experiment?name=${encodeURIComponent(name)}`);
      const runsData = await api(`/api/runs?experiment_id=${encodeURIComponent(expData.experiment_id)}`);

      const shortName = getExpShortName(name);
      runsData.runs.forEach(r => {
        r._experimentName = shortName;
        r._fullExperimentName = name;
      });

      loadedExperiments.set(name, { id: expData.experiment_id, runs: runsData.runs });
    }

    rebuildAllRuns();
    renderExpChips();

    $expInfo.textContent = [...loadedExperiments.entries()]
      .map(([n, e]) => `${n} (${e.id})`).join(", ");

    // Restore selections — only keep IDs that still exist
    const validIds = new Set(allRuns.map(r => r.run_id));
    selectedRunIds = new Set(pending.pendingRuns.filter(id => validIds.has(id)));
    selectedMetrics = new Set(pending.pendingMetrics.filter(m => allMetricNames.includes(m)));

    renderTable();
    renderSelectedRuns();
    renderMetricList();
    renderMetricPills();
    $tableSection.style.display = allRuns.length ? "" : "none";
    $metricSection.style.display = allRuns.length ? "" : "none";
    clearStatus();

    if (selectedRunIds.size && selectedMetrics.size) {
      loadCharts();
    }
  } catch (err) {
    clearStatus();
    showInfo(err.message, "error");
  }
}
```

**Step 3: Update `applyViewState` validation**

In `applyViewState` (line 1365), change the guard from:

```js
if (!state || !state.experiment) return false;
```

To:

```js
if (!state || (!state.experiment && !(state.experiments && state.experiments.length))) return false;
```

Also remove the line `$exp.value = state.experiment;` (line 1368) since we no longer populate the input from state.

**Step 4: Update export filename**

In the export handler (line 1442), change:

```js
const expName = $exp.value.trim().replace(/\//g, "_") || "view";
```

To:

```js
const expName = [...loadedExperiments.keys()].map(n => n.replace(/\//g, "_")).join("+") || "view";
```

**Step 5: Update startup `init`**

Replace the startup block (lines 1522-1542). The default behavior should add the default experiment instead of calling `loadExperiment`:

```js
(async function init() {
  const hash = location.hash;
  if (hash.startsWith("#view=")) {
    const state = decodeViewHash(hash.slice(6));
    if (state) {
      restoreFromState(state);
      return;
    }
  }
  if (hasPreloadedState) {
    try {
      const state = await api("/api/preloaded-state");
      restoreFromState(state);
      return;
    } catch (err) {
      showInfo("Failed to load preloaded state: " + err.message, "error");
    }
  }
  const defaultName = `${defaultGroup}/${defaultExperiment}`;
  addExperiment(defaultName);
})();
```

**Step 6: Remove stale `experimentId` references**

Search for any remaining uses of `experimentId` and replace:
- `$expInfo.textContent = ...experimentId...` — already handled in `addExperiment`/`removeExperiment`
- Any other references should be removed since the experiments are now tracked via `loadedExperiments`

**Step 7: Commit**

```bash
git add templates/index.html
git commit -m "feat: update state persistence for multi-experiment support"
```

---

### Verification Checklist

After all tasks, manually verify:

1. **Scroll fix**: Load experiment, select runs + metrics, click Compare. Scroll to middle of charts. Wait 30s. Confirm scroll position stays.
2. **Multi-experiment add**: Type experiment name, press Enter. Chip appears. Type another, press Enter. Second chip appears. Table shows runs from both with `exp/runName` format.
3. **Multi-experiment remove**: Click x on a chip. That experiment's runs disappear from table and selections.
4. **Charts across experiments**: Select runs from different experiments, click Compare. Charts show runs from all experiments together.
5. **Refresh**: Click refresh with multiple experiments loaded. All reload correctly.
6. **Share URL**: Click Share, open URL in new tab. All experiments restore.
7. **Export/Import**: Export JSON, clear, import. All experiments restore.
