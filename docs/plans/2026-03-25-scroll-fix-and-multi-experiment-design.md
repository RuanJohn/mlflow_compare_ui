# Design: Fix Auto-Refresh Scroll + Multi-Experiment Support

Date: 2026-03-25

## Problem Statement

Two issues with the current UI:

1. **Auto-refresh scroll drift** — Every 30s poll destroys and recreates all charts, causing the page to scroll to the chart area despite scroll-position save/restore.
2. **Single-experiment limitation** — Users can only view one experiment at a time. They want to compare runs across multiple experiments on the same charts.

## Feature 1: In-Place Chart Data Update

### Root Cause

`pollMetrics()` calls `u.destroy()` on every uPlot instance, clears `$chartGrid.innerHTML`, and recreates all charts. The `scrollTop` restore fires before uPlot finishes laying out the new charts, causing scroll drift.

### Solution

Update chart data in-place using `uPlot.setData()` instead of destroying/recreating charts.

### Changes

**Extract data-building logic:**
- Factor out `buildChartData(data, metrics)` from `drawChartsFromData()`. This function returns the `uData` arrays for each chart group without touching the DOM.

**Update `pollMetrics()`:**
- After fetching fresh data, call `buildChartData()` to get new data arrays.
- If the number of chart groups matches existing instances, call `u.setData(newData)` on each.
- If the chart count changed (rare — metric set changed), fall back to full destroy/recreate with scroll save/restore.

**Remove scroll save/restore from the happy path:**
- No DOM teardown means no layout shift means no scroll drift.

## Feature 2: Multi-Experiment Support

### State Model

Replace single-experiment state with multi-experiment:

```
// Before
let experimentId = null;
let allRuns = [];

// After
const loadedExperiments = new Map();  // name -> {id, runs[]}
// allRuns derived: merged flat list with _experimentName on each run
```

### Topbar UI

Replace: `[input] [Load] [Refresh]`

With: `[input] [Add] [Refresh]`
Below input: row of experiment chips, each with "x" to remove.

Behavior:
- Type experiment name, press Enter or click Add.
- Frontend calls `/api/experiment` + `/api/runs` for that experiment.
- Experiment appears as a removable chip. Runs merge into the table.
- Adding a duplicate experiment is a no-op.
- Removing a chip removes that experiment's runs from `allRuns`, `selectedRunIds`, and refreshes the table/charts.
- Refresh reloads all loaded experiments.

### Runs Table

- "Run Name" column displays `{expShortName}/{runName}` where `expShortName` is the last path segment of the experiment name.
- No structural changes to the table.

### Metric Selector

- Shows the union of all metric keys across all loaded experiments.
- No changes to selection behavior.

### Charts

- `getRunLabel()` returns `{expShortName}/{runName}` for chart legends.
- Runs missing a metric are simply absent from that chart (already handled — API returns no data for those run/metric combos).

### Backend

No backend changes needed. The frontend:
1. Makes separate `/api/experiment` + `/api/runs` calls per experiment.
2. Merges runs client-side.
3. Sends all selected run IDs to `/api/metric-history` regardless of which experiment they belong to.

### Share URL / Export

- State includes `loadedExperiments` (list of experiment names) instead of single experiment name.
- Import restores by loading each experiment in sequence.
