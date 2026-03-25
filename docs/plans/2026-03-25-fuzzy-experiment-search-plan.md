# Fuzzy Experiment Search Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add experiment name autocomplete with dropdown + ghost text so users can discover and select experiments by typing a substring.

**Architecture:** New `/api/experiments` backend endpoint returns all experiment names. Client fetches on load + every 10 min, caches in-memory. On input, filters with case-insensitive `includes()`, renders dropdown (max 10) + ghost text overlay. Self-contained, scoped to `#expInput`.

**Tech Stack:** Flask (Python), vanilla JS, existing CSS variables

---

### Task 1: Backend — add `/api/experiments` endpoint

**Files:**
- Modify: `mlflow_utils.py` (add `list_experiments` function)
- Modify: `app.py:25-32` (add import) and `app.py:92-102` (add route before `/api/experiment`)

**Step 1: Write `list_experiments` in mlflow_utils.py**

Add after `get_experiment_by_name` (after line 64):

```python
def list_experiments() -> list[str]:
    """Return sorted list of all experiment names."""
    client = get_client()
    experiments = client.search_experiments()
    return sorted(exp.name for exp in experiments if exp.lifecycle_stage == "active")
```

**Step 2: Add import in app.py**

In `app.py:25-32`, add `list_experiments` to the import from `mlflow_utils`.

**Step 3: Add the route in app.py**

Add before the existing `/api/experiment` route (before line 94):

```python
@app.route("/api/experiments")
def api_experiments():
    names = list_experiments()
    return json_response({"experiments": names})
```

**Step 4: Test manually**

Run: `curl http://localhost:5050/api/experiments`
Expected: `{"experiments":["group1/exp-a","group1/exp-b",...]}`

**Step 5: Commit**

```bash
git add mlflow_utils.py app.py
git commit -m "feat: add /api/experiments endpoint for experiment name list"
```

---

### Task 2: Frontend CSS — add styles for autocomplete dropdown and ghost text

**Files:**
- Modify: `templates/index.html:45-47` (after `.exp-input-area` styles)

**Step 1: Add CSS rules**

Add after line 47 (after `.exp-input-area input[type=text]:focus` rule):

```css
.exp-input-wrap{position:relative;flex:1}
.exp-input-wrap input[type=text]{width:100%;padding:6px 10px;border:1px solid var(--border);border-radius:var(--radius);background:var(--bg);color:var(--fg);font-size:13px}
.exp-input-wrap input[type=text]:focus{outline:none;border-color:var(--accent)}
.exp-ghost{position:absolute;left:0;top:0;right:0;bottom:0;padding:6px 10px;font-size:13px;font-family:inherit;color:var(--fg2);opacity:0.4;pointer-events:none;white-space:nowrap;overflow:hidden}
.exp-dropdown{position:absolute;top:100%;left:0;right:0;max-height:240px;overflow-y:auto;background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius);margin-top:2px;z-index:200;display:none}
.exp-dropdown-item{padding:6px 10px;font-size:12px;cursor:pointer;color:var(--fg)}
.exp-dropdown-item:hover,.exp-dropdown-item.active{background:var(--accent);color:#fff}
```

**Step 2: Commit**

```bash
git add templates/index.html
git commit -m "feat: add CSS for experiment autocomplete dropdown and ghost text"
```

---

### Task 3: Frontend HTML — restructure experiment input with wrapper, ghost span, and dropdown div

**Files:**
- Modify: `templates/index.html:158-160` (the `.exp-input-area` div contents)

**Step 1: Replace the input area contents**

Replace lines 158-160:
```html
  <div class="exp-input-area">
    <input type="text" id="expInput" placeholder="Experiment name, e.g. group/experiment">
    <button class="btn btn-primary" id="loadBtn">Add</button>
```

With:
```html
  <div class="exp-input-area">
    <div class="exp-input-wrap">
      <span id="expGhost" class="exp-ghost"></span>
      <input type="text" id="expInput" placeholder="Experiment name, e.g. group/experiment" autocomplete="off">
      <div id="expDropdown" class="exp-dropdown"></div>
    </div>
    <button class="btn btn-primary" id="loadBtn">Add</button>
```

Note: The existing `.exp-input-area` CSS `flex:1` and `input flex:1` rules should be updated — remove the `flex:1` from the input rule (line 46) since `.exp-input-wrap` now takes that role.

**Step 2: Commit**

```bash
git add templates/index.html
git commit -m "feat: add ghost text span and dropdown div to experiment input"
```

---

### Task 4: Frontend JS — fetch experiment names on load + periodic refresh

**Files:**
- Modify: `templates/index.html` (JS section, after DOM refs ~line 379)

**Step 1: Add state variable and DOM refs**

After the existing DOM refs block (around line 379), add:

```javascript
// Experiment autocomplete state
let allExperimentNames = [];
const $expGhost = document.getElementById("expGhost");
const $expDropdown = document.getElementById("expDropdown");
```

**Step 2: Add fetch function**

Add after the state variable:

```javascript
async function fetchExperimentNames() {
  try {
    const data = await api("/api/experiments");
    allExperimentNames = data.experiments || [];
  } catch (e) {
    console.warn("Failed to fetch experiment names:", e);
  }
}
```

**Step 3: Add init call and interval**

Add near the bottom of the JS, in the initialization section (before the final `</script>` or near the existing init code around line 1660):

```javascript
// Experiment autocomplete: fetch names on load + refresh every 10 min
fetchExperimentNames();
setInterval(fetchExperimentNames, 10 * 60 * 1000);
```

**Step 4: Commit**

```bash
git add templates/index.html
git commit -m "feat: fetch experiment names on load with 10-min refresh"
```

---

### Task 5: Frontend JS — autocomplete input handler with dropdown + ghost text

**Files:**
- Modify: `templates/index.html` (JS section, after fetchExperimentNames)

**Step 1: Add the autocomplete logic**

Add after `fetchExperimentNames`:

```javascript
let expDropdownIdx = -1;

function updateExpAutocomplete() {
  const q = $exp.value.toLowerCase().trim();
  if (!q) {
    $expDropdown.style.display = "none";
    $expDropdown.innerHTML = "";
    $expGhost.textContent = "";
    expDropdownIdx = -1;
    return;
  }
  const loaded = new Set(loadedExperiments.keys());
  const matches = allExperimentNames
    .filter(n => n.toLowerCase().includes(q) && !loaded.has(n))
    .slice(0, 10);

  if (!matches.length) {
    $expDropdown.style.display = "none";
    $expDropdown.innerHTML = "";
    $expGhost.textContent = "";
    expDropdownIdx = -1;
    return;
  }

  expDropdownIdx = -1;
  $expDropdown.style.display = "";
  $expDropdown.innerHTML = matches.map((n, i) =>
    `<div class="exp-dropdown-item" data-name="${esc(n)}">${esc(n)}</div>`
  ).join("");

  // Ghost text: show the first match, aligning the suffix after what the user typed
  setExpGhost(matches[0]);
}

function setExpGhost(match) {
  if (!match) { $expGhost.textContent = ""; return; }
  const raw = $exp.value;
  const idx = match.toLowerCase().indexOf(raw.toLowerCase());
  if (idx === 0) {
    // Match starts at beginning — show full match as ghost with user's part invisible
    $expGhost.textContent = raw + match.slice(raw.length);
  } else {
    // Match is in the middle — just show the match name as ghost
    $expGhost.textContent = match;
  }
}
```

**Step 2: Wire up the input event**

Find the existing `$exp.addEventListener("keydown"` block (line 1370). Add *before* it:

```javascript
$exp.addEventListener("input", updateExpAutocomplete);
```

**Step 3: Commit**

```bash
git add templates/index.html
git commit -m "feat: add experiment autocomplete filtering and ghost text"
```

---

### Task 6: Frontend JS — keyboard navigation (arrows, Tab, Enter, Escape)

**Files:**
- Modify: `templates/index.html` (replace existing `$exp` keydown handler at line 1370-1375)

**Step 1: Replace the existing keydown handler**

Replace the existing handler (lines 1370-1375):
```javascript
$exp.addEventListener("keydown", e => {
  if (e.key === "Enter") {
    const name = $exp.value.trim();
    if (name) { addExperiment(name); }
  }
});
```

With:
```javascript
$exp.addEventListener("keydown", e => {
  const items = $expDropdown.querySelectorAll(".exp-dropdown-item");
  const open = $expDropdown.style.display !== "none" && items.length > 0;

  if (e.key === "ArrowDown" && open) {
    e.preventDefault();
    expDropdownIdx = Math.min(expDropdownIdx + 1, items.length - 1);
    items.forEach((el, i) => el.classList.toggle("active", i === expDropdownIdx));
    setExpGhost(items[expDropdownIdx].dataset.name);
    return;
  }
  if (e.key === "ArrowUp" && open) {
    e.preventDefault();
    expDropdownIdx = Math.max(expDropdownIdx - 1, 0);
    items.forEach((el, i) => el.classList.toggle("active", i === expDropdownIdx));
    setExpGhost(items[expDropdownIdx].dataset.name);
    return;
  }
  if (e.key === "Tab" && open) {
    e.preventDefault();
    const match = expDropdownIdx >= 0 ? items[expDropdownIdx].dataset.name : items[0].dataset.name;
    $exp.value = match;
    $expDropdown.style.display = "none";
    $expDropdown.innerHTML = "";
    $expGhost.textContent = "";
    expDropdownIdx = -1;
    return;
  }
  if (e.key === "Enter") {
    if (open && expDropdownIdx >= 0) {
      e.preventDefault();
      const name = items[expDropdownIdx].dataset.name;
      $exp.value = name;
      $expDropdown.style.display = "none";
      $expDropdown.innerHTML = "";
      $expGhost.textContent = "";
      expDropdownIdx = -1;
      addExperiment(name);
    } else {
      const name = $exp.value.trim();
      if (name) {
        $expDropdown.style.display = "none";
        $expDropdown.innerHTML = "";
        $expGhost.textContent = "";
        expDropdownIdx = -1;
        addExperiment(name);
      }
    }
    return;
  }
  if (e.key === "Escape") {
    $expDropdown.style.display = "none";
    $expDropdown.innerHTML = "";
    $expGhost.textContent = "";
    expDropdownIdx = -1;
    return;
  }
});
```

**Step 2: Add click handler for dropdown items**

Add after the keydown handler:

```javascript
$expDropdown.addEventListener("click", e => {
  const item = e.target.closest(".exp-dropdown-item");
  if (!item) return;
  const name = item.dataset.name;
  $exp.value = name;
  $expDropdown.style.display = "none";
  $expDropdown.innerHTML = "";
  $expGhost.textContent = "";
  expDropdownIdx = -1;
  addExperiment(name);
});
```

**Step 3: Add blur/outside-click to close dropdown**

Add after the click handler:

```javascript
document.addEventListener("click", e => {
  if (!$exp.contains(e.target) && !$expDropdown.contains(e.target)) {
    $expDropdown.style.display = "none";
    $expGhost.textContent = "";
    expDropdownIdx = -1;
  }
});
```

**Step 4: Commit**

```bash
git add templates/index.html
git commit -m "feat: add keyboard navigation and click handling for experiment autocomplete"
```

---

### Task 7: Clean up — remove redundant CSS and verify

**Files:**
- Modify: `templates/index.html:46` (remove old input flex rule now handled by wrapper)

**Step 1: Update CSS**

Line 46 currently:
```css
.exp-input-area input[type=text]{flex:1;padding:6px 10px;border:1px solid var(--border);border-radius:var(--radius);background:var(--bg);color:var(--fg);font-size:13px}
```

Remove this rule entirely — the `.exp-input-wrap` and `.exp-input-wrap input` rules from Task 2 replace it.

**Step 2: Manual testing checklist**

- [ ] Page loads without errors, experiment list fetched in Network tab
- [ ] Typing in input shows dropdown with matching experiments
- [ ] Ghost text appears for the top match
- [ ] Arrow keys navigate the dropdown, ghost text updates
- [ ] Tab accepts the ghost text into input
- [ ] Enter with highlighted item adds the experiment
- [ ] Enter with no dropdown open adds typed name (existing behavior)
- [ ] Escape closes dropdown
- [ ] Clicking outside closes dropdown
- [ ] Already-loaded experiments don't appear in suggestions
- [ ] Clicking a dropdown item adds the experiment

**Step 3: Commit**

```bash
git add templates/index.html
git commit -m "feat: clean up CSS after experiment autocomplete refactor"
```
