"use strict";

const boardEl = document.getElementById("board");
const statusEl = document.getElementById("status");
const cells = [];

// Build the 9x9 grid of single-digit inputs.
for (let r = 0; r < 9; r++) {
  for (let c = 0; c < 9; c++) {
    const inp = document.createElement("input");
    inp.className = "cell";
    inp.maxLength = 1;
    inp.inputMode = "numeric";
    inp.autocomplete = "off";
    inp.dataset.index = r * 9 + c;

    inp.addEventListener("input", (e) => {
      // Keep only a single digit 1-9; entering a value clears "solved" styling.
      e.target.value = e.target.value.replace(/[^1-9]/g, "").slice(0, 1);
      e.target.classList.remove("solved");
    });
    inp.addEventListener("keydown", (e) => handleNav(e, r, c));

    boardEl.appendChild(inp);
    cells.push(inp);
  }
}

const idx = (r, c) => r * 9 + c;

// Arrow-key navigation between cells.
function handleNav(e, r, c) {
  const moves = { ArrowUp: [-1, 0], ArrowDown: [1, 0], ArrowLeft: [0, -1], ArrowRight: [0, 1] };
  const m = moves[e.key];
  if (!m) return;
  const nr = (r + m[0] + 9) % 9;
  const nc = (c + m[1] + 9) % 9;
  cells[idx(nr, nc)].focus();
  e.preventDefault();
}

function getGrid() {
  const g = [];
  for (let r = 0; r < 9; r++) {
    const row = [];
    for (let c = 0; c < 9; c++) {
      const v = cells[idx(r, c)].value;
      row.push(v ? parseInt(v, 10) : 0);
    }
    g.push(row);
  }
  return g;
}

// Replace the whole grid (used after scanning a photo). Clears solved styling.
function fillGrid(g) {
  for (let r = 0; r < 9; r++) {
    for (let c = 0; c < 9; c++) {
      const cell = cells[idx(r, c)];
      cell.value = g[r][c] ? String(g[r][c]) : "";
      cell.classList.remove("solved");
    }
  }
}

// Fill only the empty cells with the solution, styled as "solved".
function applySolution(g) {
  for (let r = 0; r < 9; r++) {
    for (let c = 0; c < 9; c++) {
      const cell = cells[idx(r, c)];
      if (!cell.value && g[r][c]) {
        cell.value = String(g[r][c]);
        cell.classList.add("solved");
      }
    }
  }
}

function setStatus(msg, kind = "") {
  statusEl.textContent = msg;
  statusEl.className = "status" + (kind ? " " + kind : "");
}

async function postJSON(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return { ok: res.ok, data: await res.json().catch(() => ({})) };
}

// --- Upload & scan ---
document.getElementById("file").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  setStatus("Scanning photo…", "busy");
  const form = new FormData();
  form.append("image", file);
  try {
    const res = await fetch("/api/scan", { method: "POST", body: form });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      setStatus(data.error || "Scan failed.", "error");
      return;
    }
    fillGrid(data.grid);
    setStatus("Detected the grid — review any misreads, then Solve.", "ok");
  } catch {
    setStatus("Could not reach the server.", "error");
  } finally {
    e.target.value = ""; // allow re-uploading the same file
  }
});

// --- Solve ---
document.getElementById("solve").addEventListener("click", async () => {
  setStatus("Solving…", "busy");
  try {
    const { ok, data } = await postJSON("/api/solve", { grid: getGrid() });
    if (!ok) {
      setStatus(data.error || "Could not solve.", "error");
      return;
    }
    applySolution(data.solution);
    setStatus("Solved!", "ok");
  } catch {
    setStatus("Could not reach the server.", "error");
  }
});

// --- Clear ---
document.getElementById("clear").addEventListener("click", () => {
  for (const cell of cells) {
    cell.value = "";
    cell.classList.remove("solved");
  }
  setStatus("");
  cells[0].focus();
});
