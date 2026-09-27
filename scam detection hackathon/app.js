const API = "http://127.0.0.1:8000"; 
const state = {
  status: null,
  schemas: {}, // key -> {feature_names, class_names, example_values}
};

// ---------------------------------------------------------------- tabs ---

document.querySelectorAll(".tab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(b => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach(p => p.classList.remove("active"));
    btn.classList.add("active");
    document.getElementById(`panel-${btn.dataset.tab}`).classList.add("active");
  });
});

// -------------------------------------------------------------- status ---

async function refreshStatus() {
  try {
    const res = await fetch(`${API}/api/status`);
    const data = await res.json();
    state.status = data;
    renderStatus(data);
    for (const key of ["bank", "creditcard"]) {
      if (data[key]?.model_trained) loadSchema(key);
    }
  } catch (e) {
    document.getElementById("statusLabel").textContent = "backend unreachable";
  }
}

function renderStatus(data) {
  const trained = Object.values(data).filter(d => d.model_trained).length;
  const total = Object.keys(data).length;
  const dot = document.querySelector("#statusToggle .dot");
  const label = document.getElementById("statusLabel");

  if (trained === total) { dot.className = "dot ready"; label.textContent = `${trained}/${total} models trained`; }
  else if (trained === 0) { dot.className = "dot"; label.textContent = `0/${total} models trained`; }
  else { dot.className = "dot partial"; label.textContent = `${trained}/${total} models trained`; }

  const grid = document.getElementById("statusGrid");
  grid.innerHTML = "";
  const names = { sms: "SMS scam", url: "URL scam", bank: "Bank fraud", creditcard: "Credit card fraud" };
  for (const [key, info] of Object.entries(data)) {
    const card = document.createElement("div");
    card.className = "status-card";
    card.innerHTML = `
      <span class="name">${names[key] || key}</span>
      <span class="row ${info.dataset_present ? 'ok' : 'no'}">dataset<span>${info.dataset_present ? 'found' : 'missing'}</span></span>
      <span class="row ${info.model_trained ? 'ok' : 'no'}">model<span>${info.model_trained ? 'trained' : 'not trained'}</span></span>
    `;
    grid.appendChild(card);
  }
}

document.getElementById("statusToggle").addEventListener("click", () => {
  document.getElementById("statusPanel").classList.toggle("hidden");
});

document.getElementById("trainAllBtn").addEventListener("click", async () => {
  const btn = document.getElementById("trainAllBtn");
  const log = document.getElementById("trainLog");
  btn.disabled = true;
  log.textContent = "training… this can take a few minutes depending on dataset size";
  try {
    const res = await fetch(`${API}/api/train`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    const data = await res.json();
    log.textContent = data.returncode === 0 ? "training complete" : "training finished with errors — check backend logs";
  } catch (e) {
    log.textContent = "training request failed: " + e.message;
  }
  btn.disabled = false;
  refreshStatus();
});

// --------------------------------------------------------- dynamic form --

async function loadSchema(key) {
  try {
    const res = await fetch(`${API}/api/schema/${key}`);
    if (!res.ok) return;
    const schema = await res.json();
    state.schemas[key] = schema;
    renderForm(key, schema);
  } catch (e) { /* not trained yet */ }
}

function renderForm(key, schema) {
  const container = document.getElementById(`formFields-${key}`);
  container.innerHTML = "";
  for (const feat of schema.feature_names) {
    const wrap = document.createElement("div");
    wrap.className = "field";
    const example = schema.example_values[feat] ?? 0;
    wrap.innerHTML = `
      <label for="f-${key}-${feat}">${feat}</label>
      <input type="number" step="any" id="f-${key}-${feat}" data-feat="${feat}" value="${example}" />
    `;
    container.appendChild(wrap);
  }
}

// ------------------------------------------------------------- predict ---

document.querySelectorAll('[data-action="predict"]').forEach(btn => {
  btn.addEventListener("click", () => runPredict(btn.dataset.key));
});

async function runPredict(key) {
  const outEl = document.getElementById(`output-${key}`);
  outEl.innerHTML = `<div class="empty-state">Analyzing…</div>`;

  let body;
  if (key === "sms") {
    const text = document.getElementById("smsInput").value.trim();
    if (!text) { outEl.innerHTML = `<div class="empty-state">Enter a message first.</div>`; return; }
    body = { text };
  } else if (key === "url") {
    const url = document.getElementById("urlInput").value.trim();
    if (!url) { outEl.innerHTML = `<div class="empty-state">Enter a URL first.</div>`; return; }
    body = { url };
  } else {
    const schema = state.schemas[key];
    if (!schema) { outEl.innerHTML = `<div class="empty-state">Train this model first (see status panel).</div>`; return; }
    const features = {};
    schema.feature_names.forEach(f => {
      const input = document.getElementById(`f-${key}-${f}`);
      features[f] = parseFloat(input.value) || 0;
    });
    body = { features };
  }

  try {
    const res = await fetch(`${API}/api/predict/${key}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!res.ok) {
      outEl.innerHTML = `<div class="error-box">${data.detail || "Request failed"}</div>`;
      return;
    }
    renderResult(outEl, data);
  } catch (e) {
    outEl.innerHTML = `<div class="error-box">Request failed: ${e.message}</div>`;
  }
}

function riskClass(prediction) {
  const p = prediction.toLowerCase();
  return (p.includes("scam") || p.includes("fraud") || p.includes("malicious")) ? "risk" : "safe";
}

function renderResult(outEl, data) {
  const cls = riskClass(data.prediction);

  let html = `
    <div class="verdict ${cls}">
      <span class="verdict-label">${data.prediction}</span>
      <span class="verdict-confidence">confidence ${(data.confidence * 100).toFixed(1)}%</span>
    </div>
    <div class="prob-bars">
      ${Object.entries(data.probabilities).map(([name, p]) => `
        <div class="prob-row">
          <span class="prob-name">${name}</span>
          <div class="prob-track"><div class="prob-fill" style="width:${(p * 100).toFixed(1)}%"></div></div>
          <span class="prob-value">${(p * 100).toFixed(1)}%</span>
        </div>
      `).join("")}
    </div>
  `;

  if (data.lime && data.lime.length) {
    const maxAbs = Math.max(...data.lime.map(f => Math.abs(f.weight)), 0.001);
    html += `
      <div class="section-title">LIME feature attribution</div>
      <div class="lime-bars">
        ${data.lime.map(f => {
          const pct = (Math.abs(f.weight) / maxAbs) * 50;
          const dir = f.weight >= 0 ? "pos" : "neg";
          return `
            <div class="lime-row">
              <span class="lime-feature" title="${f.feature}">${f.feature}</span>
              <div class="lime-track">
                <div class="lime-mid"></div>
                <div class="lime-fill ${dir}" style="width:${pct}%"></div>
              </div>
              <span class="lime-value">${f.weight.toFixed(3)}</span>
            </div>
          `;
        }).join("")}
      </div>
    `;
  }

  html += `<div class="section-title">Anchor rule set</div>`;
  const anchor = data.anchor;
  if (anchor && anchor.available) {
    const items = anchor.anchor_words || anchor.anchor_rules || [];
    html += `
      <div class="anchor-box">
        <div class="anchor-chips">
          ${items.map(i => `<span class="chip">${i}</span>`).join("") || "<span class='anchor-unavailable'>No anchor conditions met the threshold.</span>"}
        </div>
        <div class="anchor-meta">precision ${anchor.precision}&nbsp;&nbsp;&nbsp;coverage ${anchor.coverage}</div>
      </div>
    `;
  } else {
    html += `<div class="anchor-unavailable">Anchor explanation unavailable${anchor?.reason ? ": " + anchor.reason : ""}.</div>`;
  }

  if (data.extracted_features) {
    html += `
      <div class="section-title">Extracted URL features</div>
      <div class="dynamic-form" style="max-height:none;">
        ${Object.entries(data.extracted_features).map(([k, v]) => `
          <div class="field"><label>${k}</label><span style="font-family:var(--mono);font-size:0.82rem;">${v}</span></div>
        `).join("")}
      </div>
    `;
  }

  outEl.innerHTML = html;
}

// -------------------------------------------------------------- init -----

refreshStatus();